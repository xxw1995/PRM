import time
import os
import numpy as np
from functools import reduce
import torch
from tensorboardX import SummaryWriter
from mat.agents.qwen_lora_agent import QwenLoRAgent
from mat.models.ms_prm import MSProcessRM
from mat.models.qwen_prm import QwenProcessRM
from mat.utils.language_buffer import LanguageBuffer

# 三种训练器APPOTrainer, TPPOTrainer, GRPOTrainer，这些模块在项目中负责编制不同的任务。
from mat.trainers.llm_trainer_appo import APPOTrainer
from mat.trainers.llm_trainer_tppo import TPPOTrainer
from mat.trainers.llm_trainer_grpo import GRPOTrainer


# 将torch tensor转换为numpy数组
def _t2n(x):
    return x.detach().cpu().numpy()

class MathRunner:
    # 初始化环境、参数和资源：
    def __init__(self, config):
        self.num_agents = config['num_agents'] # 
        self.all_args = config['all_args']
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.num_env_steps = self.all_args.num_env_steps          # 环境步骤数
        self.episode_length = self.all_args.episode_length        # 回合长度
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval            # 日志间隔
        self.eval_interval = self.all_args.eval_interval
        self.save_interval = self.all_args.save_interval
        self.algo = self.all_args.algorithm_name
        self.prm_type = self.all_args.prm_type                    # 策略奖励模型类型prm_type

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models/')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        # 使用QwenLoRAgent初始化agent
        self.agent = QwenLoRAgent(self.all_args.model_name_or_path, self.all_args.max_new_tokens, self.algo)
        self.buffer = LanguageBuffer(self.all_args, self.num_agents, self.agent.tokenizer.pad_token_id)
       
       # 根据策略奖励模型类型（prm_type）实例化相应的策略奖励模型（MSProcessRM 或 QwenProcessRM）
        if self.prm_type == "MS":
            self.prm = MSProcessRM(self.all_args)
        elif self.prm_type == "Qwen":
            self.prm = QwenProcessRM(self.all_args)
        else:
            raise NotImplementedError

        # 根据算法名称（algo）实例化对应的训练器
        if self.algo == "APPO":
            self.trainer = APPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo == "TPPO":
            self.trainer = TPPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo == "GRPO":
            self.trainer = GRPOTrainer(self.all_args, self.agent, self.num_agents)
        else:
            raise NotImplementedError

    def run(self):
        obs = self.envs.reset()
        self.buffer.obs[0] = obs.copy()
        # 计算总的回合数
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        episodic_returns = []
        for episode in range(episodes):
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads  
            for step in range(self.episode_length):
                # 使用代理决定动作
                values, actions, action_tokens, log_probs = self.collect(step)
                
                # 通过奖励模型计算奖励
                rewards = self.prm.get_reward(obs, actions)

                # 执行一步环境推进，获取新观测、奖励和终止标识。
                obs, fake_rewards, dones, infos = self.envs.step(actions)

                # 更新数据缓冲区。
                data = obs, rewards, dones, values, actions, action_tokens, log_probs
                self.insert(data)
                # 对于每个完成的轨迹，记录回报。
                for i in range(self.n_rollout_threads):
                    if dones[i, 0]:
                        episodic_returns.append(rewards[i, 0])

            # 根据算法计算回报并优化模型。
            self.before_update()
            train_infos = self.trainer.train(self.buffer)      
            self.buffer.after_update()
            
            # 保存模型
            if (episode == episodes - 1 or episode % self.save_interval == 0):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                print("total_num_steps: ", total_num_steps)
                print("average_step_rewards: ", np.mean(self.buffer.rewards))
                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                train_infos["average_currect_rate"] = np.mean(episodic_returns)
                self.log_infos(train_infos, total_num_steps)
                episodic_returns = []

            # eval
            # if self.all_args.use_eval and episode % self.eval_interval == 0:
            #     self.eval(total_num_steps)
        

    # 不更新梯度
    @torch.no_grad()
    def collect(self, step):
        """
        用于从agent中收集「行为数据」
        """
        # infer_for_rollout用于在给定的观测值obs下执行动作推理
        # 使用当前时间步step的观测数据obs来执行动作推理
        behaviour_data = self.agent.infer_for_rollout(np.concatenate(self.buffer.obs[step]))

        # 分解成这4个值 动作、动作token、价值评估、动作的对数概率
        actions, action_tokens, values, log_probs = behaviour_data
        
        # [self.envs, agents]
        # 将动作、价值等数据数组按并行环境的数量分割。
        # np.split(data, self.n_rollout_threads) 将每个数组按 self.n_rollout_threads 等分，形成一个形如 [self.envs, agents] 的结构，这里 self.envs 应该代表并行环境的数量。
        values = np.array(np.split(values, self.n_rollout_threads))
        actions = np.array(np.split(actions, self.n_rollout_threads))
        action_tokens = np.array(np.split(action_tokens, self.n_rollout_threads))
        log_probs = np.array(np.split(log_probs, self.n_rollout_threads))

        return values, actions, action_tokens, log_probs

    def insert(self,data):
        obs, rewards, dones, values, actions, action_tokens, log_probs = data

        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents), dtype=np.float32)

        if self.algo == "APPO" or self.algo == "GRPO":
            self.buffer.insert_appo(obs, actions, values, rewards, masks, action_tokens, log_probs)
        elif self.algo == "TPPO":
            self.buffer.insert_tppo(obs, actions, values, rewards, masks, action_tokens, log_probs)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def before_update(self):
        """Calculate returns for the collected data."""
        next_values = self.agent.get_next_values(np.concatenate(self.buffer.obs[-1]))
        next_values = np.array(np.split(next_values, self.n_rollout_threads))
        if self.algo == "APPO":
            self.buffer.batch_process_appo(next_values)
        elif self.algo == "TPPO":
            self.buffer.batch_process_tppo(next_values)
        elif self.algo == "GRPO":
            self.buffer.batch_process_grpo()
        else:
            raise NotImplementedError

    def log_infos(self, infos, total_num_steps):
        for k, v in infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episodic_returns = []

        eval_obs = self.eval_envs.reset()
        while True:
            eval_actions, _ = self.agent.get_actions(np.concatenate(eval_obs))
            eval_actions = np.array(np.split(eval_actions, self.n_eval_rollout_threads))
            eval_obs, eval_rewards, eval_dones, _ = self.eval_envs.step(eval_actions)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones[eval_i, 0]:
                    eval_episode += 1
                    eval_episodic_returns.append(eval_rewards[eval_i])

            if eval_episode >= self.all_args.eval_episodes:
                eval_currect_rate = np.mean(eval_episodic_returns)
                env_infos = {'eval_currect_rate': eval_currect_rate}     
                print("total_num_steps: ", total_num_steps)
                print("eval_currect_rate is {}.".format(eval_currect_rate))           
                self.log_infos(env_infos, total_num_steps)
                break
                
    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.agent.save(self.save_dir, episode)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.agent.restore(model_dir)


