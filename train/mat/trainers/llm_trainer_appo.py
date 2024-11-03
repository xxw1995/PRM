import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mat.utils.util import get_gard_norm, huber_loss, mse_loss


# xxw core 
class APPOTrainer:
    def __init__(self, args, agent, num_agents):
        self.tpdv = dict(dtype=torch.float32, device=torch.device("cuda:0"))
        self.agent = agent

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.entropy_coef = args.entropy_coef
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.gradient_cp_steps = args.gradient_cp_steps


        self.policy_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.agent.actor.parameters()), lr=self.lr, eps=1e-5, weight_decay=0)
        self.critic_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.agent.critic.parameters()), lr=self.critic_lr, eps=1e-5)

    def cal_policy_loss(self, log_prob_infer, log_prob_batch, advantages_batch, entropy):
        
        log_ratio = log_prob_infer - log_prob_batch
        imp_weights = torch.exp(log_ratio)
        
        approx_kl = ((imp_weights - 1) - log_ratio).mean()
        
        surr1 = -torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        surr2 = -imp_weights * advantages_batch
        surr = torch.max(surr1, surr2)
        policy_loss = surr.mean() - self.entropy_coef * entropy.mean()
        return policy_loss, approx_kl
        
    # 计算loss 
    def cal_value_loss(self, values_infer, value_preds_batch, return_batch):
        # 对获取的value做clip
        value_pred_clipped = value_preds_batch + (values_infer - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        # 两种差值
        error_clipped = return_batch - value_pred_clipped
        error_unclipped = return_batch - values_infer
        # huber_loss
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_unclipped = huber_loss(error_unclipped, self.huber_delta)
        # mse loss
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_unclipped = mse_loss(error_unclipped)
        # 取最大再取平均
        value_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
        return value_loss * self.value_loss_coef
    
    # xxw core
    def ppo_update(self, sample):
        # 从输入 sample 中解包相关的信息，其中包括：观测、动作、log概率、价值预测、回报、优势和动作token。
        obs_batch, action_batch, log_prob_batch, \
            value_preds_batch, return_batch, advantages_batch, action_tokens_batch = sample
        # 获取tensor
        log_prob_batch = torch.from_numpy(log_prob_batch).to("cuda")
        value_preds_batch = torch.from_numpy(value_preds_batch).to("cuda")
        return_batch = torch.from_numpy(return_batch).to("cuda")
        advantages_batch = torch.from_numpy(advantages_batch).to("cuda")
        action_tokens_batch = torch.from_numpy(action_tokens_batch).to("cuda")
        batch_size = obs_batch.shape[0]
        
        """critic update -> 价值网络（critic）更新"""
        # 通过调用self.agent.get_action_values()，使用模型计算当前的状态价值values_infer。
        values_infer = self.agent.get_action_values(np.concatenate(obs_batch))
        # 利用view做形状变化
        values_infer = values_infer.view(batch_size, -1)
        # 获取value_loss 
        value_loss = self.cal_value_loss(values_infer, value_preds_batch, return_batch)
        
        # 反向传播更新参数
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        # 根据是否梯度裁剪来获取梯度
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.agent.critic.parameters())
        self.critic_optimizer.step()
        value_loss = value_loss.item()
        self.critic_optimizer.zero_grad()
        critic_grad_norm = critic_grad_norm.item()

        """policy update -> 策略网络（policy）更新"""
        self.policy_optimizer.zero_grad()
        cp_batch_size = int(batch_size // self.gradient_cp_steps)
        total_approx_kl = 0
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            # log_prob_infer：用于策略梯度更新中作为目标函数计算的一部分。
            # entropies：可用于促进策略探索，增加策略的随机性以帮助空间探索更优策略，并在训练过程中用作正则项
            log_prob_infer, entropy = self.agent.infer_for_action_update(np.concatenate(obs_batch[start:end]), 
                                                                         action_tokens_batch[start:end].view(-1, action_tokens_batch.shape[-1]))
            # 变换形状
            log_prob_infer = log_prob_infer.view(obs_batch[start:end].shape[0], -1)
            # 计算优势函数
            cp_adv_batch = advantages_batch[start:end]
            cp_adv_batch = (cp_adv_batch - cp_adv_batch.mean()) / (cp_adv_batch.std() + 1e-8)
            
            entropy = entropy.view(obs_batch[start:end].shape[0], -1)
            # 调用 self.cal_policy_loss()，结合log概率、已知log概率和标准化的优势函数计算损失
            # 并累加近似的KL散度，获得total_approx_kl
            policy_loss, approx_kl = self.cal_policy_loss(log_prob_infer, log_prob_batch[start:end], cp_adv_batch, entropy)
            total_approx_kl += approx_kl / self.gradient_cp_steps
           
            # 获取最终的policy_loss
            policy_loss /= self.gradient_cp_steps
            policy_loss.backward()

        # 策略更新停止条件
        # 若total_approx_kl的值超过0.02，则认为策略更新使得新旧策略之间的差异过大。
        if total_approx_kl > 0.02:
            # 清零策略优化器的梯度，这意味着放弃当前的梯度更新，避免对策略网络进行过大步长的更新。
            self.policy_optimizer.zero_grad()
            # 提前终止策略更新过程，返回当前的价值损失、价值网络的梯度范数，并为策略损失和策略网络的梯度范数返回零，因为这次更新被放弃。
            return value_loss, critic_grad_norm, 0, 0
       
        # 表示策略的变动在可接受范围内
        # 对策略网络的梯度裁剪，防止梯度爆炸。
        policy_grad_norm = nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
        # 进行优化步骤，应用梯度更新策略网络。
        self.policy_optimizer.step()
        policy_loss = policy_loss.item()
        self.policy_optimizer.zero_grad()
        policy_grad_norm = policy_grad_norm.item()
        # 返回：当前的价值损失、价值网络的梯度范数，策略损失、策略网络的梯度范数
        return value_loss, critic_grad_norm, policy_loss, policy_grad_norm

    # xxw train 
    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info['value_loss'] = 0
        train_info['value_grad_norm'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_grad_norm'] = 0

        update_time = 0
        for _ in range(self.ppo_epoch):
            data_generator = buffer.appo_sampler(self.num_mini_batch)
            for sample in data_generator:
                # xxw core
                # ppo_update
                value_loss, value_grad_norm, policy_loss, policy_grad_norm = self.ppo_update(sample)
                train_info['value_loss'] += value_loss
                train_info['value_grad_norm'] += value_grad_norm
                train_info['policy_loss'] += policy_loss
                train_info['policy_grad_norm'] += policy_grad_norm
                update_time += 1

        for k in train_info.keys():
            train_info[k] /= update_time
 
        return train_info

    def prep_training(self):
        self.agent.actor().train()
        self.agent.critic().train()

    def prep_rollout(self):
        self.agent.actor().eval()
        self.agent.critic().eval()
