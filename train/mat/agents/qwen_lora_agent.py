import sys
from time import sleep
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions.categorical import Categorical
import gc
import random
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import os
from peft import PeftModel
# APPOCritic
from mat.models.critic import APPOCritic, TPPOCritic

# 一个qwen+lora的model充当代理 
class QwenLoRAgent:
    def __init__(self, model_name, max_new_tokens, algo, load_path=None):
        self.device = "cuda"
        self.algo = algo
        # 指定tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left', trust_remote_code=True)
        self.tokenizer.pad_token_id = 151655 # "<|image_pad|>"
        # 获取model
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                               torch_dtype=torch.float16,
                                                               device_map="auto",
                                                               trust_remote_code=True)
        self.base_model.half().to(self.device)
        self.max_new_tokens = max_new_tokens
        
        if load_path is None:
            # 加载lora model，充当actor
            self.actor = self._init_actor().to(self.device)
            if self.algo != "GRPO":
                self.critic = self._init_critic().to(self.device)
        else:
            self.load(load_path)

    # 用lora包裹住base mode
    def _init_actor(self, lora_weights = None):
        if lora_weights is None:
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj",],
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(self.base_model, config)

            model.print_trainable_parameters()

            old_state_dict = model.state_dict
            model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(
                    self, old_state_dict()
                )
            ).__get__(model, type(model))
        else:
            model = PeftModel.from_pretrained(
                self.base_model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        model.half()
        return model

    # xxw 初始化critic
    def _init_critic(self, critic_weights = None):
        if self.algo == "APPO":
            # 利用APPOCritic充当critic
            # critic从actor(lora model)初始化而来
            # APPOCritic里面就是把model做了线性映射，n_embd -> 1024 -> 512 -> 1
            # 本质就是对一个输入x打一个分
            critic = APPOCritic(self.actor, self.tokenizer)
        elif self.algo == "TPPO":
            critic = TPPOCritic(self.actor, self.tokenizer)
        else:
            raise NotImplementedError
        if critic_weights is not None:
            critic.v_head.load_state_dict(torch.load(critic_weights, map_location= "cpu"))
        return critic
    
    def get_actions(self, obs):
        """
        Compute actions and value function predictions for the given inputs.
        """
        prompts = obs.tolist()
        token_seq = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        
        output = self.actor.generate(
            input_ids,
            attention_mask=attn_mask,
            do_sample=True,
            top_k=50,
            temperature=0.5,
            max_new_tokens=self.max_new_tokens,
            # bos_token_id=self.tokenizer.pad_token_id,
            # 1802: "и", 16748: "ки", 198: "\n", 624: ".\n", 715: " \n", 271: "\n\n", 76325: " \n\n\n\n\n"
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 198, 624, 715, 271, 76325], 
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        sequences = output.sequences
        
        actions = []
        action_tokens = torch.ones((sequences.shape[0], self.max_new_tokens), 
                                   dtype=torch.int64).to("cuda") * self.tokenizer.pad_token_id
        for i in range(sequences.shape[0]):
            action_token = sequences[i][input_ids[i].shape[0]:]
            action_tokens[i, :action_token.shape[0]] = action_token
            action = self.tokenizer.decode(action_token, skip_special_tokens=True)
            actions.append(action)
        actions = np.array(actions, dtype=np.object_)
        
        return actions, action_tokens
       
    # 获取当前action的value值
    def get_action_values(self, obs):
        obs = obs.tolist()
        # 先转换为token
        inputs = self.tokenizer(obs, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with self.actor.disable_adapter():
            # 给token做编码，做线性映射，得到一个value值
            values = self.critic(input_ids, attention_mask=attention_mask)
        return values
    
    def get_slice(self, logits, obs_full_lengths, act_real_lengths):
        action_slice = torch.zeros((logits.shape[0], self.max_new_tokens, logits.shape[-1])).to("cuda")
        for i in range(logits.shape[0]):
            start_idx = obs_full_lengths - 1
            end_idx = obs_full_lengths + act_real_lengths[i] - 1
            action_slice[i, :act_real_lengths[i]] = logits[i, start_idx:end_idx]
        return action_slice
    
    def get_token_values(self, obs, action_tokens):
        
        obs_token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        obs_input_ids = obs_token_seq["input_ids"].to("cuda")
        obs_attn_mask = obs_token_seq["attention_mask"].to("cuda")
        obs_full_lengths = obs_input_ids.shape[1]
        
        act_attn_mask = (action_tokens != 0)
        act_real_lengths = act_attn_mask.sum(dim=1)
        
        obs_act_ids = torch.cat([obs_input_ids, action_tokens], dim=1)
        obs_act_mask = torch.cat([obs_attn_mask, act_attn_mask], dim=1)
        
        with self.actor.disable_adapter():
            values = self.critic(obs_act_ids, attention_mask=obs_act_mask)
        values = self.get_slice(values, obs_full_lengths, act_real_lengths)
        return values

    # xxw core
    # get_token_logits 的目的是生成智能体在给定观测（obs）和动作（action_tokens）条件下的策略输出
    def get_token_logits(self, obs, action_tokens, batch_infer=False):
        
        # 首先使用 self.tokenizer 对 obs 进行编码，得到一个 tokens 序列，以便于进入模型进行处理
        obs_token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)

        # 得到 obs_token_seq 的 input_ids 和 attention_mask，并将它们转移到CUDA（GPU）上进行加速计算。
        obs_input_ids = obs_token_seq["input_ids"].to("cuda")
        obs_attn_mask = obs_token_seq["attention_mask"].to("cuda")
        obs_full_lengths = obs_input_ids.shape[1]

        # act_attn_mask 用于标识非零（即有效）动作token。这帮助计算每个动作序列的真实长度 act_real_lengths
        act_attn_mask = (action_tokens != 0)
        act_real_lengths = act_attn_mask.sum(dim=1)

        # 将 obs_input_ids 和 action_tokens 连接起来形成新的输入 obs_act_ids，是为了将观测状态和动作信息结合在一起，以便模型能够在理解当前环境状态的同时，考虑到其可能采取的动作
        # 在许多任务中，智能体需要根据当前的环境状态（obs）来选择合适的动作（action）-> 将两者连接起来，可以让模型同时具备关于环境和自身已采取或计划采取的动作的完整上下文。
        obs_act_ids = torch.cat([obs_input_ids, action_tokens], dim=1)
        obs_act_mask = torch.cat([obs_attn_mask, act_attn_mask], dim=1)
       
        # batch_infer 步骤的主要任务是通过模型进行推理，以获取关于这些输入的输出logits
        # 这种推理的目的是根据合并后的输入序列，让模型生成有关策略决策的信息，也就是 pi_logits 和 rho_logits
        if batch_infer:
            with self.actor.disable_adapter():
                rho_logits = self.batch_infer(self.actor, obs_act_ids, obs_act_mask, obs_full_lengths, act_real_lengths)
                        
            pi_logits = self.batch_infer(self.actor, obs_act_ids, obs_act_mask, obs_full_lengths, act_real_lengths)
        else:
            with self.actor.disable_adapter():
                rho_outputs = self.actor(input_ids=obs_act_ids, attention_mask=obs_act_mask, return_dict=True)
                rho_logits = self.get_slice(rho_outputs.logits, obs_full_lengths, act_real_lengths)
                
            pi_outputs = self.actor(input_ids=obs_act_ids, attention_mask=obs_act_mask, return_dict=True)
            pi_logits = self.get_slice(pi_outputs.logits, obs_full_lengths, act_real_lengths)
        
        # pi_logits: 被用于估计或采样行为策略，它提供了关于在特定环境状态下选择哪些可能动作的概率分布
        # rho_logits：可以用于策略基准修正或作为选择策略梯度更新的参考。这可能涉及到计算优势或者对比不同策略的效果，以便更好地更新和调整模型的策略。 
        return pi_logits, rho_logits
    
    def batch_infer(self, model, input_ids, attn_mask, obs_full_lengths, act_real_lengths, infer_batch_size=16):     
        logits = []
        for i in range(0, input_ids.shape[0], infer_batch_size):
            input_ids_batch = input_ids[i:i+infer_batch_size, :]
            attn_mask_batch = attn_mask[i:i+infer_batch_size, :]
            outputs = model(input_ids=input_ids_batch, attention_mask=attn_mask_batch, return_dict=True)
            
            logits_batch = self.get_slice(outputs.logits, obs_full_lengths, act_real_lengths)
            logits.append(logits_batch.clone())
        logits = torch.cat(logits, dim=0)
        
        return logits
        
    def get_last_token_position(self, action_tokens):
        pos = len(action_tokens) - 1
        while action_tokens[pos] == self.tokenizer.pad_token_id:
            pos -= 1
        return pos

    # xxw core 获取action model的log prob
    def get_joint_action_log_probs(self, obs, action_tokens, batch_infer=False):

        # obs: 当前观测的输入
        # action_tokens: 与动作对应的token序列
        # pi_logits: 被用于估计或采样行为策略，它提供了关于在特定环境状态下选择哪些可能动作的概率分布
        pi_logits, _ = self.get_token_logits(obs, action_tokens, batch_infer=batch_infer)

        # 通过torch.log_softmax计算这些logits的log-softmax
        # log-softmax给出了智能体在特定观察条件obs下采取每一个可能动作action的倾向性或优先级。
        pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)

        # 初始化action_log_probs和entropies列表
        # 用于存储每个样本的log概率和熵
        action_log_probs = []
        entropies = []

        # 逐序列计算:
        for i in range(pi_logits.shape[0]):
            # 确定动作序列的有效长度act_token_length
            act_token_length = self.get_last_token_position(action_tokens[i]) + 1
            # 提取当前序列的log_softmax_slice和action_token_slice
            # 目的是为了处理每个动作序列的特定片段，从而计算该序列的联合log概率
            # action_token_slice是与log_softmax_slice对应的动作token序列，这些token指示在特定时间步上智能体实际采取的动作。
            log_softmax_slice = pi_log_softmax[i, :act_token_length, :]
            action_token_slice = action_tokens[i, :act_token_length]
            # 使用这两个切片之后，可以通过torch.gather从log_softmax_slice中选择action_token_slice所指示的位置的log概率。
            token_log_probs = torch.gather(log_softmax_slice, -1, action_token_slice.unsqueeze(-1)).squeeze(-1)
            # 对这些log概率求和，得到该序列的联合log概率 action_log_prob，并存储。
            action_log_prob = token_log_probs.sum()
            action_log_probs.append(action_log_prob)
            # 使用categorical分布基于「当前动作token的logits」计算熵。
            entropy = Categorical(logits=pi_logits[i, :act_token_length, :]).entropy().mean()
            entropies.append(entropy)
        action_log_probs = torch.stack(action_log_probs)
        entropies = torch.stack(entropies)
        # log probability: 用于策略梯度更新中作为目标函数计算的一部分。
        # entropy: 可用于促进策略探索，增加策略的随机性以帮助空间探索更优策略，并在训练过程中用作正则项。
        return action_log_probs, entropies
    
    @torch.no_grad()
    def infer_for_rollout(self, obs):
        actions, action_tokens = self.get_actions(obs)
        
        if self.algo == "APPO":
            values = self.get_action_values(obs)
            values = values.float().cpu().numpy()
            action_log_probs, _ = self.get_joint_action_log_probs(obs, action_tokens, batch_infer=True)
            action_tokens = action_tokens.int().cpu().numpy()
            action_log_probs = action_log_probs.float().cpu().numpy()
            log_probs = action_log_probs
        elif self.algo == "TPPO":
            values = self.get_token_values(obs, action_tokens).squeeze(-1)
            pi_logits, _ = self.get_token_logits(obs, action_tokens, batch_infer=True)
            pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
            token_log_probs = torch.gather(pi_log_softmax, -1, action_tokens.unsqueeze(-1)).squeeze(-1)

            values = values.float().cpu().numpy()
            action_tokens = action_tokens.int().cpu().numpy()
            token_log_probs = token_log_probs.float().cpu().numpy()
            log_probs = token_log_probs
        elif self.algo == "GRPO":
            values = np.zeros((obs.shape[0],)) # fake values, grpo does not use critic
            action_log_probs, _ = self.get_joint_action_log_probs(obs, action_tokens, batch_infer=True)
            action_tokens = action_tokens.int().cpu().numpy()
            action_log_probs = action_log_probs.float().cpu().numpy()
            log_probs = action_log_probs
        else:
            raise NotImplementedError

        return actions, action_tokens, values, log_probs
    
    def get_next_tppo_values(self, obs): 
        token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        
        # values
        with self.actor.disable_adapter():
            values = self.critic(input_ids, attention_mask=attn_mask)
            values = values[:, -1]
        return values
    
    @torch.no_grad()
    def get_next_values(self, obs):
        """
        Get value function predictions.
        """
        if self.algo == "APPO":
            values = self.get_action_values(obs)
            values = values.cpu().float().numpy()
            return values
        elif self.algo == "TPPO":
            values = self.get_next_tppo_values(obs).squeeze(-1)
            values = values.cpu().float().numpy()
            return values
        elif self.algo == "GRPO":
            return np.zeros((obs.shape[0],)) # fake values, grpo does not use critic
        else: 
            raise NotImplementedError
    # xxw core
    # obs：当前智能体的观测
    # action_tokens：智能体针对观测做出的动作
    def infer_for_action_update(self, obs, action_tokens= None):
        assert action_tokens is not None, "action_tokens could not be none"
        # action_log_probs：用于策略梯度更新中作为目标函数计算的一部分
        # entropies：可用于促进策略探索，增加策略的随机性以帮助空间探索更优策略，并在训练过程中用作正则项
        action_log_probs, entropies = self.get_joint_action_log_probs(obs, action_tokens)
        return action_log_probs, entropies
    
    def infer_for_token_update(self, obs, action_tokens):
        pi_logits, rho_logits = self.get_token_logits(obs, action_tokens)
        return pi_logits, rho_logits
    
    def test_get_actions(self, obs):
        """
        Compute actions and value function predictions for the given inputs.
        """
        prompts = obs.tolist()
        token_seq = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        
        output = self.actor.generate(
            input_ids,
            attention_mask=attn_mask,
            do_sample=False,
            # top_k=50,
            # temperature=0.5,
            max_new_tokens=self.max_new_tokens,
            # bos_token_id=self.tokenizer.pad_token_id,
            # 1802: "и", 16748: "ки", 198: "\n", 624: ".\n", 715: " \n", 271: "\n\n", 76325: " \n\n\n\n\n"
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 198, 624, 715, 271, 76325], 
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        sequences = output.sequences
        
        actions = []
        for i in range(sequences.shape[0]):
            action_token = sequences[i][input_ids[i].shape[0]:]
            action = self.tokenizer.decode(action_token, skip_special_tokens=True)
            actions.append(action)
        actions = np.array(actions, dtype=np.object_)
        
        return actions

    def save(self, save_dir, episode):
        print("save model")
        exp_path = os.path.join(save_dir, "episode_{:04d}".format(episode))

        os.makedirs(exp_path, exist_ok=True)
        self.actor.save_pretrained(exp_path)

    def load(self, save_dir):
        print("load model")
        self.actor = self._init_actor(save_dir).to(self.device)

    def train(self):
        self.generator.train()
        self.critic.train()

    def eval(self):
        self.generator.eval()
        self.critic.eval()

