import copy
import itertools
import torch
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from typing import Mapping, Optional, Any, Dict, Tuple
from torch.nn import Module

from ..base.agent.agent import Agent
from ..utils import obs_to_graph

class SACAgent(Agent):
    """
    Multi-Agent Soft Actor-Critic (SAC) Agent with Centralized Critics.
    This agent implements the BaseMultiAgent interface and is designed to be used
    within the MainDriver-RolloutWorker architecture.
    """
    def __init__(self, 
                 num_agents: int,
                 models: Mapping[str, Module], 
                 device: torch.device, 
                 cfg: Optional[dict] = None):
        
        super().__init__(models, device, cfg)

        self.num_agent = num_agents

        # --- Model Registration ---
        self.policy_feature_extractor = self.models.get("policy_feature_extractor")
        self.value_feature_extractor = self.models.get("value_feature_extractor")
        self.policy = self.models.get("policy")
        self.safety = self.models.get("safety")
        self.value_1 = self.models.get("value_1")
        self.value_2 = self.models.get("value_2")
        
        # Create target networks and sync their weights
        self.target_critic_1 = copy.deepcopy(self.value_1)
        self.target_critic_2 = copy.deepcopy(self.value_2)

        # Freeze target networks - they are only updated via polyak averaging
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        for p in self.target_critic_1.parameters():
            p.requires_grad = False
        for p in self.target_critic_2.parameters():
            p.requires_grad = False

        # --- Hyperparameters ---
        self.discount_factor = self.cfg.get("discount_factor", 0.99)
        self.polyak = self.cfg.get("polyak", 0.005)
        self.grad_norm_clip = self.cfg.get("grad_norm_clip", 0.5)
        self.lr = self.cfg.get("learning_rate", 1e-3)
        self.entropy_learning_rate = self.cfg.get("entropy_learning_rate", 1e-3)
        self.minimum_buffer_size = self.cfg.get("minimum_buffer_size", 1000)

        self.num_obs = self.policy_feature_extractor.in_features
        self.num_state = self.value_feature_extractor.in_features
        self.num_act = self.policy.out_features

        if type(self.lr) == str:
            self.lr = float(self.lr)

        if type(self.entropy_learning_rate) == str:
            self.entropy_learning_rate = float(self.entropy_learning_rate)


        # --- Optimizers ---
        self.policy_optimizer = torch.optim.Adam(itertools.chain(self.policy_feature_extractor.parameters(),
                                                                 self.policy.parameters()), lr=self.lr)
        self.value_optimizer = torch.optim.Adam(itertools.chain(self.value_feature_extractor.parameters(),
                                                                 self.value_1.parameters(),
                                                                 self.value_2.parameters()), lr=self.lr)

        # --- Entropy Tuning ---
        self.learn_entropy = self.cfg.get("learn_entropy", True)
        if self.learn_entropy:
            self.target_entropy = self.cfg.get("target_entropy", -self.models['policy'].out_features) # Heuristic
            self.log_alpha = torch.tensor(self.cfg.get("initial_entropy_value", 0.2), device=self.device).log().requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.entropy_learning_rate)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = torch.tensor(self.cfg.get("initial_entropy_value", 0.2), device=self.device)

        # --- Register modules for checkpointing by the MainDriver ---
        self.checkpoint_modules["policy_feature_extractor"] = self.policy_feature_extractor
        self.checkpoint_modules["value_feature_extractor"] = self.value_feature_extractor
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value_1"] = self.value_1
        self.checkpoint_modules["value_2"] = self.value_2
        self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
        self.checkpoint_modules["value_optimizer"] = self.value_optimizer
        if self.learn_entropy:
            self.checkpoint_modules["log_alpha"] = self.log_alpha
            self.checkpoint_modules["alpha_optimizer"] = self.alpha_optimizer

    def act(self, 
            obs: torch.Tensor, 
            deterministic: bool = False, 
            safety_info: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Called by the RolloutWorker. Samples actions from the policy.
        If deterministic is True, it returns the mean of the policy distribution.

        :param states: A tensor of shape (num_agents, obs_dim).
        :param deterministic: Whether to sample from the distribution or take the mean.
        :return: A tensor of shape (num_agents, action_dim).
        """
        self.policy_feature_extractor.eval()
        self.safety.eval()
        self.policy.eval()
        with torch.no_grad():
            data = obs_to_graph(obs, self.device)
            embedding = self.policy_feature_extractor(data)
            if deterministic:
                # For evaluation, take the mean of the distribution
                mu, logp = self.policy(embedding)
                # SAC actions are squashed by tanh, so the deterministic action should also be squashed.
                actions = torch.tanh(mu)
            else:
                # For training, sample from the distribution
                # The `sample` method from ActorGaussianNet already returns tanh-squashed actions
                actions, logp = self.policy.compute(embedding)
        
            actions_s = self.safety(actions, safety_info)

        self.policy_feature_extractor.train()
        self.policy.train()
        self.safety.train()

        return actions, actions_s, logp

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """
        Returns the state of the agent for checkpointing.
        """
        checkpoint_data = {}
        for name, module in self.checkpoint_modules.items():
            if isinstance(module, torch.nn.Module) or isinstance(module, torch.optim.Optimizer):
                checkpoint_data[name] = module.state_dict()
            else:
                checkpoint_data[name] = module
        return checkpoint_data

    def update(self, batch: Dict[str, torch.Tensor | Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Called by the MainDriver. Performs one step of SAC update.
        Assumes the batch is a dictionary of tensors sampled from a replay buffer,
        where each entry corresponds to a single agent's transition.
        This version explicitly uses a shared feature extractor.
        """
        # Unpack batch from dictionary and move to device
        if type(batch['obs']) == dict:
            obs = {k: v.to(self.device) for k, v in batch['obs'].items()}
            state = {k: v.to(self.device) for k, v in batch['state'].items()}
            next_obs = {k: v.to(self.device) for k, v in batch['next_obs'].items()}
            next_state = {k: v.to(self.device) for k, v in batch['next_state'].items()}
            info = {k: v.to(self.device) for k, v in batch['info'].items()}
        else:
            obs = batch['obs'].to(self.device)
            state = batch['state'].to(self.device)
            next_obs = batch['next_obs'].to(self.device)
            next_state = batch['next_state'].to(self.device)
            info = {k: v.to(self.device) for k, v in batch['info'].items()}

        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = (batch['terminated'] | batch['truncated']).to(self.device)

        # --- Feature Extraction ---
        # Extract features for policy and critic respectively.
        safety_info = info["safety"]
        next_safety_info = info["next_safety"]
        obs_features = self.policy_feature_extractor(obs_to_graph(obs, self.device))
        state_features = self.value_feature_extractor(obs_to_graph(state, self.device))
        with torch.no_grad():
            next_obs_features = self.policy_feature_extractor(obs_to_graph(next_obs, self.device))
            next_actions, next_log_pi = self.policy.compute(next_obs_features)
            next_safe_actions = self.safety(next_actions, next_safety_info)

            next_state["graph_features"][:, (self.num_state-self.num_act):] = next_safe_actions
            next_state_features = self.value_feature_extractor(obs_to_graph(next_state, self.device))
            critic_input_next = next_state_features

        # --- Critic Loss ---
        with torch.no_grad():
            q1_next_target = self.target_critic_1.compute(critic_input_next)
            q2_next_target = self.target_critic_2.compute(critic_input_next)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_pi
            next_q_value = rewards + (~dones) * self.discount_factor * min_q_next_target

        critic_input_current = state_features
        q1_current = self.value_1.compute(critic_input_current)
        q2_current = self.value_2.compute(critic_input_current)
        critic_loss = (F.mse_loss(q1_current, next_q_value) + F.mse_loss(q2_current, next_q_value)) / 2

        value_params = itertools.chain(self.value_feature_extractor.parameters(),
                                 self.value_1.parameters(),
                                 self.value_2.parameters())
        
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_params, self.grad_norm_clip)
        self.value_optimizer.step()

        # --- Policy Loss ---
        # TODO : state_features를 critic loss 계산 이후 재계산 해야 하는지 Check
        pi, log_pi = self.policy.compute(obs_features)
        pi_safe = self.safety(pi, safety_info)
        
        state_features["graph_features"][:, (self.num_state-self.num_act):] = pi_safe
        q1_pi = self.value_1(state_features)
        q2_pi = self.value_2(state_features)
        policy_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

        # --- Alpha (Entropy) Loss ---
        if self.learn_entropy:
            alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        policy_params = itertools.chain(self.policy_feature_extractor.parameters(),
                                       self.policy.parameters())

        # --- Combined Optimization Step for Critic and Policy ---
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_params, self.grad_norm_clip)
        self.policy_optimizer.step()

        # --- Alpha Optimizer Step ---
        if self.learn_entropy:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

        # --- Target Network Update ---
        with torch.no_grad():
            for target_param, param in zip(self.target_critic_1.parameters(), self.value_1.parameters()):
                target_param.data.mul_(1.0 - self.polyak)
                target_param.data.add_(param.data * self.polyak)
            for target_param, param in zip(self.target_critic_2.parameters(), self.value_2.parameters()):
                target_param.data.mul_(1.0 - self.polyak)
                target_param.data.add_(param.data * self.polyak)
