import ray
import numpy as np
import torch
from typing import Dict, Any

from task.env.cbf_env import CBFEnv
from task.models.models import GNN_Feature_Extractor, ActorGaussianNet, CriticDeterministicNet
from task.agent.sac import SACAgent

@ray.remote
class OffPolicyWorker:
    """
    A Ray remote actor responsible for collecting experience from the environment.
    It uses a lightweight HAC agent instance to generate actions.
    """
    def __init__(self, 
                 worker_id: int, 
                 env_cfg: Dict[str, Any], 
                 agent_cfg: Dict[str, Any], 
                 model_cfg: Dict[str, Any]):
        """
        Initializes the worker, its environment, and a local lightweight agent.
        It also initializes the persistent state of the environment for rollouts.
        """
        self.worker_id = worker_id
        self.device = torch.device("cpu")

        # --- Environment ---
        self.env = CBFEnv(episode_index=0, cfg=env_cfg)
        
        # --- Lightweight Agent for Acting ---
        self.agent_cfg = agent_cfg
        self.rollout_length = self.agent_cfg['rollout']
        
        obs_dim = self.env.cfg.num_obs
        state_dim = self.env.cfg.num_state
        action_dim = self.env.cfg.num_act
        
        models = self._create_models(obs_dim, state_dim, action_dim, model_cfg)
        self.agent = SACAgent(num_agents=self.env.num_agent,
                              models=models,
                              device=self.device,
                              cfg=agent_cfg)
        
        # --- Persistent Environment State ---
        self.last_obs, _, _ = self.env.reset()
        self.last_done = True

        print(f"Worker {self.worker_id}: Initialized. Will collect {self.rollout_length} steps per call.")


    def _create_models(self, obs_dim, state_dim, action_dim, num_agents, model_cfg) -> dict:
        """Creates the policy and critic models."""

        # Feature Extractor
        global_feature_extractor = GNN_Feature_Extractor(state_dim, model_cfg['feature_extractor'])
        local_feature_extractor = GNN_Feature_Extractor(obs_dim, model_cfg['feature_extractor'])

        actor_type = model_cfg['actor']['type']
        if actor_type == "Gaussian":
            policy_input_dim = model_cfg['feature_extractor']['hidden']
            policy = ActorGaussianNet(policy_input_dim, action_dim, self.device, model_cfg['actor'])
        else:
            ValueError("[INFO] TODO : we should construct the deterministic policy for MADDPG ...")
    
        # Centralized critic input dimension: state + agent actions
        critic_input_dim = model_cfg['feature_extractor']['hidden']
        critic1 = CriticDeterministicNet(critic_input_dim, 1, self.device, model_cfg['critic'])
        critic2 = CriticDeterministicNet(critic_input_dim, 1, self.device, model_cfg['critic'])
        
        return {"policy_feature_exractor": local_feature_extractor,
                "value_feature_extractor": global_feature_extractor,
                "policy": policy, 
                "value_1": critic1, 
                "value_2": critic2}

    def set_weights(self, policy_weights: Dict[str, Dict[str, torch.Tensor]]):
        """
        Updates the local agent's policy networks with new weights from the driver.
        """
        self.agent.policy_feature_extractor.load_state_dict(policy_weights["policy_feature_extractor"])
        self.agent.policy.load_state_dict(policy_weights["policy"])

    def rollout(self, episode_index: int) -> Dict[str, Any]:
        """
        Runs one full episode in the environment to collect a trajectory.
        """
        trajectory = []
        obs, state, info = self.env.reset(episode_index=episode_index)
        terminated, truncated = np.zeros((self.env.num_agent, 1), dtype=bool), np.zeros((self.env.num_agent, 1), dtype=bool)
        episode_reward = 0
        episode_length = 0

        done = False
        while not done:
            # Convert observation to a tensor for the policy network
            if type(obs) == dict:
                obs_tensor = {'graph_features': torch.tensor(obs['graph_features'], dtype=torch.float32, device=self.device),
                              'edge_index': torch.tensor(obs['edge_index'], dtype=torch.long, device=self.device)}
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

            # Get actions from the local policy
            with torch.no_grad():
                actions, _ = self.agent.act(obs_tensor)
            
            # Step the environment
            next_obs, next_state, rewards, terminated, truncated, infos = self.env.step(actions.cpu().numpy())
            
            # Store the complete transition information
            actions = actions.cpu().numpy()
            for i in range(self.env.num_agent):
                trajectory.append({
                    "obs": obs[i],
                    "state": state[i],
                    "actions": actions[i],
                    "rewards": rewards[i],
                    "next_obs": next_obs[i],
                    "next_state": next_state[i],
                    "terminated": terminated[i],
                    "truncated": truncated[i]
                })
            
            obs = next_obs
            state = next_state
            done = np.any(terminated) | np.any(truncated)
            episode_reward += rewards.sum()
            episode_length += 1

        metrics = {
            f"episode_reward": episode_reward,
            f"episode_length": episode_length,}
        
        return {"trajectory": trajectory, "metrics": metrics, "worker_id": self.worker_id}

