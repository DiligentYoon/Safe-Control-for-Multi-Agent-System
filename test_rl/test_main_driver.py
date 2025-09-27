import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from typing import List

# Assuming the necessary classes are importable from your project structure
from task.env.cbf_env import CBFEnv
from task.agent.sac import SACAgent
from task.models.models import ActorGaussianNet, CriticDeterministicNet, GNN_Feature_Extractor, DifferentiableCBFLayer
from visualization import draw_frame, AGENT_COLORS, plot_agent_distances, plot_control_inputs, plot_cbf_values, plot_psi_values

def create_models(cfg: dict, obs_dim: int, state_dim: int, action_dim: int, device: torch.device) -> dict:
    """Helper function to create models based on the config."""
    model_cfg = cfg['model']
    
    global_feature_extractor = GNN_Feature_Extractor(state_dim, model_cfg['feature_extractor'], "global")
    local_feature_extractor = GNN_Feature_Extractor(obs_dim, model_cfg['feature_extractor'], "local")
    
    policy_input_dim = model_cfg['feature_extractor']['hidden']
    policy = ActorGaussianNet(policy_input_dim, action_dim, device, model_cfg['actor'])
    
    safety = DifferentiableCBFLayer(cfg=model_cfg["safety"])
    
    critic_input_dim = model_cfg['feature_extractor']['hidden']
    critic1 = CriticDeterministicNet(critic_input_dim, 1, device, model_cfg['critic'])
    critic2 = CriticDeterministicNet(critic_input_dim, 1, device, model_cfg['critic'])
    
    return {
        "policy_feature_extractor": local_feature_extractor,
        "value_feature_extractor": global_feature_extractor,
        "policy": policy, 
        "safety": safety,
        "value_1": critic1, 
        "value_2": critic2
    }

def run_simulation_test(cfg: dict, steps: int = 200, out_dir: str = 'test_results'):
    """
    Runs a simulation test for the CBFEnv and SACAgent, generating a GIF and plots.
    """
    print("=== Starting Simulation Test ===")
    
    # --- Output Directory ---
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results will be saved to: {out_dir}")

    # --- Device ---
    device = torch.device(cfg['env']['device'])

    # --- Environment ---
    env = CBFEnv(episode_index=0, cfg=cfg['env'])
    obs, state, info = env.reset()
    print("Environment created and reset.")

    # --- Agent & Models ---
    obs_dim = env.cfg.num_obs
    state_dim = env.cfg.num_state
    action_dim = env.cfg.num_act
    num_agents = cfg['env']['num_agent']

    # Update safety params in config from env, similar to MainDriver
    cfg['model']['safety']['a_max'] = env.cfg.max_yaw_rate
    cfg['model']['safety']['w_max'] = env.cfg.max_acceleration
    cfg['model']['safety']['d_max'] = env.cfg.d_max
    cfg['model']['safety']['d_obs'] = env.cfg.d_obs
    cfg['model']['safety']['d_safe'] = env.cfg.d_safe
    cfg['model']['safety']['max_agents'] = env.cfg.max_agents
    cfg['model']['safety']['max_obs'] = env.cfg.max_obs

    models = create_models(cfg, obs_dim, state_dim, action_dim, device)
    agent = SACAgent(num_agents=num_agents, models=models, device=device, cfg=cfg['agent'])
    print("Agent with models created.")

    # --- Simulation Loop ---
    frames: List[np.ndarray] = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 6.0))

    # Data tracking for plots
    path_history = [[] for _ in range(num_agents)]
    cbf_history = [[] for _ in range(num_agents)]

    for step_num in range(steps):
        # --- Agent takes action ---
        _, actions, _ = agent.act(obs, safety_info=info["safety"])

        # --- Environment steps ---
        next_obs, next_state, reward, terminated, truncated, next_info = env.step(actions)
        done = np.any(terminated) or np.any(truncated)
        
        # --- Record data ---
        for i in range(num_agents):
            path_history[i].append((env.world.robots[i].x, env.world.robots[i].y))
            # nominal_control_history[i].append(nominal_actions[i])
            # safe_control_history[i].append(actions[i])
            
            # Store CBF related values from the info dict returned by the safety layer
            agent_cbf_info = {}
            for key, val in info.items():
                if val is not None:
                    agent_cbf_info[key] = val[i].cpu().numpy()
            cbf_history[i].append(agent_cbf_info)


        print(f"Step: {step_num + 1}/{steps} | Reward: {np.mean(reward):.3f} | Done: {done}")

        # --- Visualization ---
        # Create a dictionary with visualization data
        viz_data = {
            "paths": path_history,
            "last_cmds": [(actions[i][0], actions[i][1]) for i in range(num_agents)],
            # Add any other data needed for visualization
        }

        # Call the new draw_frame function
        draw_frame(ax1, ax2, env, viz_data)
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf, dtype=np.uint8)[..., :3]
        frames.append(frame.copy())

        obs = next_obs
        state = next_state
        info = next_info

        if done:
            print("Simulation ended because 'done' is True.")
            break
            
    plt.close(fig)
    
    # --- Save GIF ---
    gif_path = os.path.join(out_dir, 'simulation_test.gif')
    print(f"Saving GIF to {gif_path}...")
    imageio.mimsave(gif_path, frames, fps=int(1.0/env.dt))
    print("GIF saved.")

    # # --- Generate and Save Plots ---
    # plot_agent_distances(path_history, env.cfg.d_safe, env.cfg.d_max, env.dt, save_path=os.path.join(out_dir, 'agent_distances.png'))
    # plot_control_inputs(nominal_control_history, safe_control_history, env.dt, num_agents, save_path=os.path.join(out_dir, 'control_inputs.png'))
    # plot_cbf_values(cbf_history, env.dt, num_agents, save_path=os.path.join(out_dir, 'cbf_values.png'))
    # plot_psi_values(cbf_history, env.dt, num_agents, save_path=os.path.join(out_dir, 'psi_values.png'))

    print("=== Simulation Test Finished ===")


if __name__ == '__main__':
    # Load config
    with open("config/cbf_test.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # It's good practice to run tests with deterministic behavior
    torch.manual_seed(config['env']['seed'])
    np.random.seed(config['env']['seed'])
    
    # Run the test
    run_simulation_test(config, steps=300)
