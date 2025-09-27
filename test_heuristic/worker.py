from simulation import build_minimal_env
from visualization import save_gif, plot_agent_distances, plot_control_inputs, plot_cbf_values, plot_psi_values

maps, sim = build_minimal_env()
maps.add_random_rect_obstacles(n=0, min_w_m=0.05, min_h_m=0.05,
                               max_w_m=0.2, max_h_m=0.2, seed=20)

gif_path = save_gif(maps, sim, steps=200, out_path='belief_gt.gif')
print("Saved:", gif_path)

# Plot and save the inter-agent distances
viz_data = sim.get_viz()
if viz_data["paths"]:
    plot_agent_distances(viz_data["paths"], viz_data["d_safe"], viz_data["d_max"], sim.dt, save_path="agent_distances.png")

# Plot and save control inputs
if viz_data["nominal_inputs_history"][0]:
    plot_control_inputs(viz_data["nominal_inputs_history"], viz_data["safe_inputs_history"], sim.dt, sim.num_agents, save_path="control_inputs.png")
    plot_cbf_values(viz_data["cbf_values_history"], sim.dt, sim.num_agents, save_path="cbf_values.png")
    plot_psi_values(viz_data["cbf_values_history"], sim.dt, sim.num_agents, save_path="psi_values.png")
