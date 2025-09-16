from simulation import build_minimal_env
from visualization import save_gif, plot_agent_distances

maps, sim = build_minimal_env()
maps.add_random_rect_obstacles(n=0, min_w_m=0.05, min_h_m=0.05,
                               max_w_m=0.2, max_h_m=0.2, seed=20)

gif_path = save_gif(maps, sim, steps=200, out_path='belief_gt.gif')
print("Saved:", gif_path)

# Plot the inter-agent distances
if sim.paths:
    d_safe = sim.controllers[0].d_safe
    d_max = sim.controllers[0].d_max
    plot_agent_distances(sim.paths, d_safe, d_max, sim.dt)
