from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import List
from utils import FREE, UNKNOWN, OCCUPIED, GOAL, START, FRONTIER

# ======================================================================================
# Colormaps and Normalization
# ======================================================================================
BELIEF_CMAP = colors.ListedColormap([
    '#FFFFFF',  # FREE
    '#BDBDBD',  # UNKNOWN
    '#000000',  # OCCUPIED
    '#1E90FF',  # GOAL
    '#34C759',  # START
    '#FF0000',  # FRONTIER
])
BELIEF_NORM = colors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5,5.5], BELIEF_CMAP.N)

GT_CMAP = colors.ListedColormap([
    '#FFFFFF',  # FREE
    '#FFFFFF',  # UNKNOWN (unused)
    '#000000',  # OCCUPIED
    '#FF3B30',  # GOAL
    '#34C759',  # START
    '#FFFFFF',  # FRONTIER (unused)
])
GT_NORM = BELIEF_NORM

AGENT_COLORS = ['#E6194B', '#4363D8', '#3CB44B', '#F58231', "#9D24C2", '#469990'] # Red, Blue, Green, Orange, Purple, Teal

# ======================================================================================
# Drawing Functions
# ======================================================================================
def draw_frame(ax_gt, ax_belief, maps, sim):
    ax_gt.clear(); ax_belief.clear()
    ax_gt.imshow(maps.gt, cmap=GT_CMAP, norm=GT_NORM, origin='upper')
    ax_belief.imshow(maps.belief, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin='upper')

    spec = maps.spec
    viz_data = sim.get_viz()

    def world_to_img(x, y):
        row, col = spec.world_to_grid(x, y)
        return col, row

    for i in range(sim.num_agents):
        robot = sim.robots[i]
        controller = sim.controllers[i]
        color = AGENT_COLORS[i % len(AGENT_COLORS)]

        # --- Safety and Connectivity Circles ---
        cx, cy = world_to_img(robot.x, robot.y)
        radius_min_px = controller.d_safe*0.5 / spec.res_m
        radius_max_px = controller.d_max*0.5 / spec.res_m

        for ax in (ax_gt, ax_belief):
            # Min safety distance circle (dashed)
            min_circle = plt.Circle((cx, cy), radius_min_px, color=color, fill=False, linestyle='--', linewidth=1, alpha=0.8)
            ax.add_patch(min_circle)
            
            # Max connectivity distance circle (dotted)
            max_circle = plt.Circle((cx, cy), radius_max_px, color=color, fill=False, linestyle=':', linewidth=1.2, alpha=0.7)
            ax.add_patch(max_circle)

        # --- Robot Center ---
        for ax in (ax_gt, ax_belief):
            center_dot = plt.Circle((cx, cy), 2, color=color, zorder=5)
            ax.add_patch(center_dot)

        # --- FOV sector (semi-transparent) ---
        half = math.radians(sim.sensor.fov_deg / 2.0)
        angles = np.linspace(-half, half, 20)
        poly_world = [(robot.x, robot.y)] + [
            (robot.x + sim.sensor.max_range_m * math.cos(robot.yaw + a),
             robot.y + sim.sensor.max_range_m * math.sin(robot.yaw + a)) for a in angles
        ]
        poly_img = [world_to_img(x, y) for (x, y) in poly_world]
        for ax in (ax_gt, ax_belief):
            ax.fill([p[0] for p in poly_img], [p[1] for p in poly_img],
                    alpha=0.15, color=color, zorder=2)

        # --- Path history ---
        path = viz_data["paths"][i]
        if len(path) > 1:
            xs, ys = zip(*[world_to_img(wx, wy) for wx, wy in path])
            for ax in (ax_gt, ax_belief):
                ax.plot(xs, ys, '-', linewidth=2, color=color, alpha=0.8, zorder=3)

        # --- Robot heading/velocity arrow ---
        v_cmd, _ = viz_data["last_cmds"][i]
        length = max(0.05, v_cmd * 0.5)
        x2 = robot.x + length * math.cos(robot.yaw)
        y2 = robot.y + length * math.sin(robot.yaw)
        cx, cy = world_to_img(robot.x, robot.y)
        cx2, cy2 = world_to_img(x2, y2)
        for ax in (ax_gt, ax_belief):
            ax.arrow(cx, cy, cx2 - cx, cy2 - cy, head_width=5, head_length=8,
                     fc=color, ec=color, length_includes_head=True, zorder=5)

        # --- Target point ---
        target = viz_data["targets_world"][i]
        if target is not None:
            ux, uy = world_to_img(target[0], target[1])
            for ax in (ax_gt, ax_belief):
                ax.scatter([ux], [uy], s=25, c=color, edgecolors='white',
                           linewidths=0.7, zorder=6)

        # --- Cluster points (for leader) ---
        if i == sim.leader_idx:
            cluster = viz_data["clusters_world"][i]
            if cluster is not None and len(cluster) > 0:
                U, V = zip(*[world_to_img(wx, wy) for wx, wy in cluster])
                for ax in (ax_gt, ax_belief):
                    ax.scatter(U, V, s=2, c=color, alpha=0.9, zorder=6)

    # --- Final Touches ---
    for ax in (ax_gt, ax_belief):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('Ground Truth' if ax is ax_gt else 'Belief')

def save_gif(maps, sim, steps: int = 100, out_path: str = 'run.gif', dpi: int = 120):
    import imageio
    frames: List[np.ndarray] = []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.0, 10.0))

    for step_num in range(steps):
        print(f"Step : {step_num + 1} / {steps}")
        sim.step()
        draw_frame(ax1, ax2, maps, sim)

        viz = sim.get_viz()
        if any(viz['crashed']):
            crashed_idx = viz['crashed'].index(True)
            msg = f"Agent {crashed_idx} CRASHED"
            ax1.text(0.5, 0.5, msg, transform=ax1.transAxes, ha='center', va='center', fontsize=12, color='red', weight='bold', zorder=10, bbox=dict(facecolor='white', alpha=0.8))
        elif all(viz['stopped']):
            ax1.text(0.5, 0.5, "All agents stopped", transform=ax1.transAxes, ha='center', va='center', fontsize=12, color='black', weight='bold', zorder=10, bbox=dict(facecolor='white', alpha=0.8))

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf, dtype=np.uint8)[..., :3]
        frames.append(frame.copy())

        if any(viz['crashed']) or all(viz['stopped']):
            print("Simulation ended.")
            break

    plt.close(fig)
    print(f"Saving GIF to {out_path}...")
    imageio.mimsave(out_path, frames, fps=int(1.0/sim.dt))
    print("GIF saved.")
    return out_path

def plot_agent_distances(paths: List[List[tuple[float, float]]], d_safe: float, d_max: float, dt: float):
    """
    Plots the distances between each agent and all other agents over time.

    Args:
        paths: A list of paths, where each path is a list of (x, y) tuples.
        d_safe: The minimum safety distance.
        d_max: The maximum connectivity distance.
        dt: The simulation time step.
    """
    num_agents = len(paths)
    if num_agents < 2:
        print("Not enough agents to plot distances.")
        return

    # Find the length of the shortest path to determine the number of time steps
    min_len = min(len(p) for p in paths)
    timesteps = np.arange(min_len) * dt

    # Create a figure with subplots for each agent
    fig, axes = plt.subplots(num_agents, 1, figsize=(10, 2 * num_agents), sharex=True)
    if num_agents == 1:
        axes = [axes] # Make it iterable

    fig.suptitle('Inter-Agent Distances Over Time', fontsize=16)

    for i in range(num_agents):
        ax = axes[i]
        path_i = np.array(paths[i][:min_len])

        for j in range(num_agents):
            if i == j:
                continue

            path_j = np.array(paths[j][:min_len])
            
            # Calculate the Euclidean distance at each time step
            distances = np.linalg.norm(path_i - path_j, axis=1)
            
            # Plot the distance to agent j
            ax.plot(timesteps, distances, label=f'Distance to Agent {j}', color=AGENT_COLORS[j % len(AGENT_COLORS)])

        # Plot d_safe and d_max lines
        ax.axhline(y=d_safe, color='r', linestyle='--', label=f'd_safe ({d_safe}m)')
        ax.axhline(y=d_max, color='b', linestyle=':', label=f'd_max ({d_max}m)')
        
        ax.set_title(f'Agent {i}')
        ax.set_ylabel('Distance (m)')
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
