from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import List
from utils import FREE, UNKNOWN, OCCUPIED, GOAL, START, FRONTIER

# BELIEF: frontier=RED, occupied=BLACK; goal=BLUE
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

def draw_frame(ax_gt, ax_belief, maps, sim):
    gt = maps.gt
    bel = maps.belief

    ax_gt.clear(); ax_belief.clear()
    ax_gt.imshow(gt, cmap=GT_CMAP, norm=GT_NORM, origin='upper')
    ax_belief.imshow(bel, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin='upper')

    # --- 공통 유틸 ---
    r = sim.robots[0]
    sensor = sim.sensor
    spec = maps.spec
    rx, ry, th = r.x, r.y, r.yaw

    def world_to_img(x, y):
        row, col = spec.world_to_grid(x, y)
        return col, row  # imshow 좌표 (u=col, v=row)

    # --- FOV sector (semi-transparent) ---
    half = math.radians(sensor.fov_deg / 2.0)
    angles = np.linspace(-half, half, 50)
    poly_world = [(rx, ry)] + [
        (rx + sensor.max_range_m * math.cos(th + a),
         ry + sensor.max_range_m * math.sin(th + a)) for a in angles
    ]
    poly_img = [world_to_img(x, y) for (x, y) in poly_world]
    for ax in (ax_gt, ax_belief):
        ax.fill([p[0] for p in poly_img], [p[1] for p in poly_img],
                alpha=0.18, color='tab:blue', zorder=2)

    # --- path history (cyan) ---
    if len(sim.path) > 1:
        xs = []; ys = []
        for (wx, wy) in sim.path:
            u, v_ = world_to_img(wx, wy)
            xs.append(u); ys.append(v_)
        for ax in (ax_gt, ax_belief):
            ax.plot(xs, ys, '-', linewidth=2, color='cyan', alpha=0.8, zorder=3)

    # --- MPC predicted trajectory (orange) ---
    if sim.last_world_traj is not None:
        traj = np.array(sim.last_world_traj)
        xs = []; ys = []
        for wx, wy, _ in traj:
            u, v_ = world_to_img(wx, wy)
            xs.append(u); ys.append(v_)
        for ax in (ax_gt, ax_belief):
            ax.plot(xs, ys, '-', linewidth=2, color='orange', zorder=4)

    # --- robot heading/velocity arrow (topmost) ---
    v_cmd, _ = sim.last_cmd
    length = max(0.05, v_cmd * 0.5)
    x2 = rx + length * math.cos(th)
    y2 = ry + length * math.sin(th)
    cx, cy = world_to_img(rx, ry)
    cx2, cy2 = world_to_img(x2, y2)
    for ax in (ax_gt, ax_belief):
        ax.arrow(cx, cy, cx2 - cx, cy2 - cy, head_width=5, head_length=8,
                 fc='cyan', ec='cyan', length_includes_head=True, zorder=5)

    # ======================================================
    # VIZ 오버레이 (Simulator가 채운 디버그 마커들 항상 표시)
    # ======================================================

    # 1) Target point (파랑 점) + heading ref at target (보라 화살표)
    if sim.last_target_world is not None:
        tx, ty = sim.last_target_world
        ux, uy = world_to_img(tx, ty)
        for ax in (ax_gt, ax_belief):
            ax.scatter([ux], [uy], s=25, c='#1E90FF', edgecolors='white',
                       linewidths=0.7, zorder=6)
    if sim.last_heading_arrow is not None:
        (x0, y0), (x1, y1) = sim.last_heading_arrow
        u0, v0 = world_to_img(x0, y0)
        u1, v1 = world_to_img(x1, y1)
        for ax in (ax_gt, ax_belief):
            ax.arrow(u0, v0, u1 - u0, v1 - v0, head_width=5, head_length=8,
                     fc='#7E57C2', ec='#7E57C2', length_includes_head=True, zorder=6)

    # 2) Cluster centroid (초록 점)
    if sim.last_centroid_world is not None:
        cxw, cyw = sim.last_centroid_world
        uc, vc = world_to_img(cxw, cyw)
        for ax in (ax_gt, ax_belief):
            ax.scatter([uc], [vc], s=20, c='#34C759', edgecolors='black',
                       linewidths=0.6, zorder=6)

    # 3) Cluster points (자홍색 작은 점들)
    if sim.last_cluster_poly_world is not None and len(sim.last_cluster_poly_world) > 0:
        U = []; V = []
        for (wx, wy) in sim.last_cluster_poly_world:
            u, v_ = world_to_img(wx, wy)
            U.append(u); V.append(v_)
        for ax in (ax_gt, ax_belief):
            ax.scatter(U, V, s=2, c='#D81B60', alpha=0.9, zorder=6)

    # 4) Principal axis t (노랑) / Normal n (마젠타) – 중심에서 화살표
    # if sim.last_axes_world is not None and sim.last_centroid_world is not None:
    #     (x0t, y0t), (x1t, y1t) = sim.last_axes_world.get("t", (None, None))
    #     (x0n, y0n), (x1n, y1n) = sim.last_axes_world.get("n", (None, None))
    #     if x0t is not None:
    #         ut0, vt0 = world_to_img(x0t, y0t)
    #         ut1, vt1 = world_to_img(x1t, y1t)
    #     if x0n is not None:
    #         un0, vn0 = world_to_img(x0n, y0n)
    #         un1, vn1 = world_to_img(x1n, y1n)
    #     for ax in (ax_gt, ax_belief):
    #         if x0t is not None:
    #             ax.arrow(ut0, vt0, ut1 - ut0, vt1 - vt0, head_width=4, head_length=6,
    #                      fc='#FDD835', ec='#FDD835', length_includes_head=True, alpha=0.9, zorder=6)
    #         if x0n is not None:
    #             ax.arrow(un0, vn0, un1 - un0, vn1 - vn0, head_width=4, head_length=6,
    #                      fc='#EC407A', ec='#EC407A', length_includes_head=True, alpha=0.9, zorder=6)

    # ------------------------------------------------------

    for ax in (ax_gt, ax_belief):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('Ground Truth' if ax is ax_gt else 'Belief')

def save_gif(maps, sim, steps: int = 100, out_path: str = 'run.gif', dpi: int = 120):
    import imageio
    frames: List[np.ndarray] = []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.0))  # 2행 × 1열

    for _ in range(steps):
        print(f"Step : {len(frames) + 1} / {steps}")
        sim.step()
        draw_frame(ax1, ax2, maps, sim)
        
        if getattr(sim, 'stopped', False) and not getattr(sim, 'crashed', False):
            for ax in (ax1, ax2):
                ax.text(0.5, 0.5, getattr(sim, 'stop_msg', 'STOPPED'),
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=4, color='black', weight='bold', zorder=10,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round'))
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf, dtype=np.uint8)[..., :3]
        frames.append(frame.copy())
        
        if getattr(sim, 'crashed', False) or getattr(sim, 'stopped', False):
            break

    imageio.mimsave(out_path, frames, fps=int(1.0/sim.dt))
    plt.close(fig)
    return out_path