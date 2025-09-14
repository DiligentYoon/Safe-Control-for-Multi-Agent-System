# =============================
# controller_cbf.py
# =============================
from __future__ import annotations
import numpy as np
import casadi as ca
from typing import Tuple, List, Optional, Dict

class CBFController:
    def __init__(self,
                 v_max: float = 0.3,
                 w_max: float = 1.0,
                 d_safe: float = 0.05,
                 max_obs: int = 64,
                 gamma: float = 5.0,  # CBF parameter
                 w_slack: float = 3000.0,
                 k_v: float = 2.0,
                 k_w: float = 2.0,    # P-controller gain for omega
                 cluster_radius_m: float = 0.03,
                 ):
        """
        Controller based on Control Barrier Functions (CBF) formulated as a Quadratic Program (QP).

        - Dynamics: Unicycle model (in local frame)
        - Safety: Collision avoidance via CBF constraints.
        - Objective: Track a nominal controller's output [v_ref, w_ref].
        - Nominal Controller: A simple P-controller to steer towards a target.
        - Target Selection: Clustered frontier points.
        """
        self.v_max = v_max
        self.w_max = w_max
        self.d_safe = d_safe
        self.max_obs = max_obs
        self.gamma = gamma
        self.w_slack = w_slack
        self.k_v = k_v
        self.k_w = k_w
        self.cluster_radius_m = cluster_radius_m
        self._build_qp()

    # ---------- QP build (called once in __init__) ----------
    def _build_qp(self):
        # Optimization variables: u = [v, w]
        u = ca.MX.sym('u', 2)
        v, w = u[0], u[1]

        # Parameters
        u_ref = ca.MX.sym('u_ref', 2)      # Nominal control [v_ref, w_ref]
        delta = ca.MX.sym('delta')
        p_obs = ca.MX.sym('p_obs', 2 * self.max_obs) # Obstacle positions (lx, ly)

        # Objective function: min ||u - u_ref||^2
        J = ca.sumsqr(u - u_ref) + self.w_slack * delta**2

        # Constraints
        g = []
        lbg = []
        ubg = []

        # CBF constraints for each obstacle
        # h(p) = ||p - p_obs||^2 - d_safe^2 >= 0
        # In local frame: p=(0,0), p_obs=(lx,ly), so h = lx^2 + ly^2 - d_safe^2
        # CBF constraint: L_f h + L_g h * u >= -gamma * h
        # L_f h = 0 (for our system)
        # L_g h * u = -2 * lx * v
        # So, for each obstacle i: -2 * l x_i * v >= -gamma * (lx_i^2 + ly_i^2 - d_safe^2)
        for i in range(self.max_obs):
            lx = p_obs[2 * i]
            ly = p_obs[2 * i + 1]
            h = lx**2 + ly**2 - self.d_safe**2
            cbf_constraint = -2 * lx * v + self.gamma * h + delta
            g.append(cbf_constraint)
            lbg.append(0.0)
            ubg.append(ca.inf)

        # Create QP solver
        qp = {'x': ca.vertcat(u, delta), 'f': J, 'g': ca.vertcat(*g), 'p': ca.vertcat(u_ref, p_obs)}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'warn_initial_bounds': False}
        self.solver = ca.nlpsol('solver', 'ipopt', qp, opts)

        # Variable and constraint bounds
        self.lbx = np.array([0.0, -self.w_max, 0.0])
        self.ubx = np.array([self.v_max, self.w_max, ca.inf])
        self.lbg = np.array(lbg)
        self.ubg = np.array(ubg)
    # ---------- Nominal Controller ----------
    def _get_nominal_control(self, p_target: Tuple[float, float]) -> Tuple[float, float]:
        """
        Simple P-controller to generate a desired (v_ref, w_ref).
        """
        lx, ly = p_target
        dist_to_target = np.sqrt(lx**2 + ly**2)
        angle_to_target = np.arctan2(ly, lx)

        # P-control for both v and w
        v_ref = self.k_v * dist_to_target
        w_ref = self.k_w * angle_to_target
        
        # Clip controls to their maximum values
        v_ref = np.clip(v_ref, 0.0, self.v_max)
        w_ref = np.clip(w_ref, -self.w_max, self.w_max)
        
        return v_ref, w_ref

    # ---------- Target Selection (from controller2.py) ----------
    def _split_clusters_by_radius(self, F: np.ndarray) -> List[np.ndarray]:
        if F.shape[0] == 0: return []
        r = self.cluster_radius_m
        N = F.shape[0]
        adj = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(F[i] - F[j]) <= r:
                    adj[i].append(j)
                    adj[j].append(i)
        seen = np.zeros(N, dtype=bool)
        clusters: List[np.ndarray] = []
        for i in range(N):
            if seen[i]: continue
            stack = [i]
            seen[i] = True
            comp_idx = []
            while stack:
                u = stack.pop()
                comp_idx.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            clusters.append(F[np.array(comp_idx)])
        return clusters

    @staticmethod
    def _polyline_length_by_angle(cluster: np.ndarray) -> float:
        if cluster.shape[0] < 2: return 0.0
        ang = np.arctan2(cluster[:, 1], cluster[:, 0])
        idx = np.argsort(ang)
        P = cluster[idx]
        diffs = np.diff(P, axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())

    def _choose_cluster(self, F: np.ndarray) -> np.ndarray:
        clusters = self._split_clusters_by_radius(F)
        if not clusters: return F
        lengths = [self._polyline_length_by_angle(c) for c in clusters]
        Lmax = np.max(lengths)
        cand = [i for i, L in enumerate(lengths) if np.isclose(L, Lmax)]
        if len(cand) == 1: return clusters[cand[0]]
        ymeans = [clusters[i][:, 1].mean() for i in cand]
        j = cand[int(np.argmin(ymeans))]
        return clusters[j]

    def _pick_target(self, F: np.ndarray) -> Tuple[Tuple[float, float], Dict[str, np.ndarray]]:
        C = self._choose_cluster(F)
        if C.shape[0] == 0:
            return (self.v_max, 0.0), {} # Fallback target
        
        # Target is the mean of the chosen cluster
        c = C.mean(axis=0)
        p_tgt = (float(c[0]), float(c[1]))

        viz = {
            "target_local": np.asarray(p_tgt, dtype=float),
            "cluster_pts_local": C.astype(float),
        }
        return p_tgt, viz

    # ---------- Main `plan` method ----------
    def plan(self,
             frontier_local: List[Tuple[float, float]],
             obs_local: List[Tuple[float, float]] | None
             ) -> Tuple[float, float, None, Dict]:
        """
        Plan a safe control action [v, w].
        """
        if not frontier_local:
            return 0.0, 0.0, None, {}

        # 1. Select a target from the frontier
        F = np.array(frontier_local, dtype=float).reshape(-1, 2)
        p_target, viz_data = self._pick_target(F)

        # 2. Get nominal control
        v_ref, w_ref = self._get_nominal_control(p_target)
        u_ref_vec = np.array([v_ref, w_ref])

        # 3. Pack obstacles into a fixed-size vector for the solver
        pobs_vec = np.full(2 * self.max_obs, 1e6, dtype=float)
        if obs_local is not None and len(obs_local) > 0:
            O = np.asarray(obs_local, dtype=float).reshape(-1, 2)
            m = min(self.max_obs, O.shape[0])
            pobs_vec[:2 * m] = O[:m].ravel()

        # 4. Solve the QP
        p_vec = np.concatenate([u_ref_vec, pobs_vec])
        sol = self.solver(x0=[v_ref, w_ref, 0.0], lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg, p=p_vec)
        
        u_safe = np.array(sol['x']).ravel()
        v_cmd, w_cmd = float(u_safe[0]), float(u_safe[1])

        return v_cmd, w_cmd, None, viz_data
