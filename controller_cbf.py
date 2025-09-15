# =============================
# controller_cbf.py
# =============================
from __future__ import annotations
import numpy as np
import casadi as ca
from typing import Tuple, List, Optional

class DecentralizedCBFController:
    def __init__(self,
                 v_max: float = 0.3,
                 w_max: float = 1.0,
                 d_safe: float = 0.1,   # Min safety distance
                 d_max: float = 0.5,    # Max connectivity distance
                 L: float = 0.05,       # Look-ahead distance
                 max_obs: int = 32,
                 max_agents: int = 5,
                 gamma_avoid: float = 5.0,
                 gamma_conn: float = 1.0,
                 w_slack: float = 100000.0, # Increased slack penalty for avoidance
                 k_v: float = 1.5,
                 k_w: float = 2.5,
                 ):
        self.v_max = v_max
        self.w_max = w_max
        self.d_safe = d_safe
        self.d_max = d_max
        self.L = L
        self.max_obs = max_obs
        self.max_agents = max_agents
        self.gamma_avoid = gamma_avoid
        self.gamma_conn = gamma_conn
        self.w_slack = w_slack
        self.k_v = k_v
        self.k_w = k_w
        self._build_qp()

    def _build_qp(self):
        # Opt variables: u = [v, w], delta_avoid (slack for avoidance)
        u = ca.MX.sym('u', 2)
        v, w = u[0], u[1]
        delta_avoid = ca.MX.sym('delta_avoid')

        # QP Parameters
        u_ref = ca.MX.sym('u_ref', 2)
        p_obs = ca.MX.sym('p_obs', 2 * self.max_obs)
        p_agents = ca.MX.sym('p_agents', 2 * self.max_agents)
        v_agents_local = ca.MX.sym('v_agents_local', 2 * self.max_agents)
        agent_active = ca.MX.sym('agent_active', self.max_agents)

        # Objective
        J = ca.sumsqr(u - u_ref) + self.w_slack * delta_avoid**2

        g, lbg, ubg = [], [], []
        L = self.L # Look-ahead distance

        # --- Static Obstacle Constraints ---
        for i in range(self.max_obs):
            lx = p_obs[2 * i]
            ly = p_obs[2 * i + 1]
            
            # h based on look-ahead point
            h_avoid = (lx - L)**2 + ly**2 - self.d_safe**2
            
            # Lie derivative including w and v
            Lgh = -2 * L * ly * w - 2 * (lx - L) * v
            
            # Full CBF constraint
            g.append(Lgh + self.gamma_avoid * h_avoid + delta_avoid)
            lbg.append(0.0)
            ubg.append(ca.inf)

        # --- Multi-Agent Dynamic Constraints ---
        for i in range(self.max_agents):
            lx = p_agents[2 * i]
            ly = p_agents[2 * i + 1]
            v_jx_local = v_agents_local[2 * i]
            v_jy_local = v_agents_local[2 * i + 1]

            # 1. Collision Avoidance (min distance) - SOFT
            h_avoid = (lx - L)**2 + ly**2 - (self.d_safe*2)**2
            Lfh_avoid = 2 * ((lx - L) * v_jx_local + ly * v_jy_local)
            Lgh_avoid = -2 * L * ly * w - 2 * (lx - L) * v
            
            g.append(agent_active[i] * (Lfh_avoid + Lgh_avoid + self.gamma_avoid * h_avoid + delta_avoid))
            lbg.append(0.0)
            ubg.append(ca.inf)

            # 2. Connectivity (max distance) - HARD
            h_conn = (self.d_max*2)**2 - ((lx - L)**2 + ly**2)
            Lfh_conn = -2 * ((lx - L) * v_jx_local + ly * v_jy_local)
            Lgh_conn = 2 * L * ly * w + 2 * (lx - L) * v

            g.append(agent_active[i] * (Lfh_conn + Lgh_conn + self.gamma_conn * h_conn))
            lbg.append(0.0)
            ubg.append(ca.inf)

        # Create QP solver and a function to evaluate constraints
        qp_vars = ca.vertcat(u, delta_avoid)
        params = ca.vertcat(u_ref, p_obs, p_agents, v_agents_local, agent_active)
        qp = {'x': qp_vars, 'f': J, 'g': ca.vertcat(*g), 'p': params}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'warn_initial_bounds': False}
        self.solver = ca.nlpsol('solver', 'ipopt', qp, opts)
        
        # DEBUG: Create a function to evaluate the constraints
        self.eval_g = ca.Function('eval_g', [qp_vars, params], [ca.vertcat(*g)], ['x', 'p'], ['g'])

        self.lbx = np.array([0.0, -self.w_max, 0.0])
        self.ubx = np.array([self.v_max, self.w_max, ca.inf])
        self.lbg = np.array(lbg)
        self.ubg = np.array(ubg)

    def _get_nominal_control(self, p_target: Tuple[float, float]) -> Tuple[float, float]:
        lx, ly = p_target
        dist_to_target = np.sqrt(lx**2 + ly**2)
        angle_to_target = np.arctan2(ly, lx)
        v_ref = np.clip(self.k_v * dist_to_target, 0.0, self.v_max)
        w_ref = np.clip(self.k_w * angle_to_target, -self.w_max, self.w_max)
        return v_ref, w_ref

    def compute_control(self,
             agent_idx: int, # For debugging
             leader_idx: int, # For debugging
             p_target: Tuple[float, float],
             obs_local: Optional[List[Tuple[float, float]]],
             other_robots_local: Optional[List[Tuple[float, float]]],
             other_robots_vel_local: Optional[List[Tuple[float, float]]]
             ) -> Tuple[float, float]:
        
        v_ref, w_ref = self._get_nominal_control(p_target)
        u_ref_vec = np.array([v_ref, w_ref])

        pobs_vec = np.full(2 * self.max_obs, 1e6, dtype=float)
        if obs_local:
            O = np.asarray(obs_local, dtype=float).reshape(-1, 2)
            m = min(self.max_obs, O.shape[0])
            pobs_vec[:2 * m] = O[:m].ravel()

        pagents_vec = np.full(2 * self.max_agents, 0.0, dtype=float)
        agent_active_vec = np.zeros(self.max_agents, dtype=float)
        if other_robots_local:
            A = np.asarray(other_robots_local, dtype=float).reshape(-1, 2)
            m = min(self.max_agents, A.shape[0])
            pagents_vec[:2 * m] = A[:m].ravel()
            agent_active_vec[:m] = 1.0

        vagents_vec = np.zeros(2 * self.max_agents, dtype=float)
        if other_robots_vel_local:
            V = np.asarray(other_robots_vel_local, dtype=float).reshape(-1, 2)
            m = min(self.max_agents, V.shape[0])
            vagents_vec[:2 * m] = V[:m].ravel()

        p_vec = np.concatenate([u_ref_vec, pobs_vec, pagents_vec, vagents_vec, agent_active_vec])
        x0 = [v_ref, w_ref, 0.0]
        sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg, p=p_vec)
        
        u_safe = np.array(sol['x']).ravel()
        v_cmd, w_cmd = float(u_safe[0]), float(u_safe[1])

        # --- DEBUGGING BLOCK ---
        # if agent_idx != leader_idx:
        #     g_val = self.eval_g(x=sol['x'], p=p_vec)['g']
        #     print(f"""--- Follower {agent_idx} Debug ---
        #     Target (local): {p_target[0]:.2f}, {p_target[1]:.2f}
        #     u_ref: v={v_ref:.2f}, w={w_ref:.2f}
        #     u_sol: v={v_cmd:.2f}, w={w_cmd:.2f}
        #     Constraints (g):
        #     {g_val}
        #     """)

        return v_cmd, w_cmd