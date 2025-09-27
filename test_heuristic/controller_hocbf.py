# =============================
# controller_cbf_hocbf.py
# =============================
from __future__ import annotations
import numpy as np
import casadi as ca
from typing import Tuple, List, Optional

class DecentralizedHOCBFController:
    def __init__(self,
                 v_max: float = 0.3,
                 w_max: float = 1.0,
                 a_max: float = 0.5,     
                 d_safe: float = 0.1,   
                 d_max: float = 0.2,
                 max_obs: int = 32,
                 max_agents: int = 5,
                 stiffness: float = 6.0,
                 damping: float = 5.0,
                 w_slack: float = 10000,
                 k_v: float = 1.5,
                 k_w: float = 1.0,
                 dt: float = 0.1,   
                 ):
        self.v_max = v_max
        self.w_max = w_max
        self.a_max = a_max
        self.d_safe = d_safe
        self.d_max = d_max
        self.max_obs = max_obs
        self.max_agents = max_agents
        self.damping = damping
        self.stiffness = stiffness
        self.w_slack = w_slack
        self.k_v = k_v
        self.k_w = k_w
        self.dt = dt
        self._build_qp()

    def _build_qp(self):
        # [HOCBF] Opt variables: u = [a, w], delta_avoid (slack)
        # a: linear acceleration, w: angular velocity
        u = ca.MX.sym('u', 2)
        a, w = u[0], u[1]
        delta_avoid = ca.MX.sym('delta_avoid')

        # [HOCBF] QP Parameters
        u_ref = ca.MX.sym('u_ref', 2)             # Nominal control [a_ref, w_ref]
        v_current = ca.MX.sym('v_current')      # Current linear velocity
        p_obs = ca.MX.sym('p_obs', 2 * self.max_obs)
        p_agents = ca.MX.sym('p_agents', 2 * self.max_agents)
        v_agents_local = ca.MX.sym('v_agents_local', 2 * self.max_agents)
        agent_active = ca.MX.sym('agent_active', self.max_agents)

        # Objective: Minimize deviation from nominal acceleration/angular velocity
        J = ca.sumsqr(u - u_ref) + self.w_slack * delta_avoid**2

        g, lbg, ubg = [], [], []

        # --- Soft Static Obstacle Constraints (1st Order CBF, adapted for acceleration control) ---
        for i in range(self.max_obs):
            lx = p_obs[2 * i]
            ly = p_obs[2 * i + 1]
            h_obs = lx**2 + ly**2 - self.d_safe**2 # Simplified h for static obs
            h_dot_obs = -2 * lx * v_current
            h_dot_dot_obs = 2*v_current**2 - 2*w*ly*v_current - 2*lx*a

            psi_2_obs = h_dot_dot_obs + self.damping * h_dot_obs + self.stiffness * h_obs

            g.append(psi_2_obs + delta_avoid)
            lbg.append(0.0)
            ubg.append(ca.inf)

        # --- Multi-Agent Dynamic Constraints (HOCBF) ---
        for i in range(self.max_agents):
            lx = p_agents[2 * i]
            ly = p_agents[2 * i + 1]
            v_jx_local = v_agents_local[2 * i]
            v_jy_local = v_agents_local[2 * i + 1]
            
            # --- 1. Collision Avoidance (min distance) - 임시 HARD, SOFT로 변경 예정 -
            # [HOCBF] Define h, h_dot_avoid, and psi_1_avoid for the agent
            h_avoid = lx**2 + ly**2 - self.d_safe**2
            h_dot_avoid = -2 * lx * v_current + 2 * (lx * v_jx_local + ly * v_jy_local)
            # [HOCBF] Calculate h_dot_dot_avoid assuming other agent's acceleration is zero
            # d/dt(-2*lx*v) -> 2*v^2 - 2*w*ly*v - 2*lx*a
            h_dot_dot_self_avoid = 2*v_current**2 - 2*w*ly*v_current - 2*lx*a - 2*v_current*v_jx_local
            v_dot_p = -v_current * v_jx_local + w * ly * v_jx_local + v_jx_local**2 - w * lx * v_jy_local + v_jy_local**2
            h_dot_dot_other_avoid = 2 * v_dot_p
            h_dot_dot_avoid = h_dot_dot_self_avoid + h_dot_dot_other_avoid

            psi_2_avoid = h_dot_dot_avoid + self.damping * h_dot_avoid + self.stiffness * h_avoid

            # Add soft constraint to QP
            g.append(agent_active[i] * psi_2_avoid)
            lbg.append(0.0)
            ubg.append(ca.inf)

            # --- 2. Connectivity (max distance) - HARD -
            h_conn = self.d_max**2 - (lx**2 + ly**2)
            h_dot_conn = -h_dot_avoid 
            # h_dot_dot_conn은 h_dot_dot_avoid의 부호 반대
            h_dot_dot_conn = -h_dot_dot_avoid

            psi_2_conn = h_dot_dot_conn + self.damping * h_dot_conn + self.stiffness * h_conn

            g.append(agent_active[i] * psi_2_conn)
            lbg.append(0.0)
            ubg.append(ca.inf)         

        # Create QP solver
        qp_vars = ca.vertcat(u, delta_avoid)
        params = ca.vertcat(u_ref, v_current, p_obs, p_agents, v_agents_local, agent_active)
        qp = {'x': qp_vars, 'f': J, 'g': ca.vertcat(*g), 'p': params}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'warn_initial_bounds': False}
        self.solver = ca.nlpsol('solver', 'ipopt', qp, opts)

        # Add a function to evaluate the constraints for debugging
        self.eval_g = ca.Function('eval_g', [qp_vars, params], [ca.vertcat(*g)], ['x', 'p'], ['g'])
        
        # [HOCBF] Update variable bounds for [a, w, delta]
        self.lbx = np.array([-self.a_max, -self.w_max, 0.0])
        self.ubx = np.array([self.a_max, self.w_max, ca.inf])
        self.lbg = np.array(lbg)
        self.ubg = np.array(ubg)

    def _get_nominal_control(self, p_target: Tuple[float, float], v_current: float) -> Tuple[float, float]:
        # [HOCBF] Nominal controller now outputs acceleration reference
        lx, ly = p_target
        dist_to_target = np.sqrt(lx**2 + ly**2)
        angle_to_target = np.arctan2(ly, lx)

        # Target velocity based on distance
        v_target = np.clip(self.k_v * dist_to_target, 0.0, self.v_max)
        
        # P-control for acceleration
        a_ref = self.k_v * (v_target - v_current)
        a_ref = np.clip(a_ref, -self.a_max, self.a_max)

        # P-control for angular velocity
        w_ref = np.clip(self.k_w * angle_to_target, -self.w_max, self.w_max)
        
        return a_ref, w_ref

    def compute_control(self,
                        p_target: Tuple[float, float],
                        v_current: float, # [HOCBF] Current velocity is now an input
                        obs_local: Optional[List[Tuple[float, float]]],
                        other_robots_local: Optional[List[Tuple[float, float]]],
                        other_robots_vel_local: Optional[List[Tuple[float, float]]]
                        ) -> Tuple[float, float, float, float, dict]:
        
        # [HOCBF] Get nominal acceleration and angular velocity
        a_ref, w_ref = self._get_nominal_control(p_target, v_current)
        u_ref_vec = np.array([a_ref, w_ref])

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
        
        # [HOCBF] Assemble full parameter vector including v_current
        p_vec = np.concatenate([u_ref_vec, [v_current], pobs_vec, pagents_vec, vagents_vec, agent_active_vec])
        x0 = [a_ref, w_ref, 0.0]
        sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg, p=p_vec)
        
        u_safe = np.array(sol['x']).ravel()
        a_cmd, w_cmd = float(u_safe[0]), float(u_safe[1])

        # [HOCBF] Integrate acceleration to get next velocity command for the robot
        v_cmd = v_current + a_cmd * self.dt
        v_cmd = np.clip(v_cmd, 0.0, self.v_max)

        # Return nominal and safe inputs
        v_ref_integrated = v_current + a_ref * self.dt
        v_ref_integrated = np.clip(v_ref_integrated, 0.0, self.v_max)
        
        # --- CBF & HOCBF Value Calculation for Visualization ---
        cbf_values = {}
        if obs_local:
            O = np.asarray(obs_local, dtype=float).reshape(-1, 2)
            h_obs = np.linalg.norm(O, axis=1)**2 - self.d_safe**2
            cbf_values['obs_avoid'] = h_obs

        if other_robots_local:
            A = np.asarray(other_robots_local, dtype=float).reshape(-1, 2)
            dists_sq = np.linalg.norm(A, axis=1)**2
            cbf_values['agent_avoid'] = dists_sq - self.d_safe**2
            cbf_values['agent_conn'] = self.d_max**2 - dists_sq

            # Also get the full psi values from the solver
            g_val = self.eval_g(x=sol['x'], p=p_vec)['g']
            agent_g_values = g_val[self.max_obs:]
            
            psi_avoids = []
            psi_conns = []
            for i in range(self.max_agents):
                if agent_active_vec[i] > 0:
                    psi_avoids.append(float(agent_g_values[2*i]))
                    psi_conns.append(float(agent_g_values[2*i+1]))
            
            if psi_avoids:
                cbf_values['psi_agent_avoid'] = np.array(psi_avoids)
            if psi_conns:
                cbf_values['psi_agent_conn'] = np.array(psi_conns)

        return v_cmd, w_cmd, v_ref_integrated, w_ref, cbf_values