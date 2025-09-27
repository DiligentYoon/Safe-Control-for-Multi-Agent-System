# =============================
# controller.py (updated)
# =============================
from __future__ import annotations
import numpy as np
import casadi as ca
from typing import Tuple, List, Optional, Dict

class ShortHorizonMPC:
    def __init__(self,
                 dt: float = 0.15,
                 H: int = 8,
                 v_max: float = 0.3,
                 w_max: float = 1.0,
                 w_goal: float = 1.0,
                 w_yaw: float = 1.0,
                 w_smooth: float = 0.05,
                 d_safe: float = 0.08,
                 max_obs: int = 64,
                 w_slack: float = 3000.0,
                 r_goal: float = 0.1,
                 w_prog: float = 0.05,
                 backoff_m: float = 0.05,
                 cluster_radius_m: float = 0.03,    # ★ 추가: 군집 반경(미터) ← 3cells 권장
                 w_yaw_T: float = 1.5
                 ):
        """
        MPC with:
          - Unicycle dynamics, forward-only v >= 0
          - Soft collision constraints with per-step slack s_k (k=0..H)
          - Terminal collision constraint (k=H)
          - Terminal goal reach: ||p_H - p_target|| <= r_goal (soft via sg >= 0)
          - Yaw alignment to an external reference direction (unit vector), no atan2
          - Stage shaping: penalize positive increases in distance to target
        """
        self.dt = dt
        self.H = H
        self.v_max = v_max
        self.w_max = w_max
        self.w_goal = w_goal
        self.w_yaw = w_yaw
        self.w_smooth = w_smooth
        self.d_safe = d_safe
        self.max_obs = max_obs
        self.w_slack = w_slack
        self.r_goal = r_goal
        self.w_prog = w_prog
        self.backoff_m = backoff_m
        self.cluster_radius_m = float(cluster_radius_m)
        self.w_yaw_T = w_yaw_T 
        self._build()
        
    # ---------- internal: angle-based clustering & selection ----------
    @staticmethod
    def _angles(points: np.ndarray) -> np.ndarray:
        return np.arctan2(points[:, 1], points[:, 0])

    def _split_clusters_by_radius(self, F: np.ndarray) -> List[np.ndarray]:
        """
        거리 ≤ cluster_radius_m 이내 이웃을 간선으로 하는 그래프의 연결요소를 군집으로 반환.
        """
        if F.shape[0] == 0:
            return []
        r = self.cluster_radius_m
        N = F.shape[0]
        # 인접 리스트 구성 (O(N^2), N이 작으므로 충분)
        adj = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(F[i] - F[j]) <= r:
                    adj[i].append(j)
                    adj[j].append(i)
        # 연결 요소 탐색
        seen = np.zeros(N, dtype=bool)
        clusters: List[np.ndarray] = []
        for i in range(N):
            if seen[i]:
                continue
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
        if cluster.shape[0] < 2:
            return 0.0
        ang = np.arctan2(cluster[:, 1], cluster[:, 0])
        idx = np.argsort(ang)
        P = cluster[idx]
        diffs = np.diff(P, axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())

    def _choose_cluster(self, F: np.ndarray) -> np.ndarray:
        """
        반경 기반으로 분할된 군집 중,
        - 길이 L 최대 (각도순 폴리라인 길이)
        - 동률이면 평균 y가 작은(아래쪽) 군집
        """
        clusters = self._split_clusters_by_radius(F)
        if not clusters:
            return F  # fallback
        lengths = [self._polyline_length_by_angle(c) for c in clusters]
        Lmax = np.max(lengths)
        cand = [i for i, L in enumerate(lengths) if np.isclose(L, Lmax, rtol=1e-6, atol=1e-9)]
        if len(cand) == 1:
            return clusters[cand[0]]
        ymeans = [clusters[i][:, 1].mean() for i in cand]
        j = cand[int(np.argmin(ymeans))]
        return clusters[j]
    
    @staticmethod
    def _principal_axis(cluster: np.ndarray) -> np.ndarray:
        """
        PCA 주축 (unit vector). 포인트가 2개 미만이면 원점->중심 방향으로 대체.
        """
        c = cluster.mean(axis=0)
        X = cluster - c
        if X.shape[0] >= 2:
            C = (X.T @ X) / max(1, X.shape[0])
            w, V = np.linalg.eigh(C)
            t = V[:, int(np.argmax(w))]  # principal direction
            t = t / (np.linalg.norm(t) + 1e-12)
            return t
        # fallback: 중심 방향
        v = c.copy()
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.array([1.0, 0.0])
        return v / n

    @staticmethod
    def _rotate90(v: np.ndarray) -> np.ndarray:
        return np.array([-v[1], v[0]])

    def _pick_target_and_headingref(
            self, F: np.ndarray
        ) -> Tuple[Tuple[float, float], Tuple[float, float], Dict[str, np.ndarray]]:
    
        C = self._choose_cluster(F)   # (Nc, 2)
        c = C.mean(axis=0)            # 군집 중심(원래)
        t = self._principal_axis(C)   # 주축 (단위)
    
        # --- 법선 후보 2개(±) 중에서 전방(+x)와 더 가까운 쪽 선택 ---
        def rot90(v: np.ndarray) -> np.ndarray:
            return np.array([-v[1], v[0]], dtype=float)
    
        n1 = rot90(t)
        n1 /= (np.linalg.norm(n1) + 1e-12)
        n2 = -n1
        h = np.array([1.0, 0.0], dtype=float)
    
        # 전방성 기준: h·n 최대
        n_star = n1 if (h @ n1) >= (h @ n2) else n2
    
        # (선택) 수치적 안전장치: 거의 수평이 아닐 때도 x<0 방지
        if n_star[0] < 0.0:
            n_star = -n_star  # 항상 전방 성분을 양(+)으로
    
        # --- backoff: 군집 중심을 원점(로봇) 쪽으로 self.backoff_m 만큼 당김 ---
        c_back = c.copy()
        nrm = float(np.linalg.norm(c))
        if nrm > 1e-9 and self.backoff_m > 0.0:
            alpha = min(1.0, self.backoff_m / nrm)
            c_back = (1.0 - alpha) * c
            if c_back[0] <= 1e-3:  # 너무 뒤로 가지 않도록 전방성 보장
                c_back[0] = 1e-3
    
        p_tgt = (float(c_back[0]), float(c_back[1]))
        h_ref = (float(n_star[0]), float(n_star[1]))
    
        viz = {
            "target_local": np.asarray(p_tgt, dtype=float),
            "h_ref_local": np.asarray(h_ref, dtype=float),
            "centroid_local": c.astype(float),
            "cluster_pts_local": C.astype(float),
            "t_star_local": t.astype(float),       # 주축 (부호는 시각화 용도면 원본 유지)
            "n_star_local": n_star.astype(float),  # 최종 선택된 법선
        }
        return p_tgt, h_ref, viz

    # ---------- MPC build ----------
    def _build(self):
        H = self.H; dt = self.dt
        x = ca.MX.sym('x', 3 * (H + 1))           # states: [px,py,theta] for k=0..H
        u = ca.MX.sym('u', 2 * H)                 # inputs: [v,w] for k=0..H-1
        s = ca.MX.sym('s', H + 1)                 # collision slacks for k=0..H
        sg = ca.MX.sym('sg', 1)                   # terminal goal slack ≥ 0
        x0 = ca.MX.sym('x0', 3)
        p_target = ca.MX.sym('pf', 2)             # (px,py) target position (centroid)
        h_ref = ca.MX.sym('href', 2)              # desired heading unit vector (principal normal)
        p_obs = ca.MX.sym('pobs', 2 * self.max_obs)

        def xs(k):
            return slice(3 * k, 3 * k + 3)

        def us(k):
            return slice(2 * k, 2 * k + 2)

        g = []
        lbg = []
        ubg = []
        J = 0

        # initial condition
        g.append(x[xs(0)] - x0); lbg += [0, 0, 0]; ubg += [0, 0, 0]

        eps = 1e-9
        epsd = 1e-9

        # stage constraints & costs
        dist_prev = None
        for k in range(H):
            px, py, th = x[xs(k)][0], x[xs(k)][1], x[xs(k)][2]
            v, w = u[us(k)][0], u[us(k)][1]

            # dynamics
            pxn = px + v * ca.cos(th) * dt
            pyn = py + v * ca.sin(th) * dt
            thn = th + w * dt
            g.append(x[xs(k + 1)] - ca.vertcat(pxn, pyn, thn))
            lbg += [0, 0, 0]; ubg += [0, 0, 0]

            # distances to position target
            dx = p_target[0] - px
            dy = p_target[1] - py
            dist2 = dx * dx + dy * dy

            # (A) position shaping (small) — keep if needed
            J += self.w_goal * dist2

            # (B) yaw alignment to external heading reference (unit vector)
            #     cost ~ (1 - h·h_ref)^2, where h=[cos th, sin th]
            cth = ca.cos(th); sth = ca.sin(th)
            dot_h = cth * h_ref[0] + sth * h_ref[1]
            J += self.w_yaw * (1.0 - dot_h) ** 2

            # (C) control smoothing
            if k > 0:
                v_prev, w_prev = u[us(k - 1)][0], u[us(k - 1)][1]
                J += self.w_smooth * ((v - v_prev) ** 2 + (w - w_prev) ** 2)

            # (D) stage shaping: penalize positive increase in distance to target
            dist_k = ca.sqrt(dist2 + epsd)
            if dist_prev is not None:
                delta = dist_k - dist_prev
                # smooth ReLU: 0.5*(x + sqrt(x^2 + eps))
                relu = 0.5 * (delta + ca.sqrt(delta * delta + eps))
                J += self.w_prog * (relu ** 2)
            dist_prev = dist_k

            # (E) collision soft constraints & slack penalty (per step)
            J += self.w_slack * (s[k] ** 2)
            for j in range(self.max_obs):
                ox = p_obs[2 * j + 0]
                oy = p_obs[2 * j + 1]
                d2 = (px - ox) ** 2 + (py - oy) ** 2
                g.append(d2 - (self.d_safe ** 2) + s[k])
                lbg.append(0.0); ubg.append(ca.inf)

        # terminal collision constraints (k=H)
        pxT, pyT, thT = x[xs(H)][0], x[xs(H)][1], x[xs(H)][2]
        J += self.w_slack * (s[H] ** 2)
        for j in range(self.max_obs):
            ox = p_obs[2 * j + 0]
            oy = p_obs[2 * j + 1]
            d2T = (pxT - ox) ** 2 + (pyT - oy) ** 2
            g.append(d2T - (self.d_safe ** 2) + s[H])
            lbg.append(0.0); ubg.append(ca.inf)

        # terminal goal reach: ||p_H - p_target|| <= r_goal  (soft via sg ≥ 0)
        dxT = p_target[0] - pxT
        dyT = p_target[1] - pyT
        goal_expr = (self.r_goal ** 2) - (dxT * dxT + dyT * dyT) + sg
        g.append(goal_expr)
        lbg.append(0.0); ubg.append(ca.inf)
        J += self.w_slack * (sg ** 2)
        
        cthT = ca.cos(thT); sthT = ca.sin(thT)
        dot_hT = cthT * h_ref[0] + sthT * h_ref[1]
        J += (getattr(self, "w_yaw_T", self.w_yaw)) * (1.0 - dot_hT)**2

        # variable bounds
        lbx = []; ubx = []
        for k in range(H + 1):
            lbx += [-ca.inf, -ca.inf, -ca.inf]
            ubx += [ ca.inf,  ca.inf,  ca.inf]
        for k in range(H):
            lbx += [0.0, -self.w_max]
            ubx += [self.v_max,  self.w_max]
        for k in range(H + 1):
            lbx += [0.0]
            ubx += [ca.inf]
        # terminal goal slack sg ≥ 0
        lbx += [0.0]
        ubx += [ca.inf]

        # make NLP
        nlp = {
            'x': ca.vertcat(x, u, s, sg),
            'f': J,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(x0, p_target, h_ref, p_obs)
        }
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # store dims & bounds
        self.nx = 3 * (H + 1)
        self.nu = 2 * H
        self.ns = H + 1  # collision slacks only (sg handled separately)
        self.lbx = np.array(lbx, float); self.ubx = np.array(ubx, float)
        self.lbg = np.array(lbg, float); self.ubg = np.array(ubg, float)

    def solve(self,
              x0=(0.0, 0.0, 0.0),
              p_target=(0.3, 0.0),
              h_ref=(1.0, 0.0),
              obs_local: Optional[np.ndarray] = None
              ) -> Tuple[float, float, np.ndarray]:
        # initial guesses
        x_init = np.zeros(self.nx)
        u_init = np.zeros(self.nu)
        s_init = np.zeros(self.ns)
        for k in range(self.H + 1):
            x_init[3 * k + 0] = x0[0]
            x_init[3 * k + 1] = x0[1]
            x_init[3 * k + 2] = x0[2]
        for k in range(self.H):
            u_init[2 * k + 0] = 0.1
            u_init[2 * k + 1] = 0.0

        # pack obstacles into fixed-size vector
        pobs = np.full(2 * self.max_obs, 1e6, dtype=float)
        if obs_local is not None and len(obs_local) > 0:
            O = np.asarray(obs_local, dtype=float).reshape(-1, 2)
            m = min(self.max_obs, O.shape[0])
            pobs[:2 * m] = O[:m].ravel()

        p_vec = np.hstack([np.array(x0, float),
                           np.array(p_target, float),
                           np.array(h_ref, float),
                           pobs])
        sg_init = np.array([0.0])
        z0 = np.hstack([x_init, u_init, s_init, sg_init])

        sol = self.solver(x0=z0, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg, p=p_vec)
        z = np.array(sol['x']).ravel()
        v0 = z[self.nx + 0]
        w0 = z[self.nx + 1]
        xs = z[:self.nx].reshape(-1, 3)
        return float(v0), float(w0), xs

    # high-level planning: use only FOV-local frontier & obstacles
    def plan(self,
             frontier_local: List[Tuple[float, float]],
             obs_local: List[Tuple[float, float]] | None):
        if not frontier_local:
            return None
        F = np.array(frontier_local, dtype=float).reshape(-1, 2)
        p_tgt, h_ref, viz = self._pick_target_and_headingref(F)
        O = (np.array(obs_local, dtype=float).reshape(-1, 2)
             if (obs_local is not None and len(obs_local) > 0) else None)
        v_cmd, w_cmd, xs_local = self.solve((0.0, 0.0, 0.0), p_tgt, h_ref, O)
        return v_cmd, w_cmd, xs_local, viz