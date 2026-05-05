"""
RMP2 Bridge for Isaac Sim + VR Teleoperation (Phase 1b)

Pure NumPy implementation of the RMP2 algorithm (Li et al., R:SS 2021).
Lightweight solver that runs at 50Hz on CPU alongside Isaac Sim.

Architecture:
    Forward Pass:  q -> [eef_pos, eef_quat, cspace, collision_spheres]
    RMP Eval:      each leaf returns (metric, acceleration)
    Backward Pass: M*q_ddot = sum(J_i^T @ M_i @ a_i)
    Integration:   q_target = q + dt*qd + 0.5*dt^2*q_ddot

RMP Leaves:
    1. TargetAttractor     - drives EEF toward VR-mapped goal position (3D)
    2. OrientationAttractor - drives EEF toward VR-mapped goal rotation (SO(3))
    3. CSpaceTarget        - attracts toward default/rest joint config
    4. JointLimit          - exponential barrier near joint limits
    5. JointVelocityCap    - dampens near max velocity
    6. JointDamping        - velocity-dependent jerk suppression
    7. ObstacleAvoidance   - per-obstacle repulsion
"""

import numpy as np
import math


# ====== ======================================== ==============================
# RMP Leaf Classes
# ====== ======================================== ==============================

class TargetAttractor:
    """Drives end-effector toward goal position (3D Cartesian).

    PD-like acceleration with distance-dependent metric weighting.
    Higher metric when far, directional near goal.
    """

    name = "target"

    def __init__(self,
                 accel_p_gain=50.0,
                 accel_d_gain=10.0,
                 accel_norm_eps=1e-3,
                 metric_alpha_length_scale=0.05,
                 min_metric_alpha=0.1,
                 max_metric_scalar=2.0,
                 min_metric_scalar=0.1,
                 proximity_boost_scalar=3.0,
                 proximity_boost_length_scale=0.1):
        self.accel_p_gain = accel_p_gain
        self.accel_d_gain = accel_d_gain
        self.accel_norm_eps = accel_norm_eps
        self.metric_alpha_length_scale = metric_alpha_length_scale
        self.min_metric_alpha = min_metric_alpha
        self.max_metric_scalar = max_metric_scalar
        self.min_metric_scalar = min_metric_scalar
        self.proximity_boost_scalar = proximity_boost_scalar
        self.proximity_boost_length_scale = proximity_boost_length_scale

    def eval(self, x, xd, **features):
        goal = np.asarray(features.get("goal", np.zeros(3)), dtype=np.float64)
        delta = goal - np.asarray(x, dtype=np.float64)
        delta_norm = np.linalg.norm(delta) + self.accel_norm_eps
        delta_hat = delta / delta_norm

        # Smooth PD: full gain when far, proportional when near
        # Transition at metric_alpha_length_scale
        if delta_norm < self.metric_alpha_length_scale:
            # Proportional near goal (softens as we approach)
            accel_p = self.accel_p_gain * delta
        else:
            # Constant-magnitude when far (fast response)
            accel_p = self.accel_p_gain * delta_hat

        # Damping: velocity-dependent, stronger near goal
        boost_scaled = delta_norm / self.proximity_boost_length_scale
        boost = np.exp(-0.5 * boost_scaled ** 2)
        effective_d_gain = self.accel_d_gain * (1.0 + self.proximity_boost_scalar * boost)
        accel_d = -effective_d_gain * np.asarray(xd, dtype=np.float64)

        accel = accel_p + accel_d
        
        # PD-like acceleration
        # accel = (self.accel_p_gain * delta / delta_norm
        #          - self.accel_d_gain * np.asarray(xd, dtype=np.float64))
        

        # Metric: distance-dependent weighting
        scaled_dist = delta_norm / self.metric_alpha_length_scale
        alpha = ((1 - self.min_metric_alpha) * np.exp(-0.5 * scaled_dist ** 2)
                 + self.min_metric_alpha)

        # Directional metric: high in delta direction, low orthogonal
        S = np.outer(delta_hat, delta_hat)
        I = np.eye(3, dtype=np.float64)
        metric = (alpha * self.max_metric_scalar * I
                  + (1 - alpha) * self.min_metric_scalar * S)

        # Proximity boost near target
        boost_scaled = delta_norm / self.proximity_boost_length_scale
        boost_a = np.exp(-0.5 * boost_scaled ** 2)
        boost = boost_a * self.proximity_boost_scalar + (1 - boost_a)
        metric = boost * metric

        return metric, accel


class OrientationAttractor:
    """Drives end-effector orientation toward goal (SO(3) quaternion).

    Uses axis-angle log-map of quaternion error mapped through the
    rotational Jacobian. PD-like on the angular error with adaptive metric.

    SO(3) error: delta = 2 * log(q_curr^-1 * q_goal).vec_part
    Acceleration:   a = k_p * delta - k_d * omega
    Metric:         adaptive scalar, boosts as error shrinks
    """

    name = "orientation"

    def __init__(self,
                 accel_p_gain=100.0,
                 accel_d_gain=10.0,
                 accel_norm_eps=1e-4,
                 metric_base=0.2,
                 metric_proximity_boost=1.0,
                 metric_length_scale=0.2):
        self.accel_p_gain = accel_p_gain
        self.accel_d_gain = accel_d_gain
        self.accel_norm_eps = accel_norm_eps
        self.metric_base = metric_base
        self.metric_proximity_boost = metric_proximity_boost
        self.metric_length_scale = metric_length_scale

    @staticmethod
    def quat_mul(qa, qb):
        """Multiply two quaternions in (x,y,z,w) order -> (x,y,z,w)."""
        x1, y1, z1, w1 = qa
        x2, y2, z2, w2 = qb
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dtype=np.float64)

    @staticmethod
    def quat_inv(q):
        """Invert unit quaternion (x,y,z,w) -> (x,y,z,w)."""
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    def quat_error(self, q_curr, q_goal):
        """Return axis-angle error vector (3D) from current to goal orientation.

        Uses logarithmic map: q_rel = q_curr^-1 * q_goal, then extracts
        the rotation-axis * angle vector. For small errors this is
        approximately the small-angle rotation vector.
        """
        q_rel = self.quat_mul(self.quat_inv(q_curr), q_goal)
        vec_norm = np.linalg.norm(q_rel[:3]) + self.accel_norm_eps
        angle = 2.0 * np.arctan2(vec_norm, abs(q_rel[3]))
        sign_w = 1.0 if q_rel[3] >= 0 else -1.0
        delta = sign_w * angle * q_rel[:3] / vec_norm
        return delta

    def eval(self, quat_curr, omega, **features):
        """
        Args:
            quat_curr: current orientation (x,y,z,w)
            omega:     angular velocity (3D) from J_rot @ qd
        Returns:
            metric: 3x3 matrix
            accel:  3D vector (angular acceleration)
        """
        quat_curr = np.asarray(quat_curr, dtype=np.float64)
        quat_goal = np.asarray(features.get(
            "goal_quat", np.array([0, 0, 0, 1.0])), dtype=np.float64)

        delta = self.quat_error(quat_curr, quat_goal)
        delta_norm = np.linalg.norm(delta) + self.accel_norm_eps

        omega = np.asarray(omega, dtype=np.float64)
        scaled_delta = min(delta_norm, np.pi)  # cap at 180 deg
        delta_hat = delta / delta_norm
        
         # Smooth PD: proportional near goal, constant-magnitude when far
        # Matches TargetAttractor pattern for stable composition
        if delta_norm < self.metric_length_scale:
            accel_p = self.accel_p_gain * delta
        else:
            accel_p = self.accel_p_gain * delta_hat

        accel = accel_p - self.accel_d_gain * omega
        
        # # Robust PD acceleration
        # accel = (self.accel_p_gain * delta / (scaled_delta + self.accel_norm_eps)
        #          - self.accel_d_gain * omega)
            
        # Metric: base + proximity boost as error shrinks
        scaled_dist = delta_norm / self.metric_length_scale
        boost = np.exp(-0.5 * scaled_dist ** 2)
        metric_scalar = self.metric_base + self.metric_proximity_boost * boost
        metric = metric_scalar * np.eye(3, dtype=np.float64)

        return metric, accel


class CSpaceTarget:
    """Attracts robot toward a default/rest joint configuration."""

    name = "cspace_target"

    def __init__(self, metric_scalar=0.05, position_gain=1.0,
                 damping_gain=0.2, robust_thresh=0.5, inertia=1e-4):
        self.metric_scalar = metric_scalar
        self.position_gain = position_gain
        self.damping_gain = damping_gain
        self.robust_thresh = robust_thresh
        self.inertia = inertia
        self.n = None

    def eval(self, x, xd, **features):
        n = len(np.asarray(x))
        x = np.asarray(x, dtype=np.float64)
        xd = np.asarray(xd, dtype=np.float64)
        x_norm = np.linalg.norm(x)
        if x_norm < self.robust_thresh:
            qdd_pos = -x * self.position_gain
        else:
            qdd_pos = -self.robust_thresh * (x / x_norm) * self.position_gain
        qdd_vel = -self.damping_gain * xd
        accel = qdd_pos + qdd_vel
        metric = np.eye(n, dtype=np.float64) * (self.metric_scalar + self.inertia)
        return metric, accel


class ObstacleAvoidance:
    """Repels from obstacles using distance-based repulsion.

    Input x is a scalar distance to the obstacle (not a vector).
    Input xd is a scalar rate-of-change of that distance.
    """

    name = "obstacle"

    def __init__(self, margin=0.1, repulsion_gain=40.0,
                 repulsion_std_dev=0.3, metric_scalar=3.5,
                 metric_modulation_radius=0.5,
                 metric_exploder_std_dev=0.5,
                 metric_exploder_eps=1e-5,
                 damping_gain=100.0,
                 damping_std_dev=0.1,
                 damping_robustness_eps=1e-5,
                 damping_velocity_gate_length_scale=0.1):
        self.margin = margin
        self.repulsion_gain = repulsion_gain
        self.repulsion_std_dev = repulsion_std_dev
        self.metric_scalar = metric_scalar
        self.metric_modulation_radius = metric_modulation_radius
        self.metric_exploder_std_dev = metric_exploder_std_dev
        self.metric_exploder_eps = metric_exploder_eps
        self.damping_gain = damping_gain
        self.damping_std_dev = damping_std_dev
        self.damping_robustness_eps = damping_robustness_eps
        self.damping_velocity_gate_length_scale = damping_velocity_gate_length_scale

    def eval(self, d, xd_coll, **features):
        d = float(d)
        xd_coll = float(xd_coll)
        d = max(d - self.margin, 0.0)

        # Smooth activation gate: active only within modulation radius
        if d > self.metric_modulation_radius or d <= 0:
            return (np.array([[0.0]], dtype=np.float64),
                    np.array([0.0], dtype=np.float64))

        gate = ((d / self.metric_modulation_radius) ** 2
                - 2 * d / self.metric_modulation_radius + 1)
        base_metric = (self.metric_scalar
                       / (d / self.metric_exploder_std_dev
                          + self.metric_exploder_eps))
        metric_val = base_metric * gate

        # Repulsion acceleration
        xdd_repel = self.repulsion_gain * np.exp(-d / self.repulsion_std_dev)

        # Velocity damping near obstacle
        sig = 1.0 / (1.0 + np.exp(-xd_coll /
                                   self.damping_velocity_gate_length_scale))
        z = d / self.damping_std_dev + self.damping_robustness_eps
        xdd_damping = -(1 - sig) * self.damping_gain * xd_coll / z

        accel_val = xdd_repel + xdd_damping

        # Reduce metric when moving away
        if xd_coll > 0:
            metric_val *= (1 - sig)

        return (np.array([[metric_val]], dtype=np.float64),
                np.array([accel_val], dtype=np.float64))


class JointLimit:
    """Exponential barrier function near joint limits.

    Matches the official rmp2 repo implementation (rmps.py:127-162).
    Parameters match franka_config.yaml defaults.
    """

    name = "joint_limit"

    def __init__(self,
                 metric_scalar=0.1,
                 metric_length_scale=0.01,
                 metric_exploder_eps=0.001,
                 metric_velocity_gate_length_scale=0.01,
                 accel_damper_gain=200.0,
                 accel_potential_gain=1.0,
                 accel_potential_exploder_length_scale=0.1,
                 accel_potential_exploder_eps=0.01):
        self.metric_scalar = metric_scalar
        self.metric_length_scale = metric_length_scale
        self.metric_exploder_eps = metric_exploder_eps
        self.metric_velocity_gate_length_scale = metric_velocity_gate_length_scale
        self.accel_damper_gain = accel_damper_gain
        self.accel_potential_gain = accel_potential_gain
        self.accel_potential_exploder_length_scale = accel_potential_exploder_length_scale
        self.accel_potential_exploder_eps = accel_potential_exploder_eps

    def eval(self, dist, vel, **features):
        dist = np.maximum(np.asarray(dist, dtype=np.float64), 0.0)
        vel = np.asarray(vel, dtype=np.float64)
        n = len(dist)

        # Metric: exponential barrier with sigmoid velocity gate
        # When vel > 0 (moving away from limit), sigmoid approaches 1 -> metric suppressed
        metric_before = self.metric_scalar / (
            dist / self.metric_length_scale + self.metric_exploder_eps)
        sig = 1.0 / (1.0 + np.exp(-vel / self.metric_velocity_gate_length_scale))
        metric = (1 - sig) * metric_before

        # Acceleration: repel from limit + damp velocity
        scaled_x = dist / self.accel_potential_exploder_length_scale
        xdd_pos = self.accel_potential_gain / (
            scaled_x ** 2 + self.accel_potential_exploder_eps)
        xdd_vel = -self.accel_damper_gain * vel
        accel = xdd_pos + xdd_vel

        return np.diag(metric), accel


class JointVelocityCap:
    """Damps velocity near maximum allowed. Only active close to v_max."""

    name = "vel_cap"

    def __init__(self, max_velocity=2.0, velocity_damping_region=0.15,
                 damping_gain=5.0, metric_weight=1.0):
        self.max_velocity = max_velocity
        self.velocity_damping_region = velocity_damping_region
        self.damping_gain = damping_gain
        self.metric_weight = metric_weight
        self.damped_cutoff = self.max_velocity - self.velocity_damping_region

    def eval(self, x, xd, **features):
        xd = np.asarray(xd, dtype=np.float64)
        n = len(xd)
        delta_vel = np.abs(xd) - self.damped_cutoff

        metric = np.zeros(n, dtype=np.float64)
        accel = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if delta_vel[i] > 0:
                clipped = min(delta_vel[i],
                              self.velocity_damping_region - 1e-6)
                ratio = clipped / self.velocity_damping_region
                metric[i] = self.metric_weight / (1.0 - ratio ** 2 + 1e-8)
                accel[i] = (-self.damping_gain * delta_vel[i]
                            * np.sign(xd[i]))

        return np.diag(metric), accel


class JointDamping:
    """Nonlinear velocity-dependent damping. Suppresses jerk from VR jitter."""

    name = "damping"

    def __init__(self, accel_d_gain=5.0, metric_scalar=0.005, inertia=1e-4):
        self.accel_d_gain = accel_d_gain
        self.metric_scalar = metric_scalar
        self.inertia = inertia

    def eval(self, x, xd, **features):
        xd = np.asarray(xd, dtype=np.float64)
        n = len(xd)
        xd_norm = np.linalg.norm(xd)
        nonlinear_gain = self.accel_d_gain * xd_norm
        accel = -nonlinear_gain * xd
        nonlinear_scalar = self.metric_scalar * xd_norm
        metric = np.eye(n, dtype=np.float64) * (nonlinear_scalar + self.inertia)
        return metric, accel


# ====== ======================================== ==============================
# RMP2 Solver
# ====== ======================================== ==============================

class RMP2Solver:
    """RMP2 solver for teleoperation.

    Composes multiple RMP leaves into a weighted geometric optimization.
    Solves M_agg * q_ddot = f_agg via Cholesky for optimal joint acceleration.

    Supports both position-only and full 6-DOF (pos + orient) control.

    Usage:
        solver = RMP2Solver(n_joints=6, fk_fn=forward_kinematics, ...)
        q_ddot = solver.solve(q, qd, goals=[vr_goal], goal_quat=[x,y,z,w],
                               obstacles=[obs_positions])
        q_new, qd_new = solver.integrate(q, qd, q_ddot)
    """

    def __init__(self, n_joints, fk_fn, collision_fn=None,
                 joint_limits_low=None, joint_limits_high=None,
                 default_q=None, dt=0.02):
        """
        Args:
            n_joints: number of actuated joints (excluding gripper)
            fk_fn(q) -> (eef_pos, eef_quat, J_spatial): forward kinematics
                   returning 3D position, quaternion (x,y,z,w),
                   and 6xN spatial Jacobian (pos + rot).
            collision_fn(q, eef_pos, obstacles) -> list of float distances
            joint_limits_low / high: arrays of length n_actuated
            default_q: rest joint configuration
            dt: integration timestep for RMP2 solve
        """
        self.n_joints = n_joints
        self.fk_fn = fk_fn
        self.collision_fn = collision_fn
        self.joint_limits_low = (np.array(joint_limits_low, dtype=np.float64)
                                 if joint_limits_low is not None else None)
        self.joint_limits_high = (np.array(joint_limits_high, dtype=np.float64)
                                  if joint_limits_high is not None else None)
        self.default_q = (np.array(default_q, dtype=np.float64)
                          if default_q is not None
                          else np.zeros(n_joints, dtype=np.float64))
        self.dt = dt

        # Fixed leaves: [pos, orient, cspace, jl, vel_cap, damping]
        # Obstacle leaves are cached (created once per obstacle count)
        self._fixed_leaves = [
            TargetAttractor(),
            OrientationAttractor(),
            CSpaceTarget(),
            JointLimit(),
            JointVelocityCap(),
            JointDamping(),
        ]
        # self._leaf_names = ([
        #     "target", "orientation", "cspace", "joint_limit", "vel_cap", "damping"
        # ])
        self._leaf_names = (["target", "orientation"])
        # self._leaf_names = (["orientation"])
        # self._leaf_names = (["target", "orientation", "cspace", "joint_limit",
                            #  "vel_cap", "damping"])

        # Cached obstacle leaves (reused instead of recreated every frame)
        self._obs_leaves = []

        # Curvature state (persist q, J between solves for finite-diff)
        self._prev_q = None
        self._prev_J = None
        self._fd_eps = 1e-4

    # ---- Curvature helper: finite-difference second derivative ----
    def _compute_curvatures(self, q, qd, leaves_data):
        """Compute Christoffel-like curvature per leaf using central differences.

        The official rmp2 repo uses TF autodiff to get dJ/dq.  We use
        central differences:  dJ_k = (J(q+eps*e_k) - J(q-eps*e_k)) / (2*eps)
        and then   crv = sum_k  dJ_k * qd_k  (matrix-vector product).

        Leaves whose Jacobian is constant (cspace, joint_limit, vel_cap,
        damping) have zero curvature.  Obstacle leaves are 1-D and
        contribute negligible curvature, so they are set to zero as well.
        """
        n = self.n_joints
        eps = self._fd_eps
        crvs = []

        for i, (x_leaf, xd_leaf, J_leaf) in enumerate(leaves_data):
            leaf_dim = J_leaf.shape[0]
            if i < len(self._leaf_names):
                name = self._leaf_names[i]
            else:
                name = "obs_" + str(i - len(self._leaf_names))

            # Constant-Jacobian leaves -> zero curvature
            if name in ("cspace", "joint_limit", "vel_cap", "damping", "obstacle"):
                crvs.append(np.zeros(leaf_dim, dtype=np.float64))
                continue

            # Task-space leaves (target, orientation) need curvature
            crv = np.zeros(leaf_dim, dtype=np.float64)
            for k in range(n):
                q_plus = q.copy()
                q_plus[k] += eps
                q_minus = q.copy()
                q_minus[k] -= eps

                fk_plus = self.fk_fn(q_plus)
                fk_minus = self.fk_fn(q_minus)
                Jp = np.asarray(fk_plus[2], dtype=np.float64)
                Jm = np.asarray(fk_minus[2], dtype=np.float64)

                # Ensure Jacobian is (6, n_joints) - transpose if needed
                if Jp.shape == (n, 6):
                    Jp = Jp.T
                    Jm = Jm.T

                # Select the task-space slice
                if name == "target":
                    dJ = (Jp[:3, :] - Jm[:3, :]) / (2.0 * eps)
                elif name == "orientation":
                    dJ = (Jp[3:6, :] - Jm[3:6, :]) / (2.0 * eps)
                else:
                    crvs.append(np.zeros(leaf_dim, dtype=np.float64))
                    break

                # crv += dJ @ qd * qd[k]
                crv += (dJ @ qd) * qd[k]

            else:
                crvs.append(crv)
                continue

            crvs.append(np.zeros(leaf_dim, dtype=np.float64))

        return crvs

    # ---- Forward pass - build (x, xd_leaf, J) per leaf ----
    def forward_pass(self, q, qd, obstacles=None):
        """Return list of (state, velocity, Jacobian) tuples for each leaf."""
        obstacles = obstacles or []
        results = []
        n = self.n_joints

        # FK: expects (eef_pos, eef_quat, J_spatial_6xN)
        fk_out = self.fk_fn(q)
        eef_pos = np.asarray(fk_out[0], dtype=np.float64)
        eef_quat = np.asarray(fk_out[1], dtype=np.float64)  # (x,y,z,w)
        J_spatial = np.asarray(fk_out[2], dtype=np.float64)  # (6, n)

        # 1. Target position RMP (3-D Cartesian)
        J_pos = J_spatial[:3, :]
        xd_pos = J_pos @ qd
        results.append((eef_pos, xd_pos, J_pos))

        # 2. Target orientation RMP (SO(3))
        J_rot = J_spatial[3:6, :]
        omega = J_rot @ qd  # angular velocity in task frame
        results.append((eef_quat, omega, J_rot))

        # 3. CSpace target
        q_from_default = np.asarray(q, dtype=np.float64) - self.default_q
        results.append((q_from_default, qd, np.eye(n, dtype=np.float64)))

        # 4. Joint limits
        if (self.joint_limits_low is not None
                and self.joint_limits_high is not None):
            dist_low = np.asarray(q, dtype=np.float64) - self.joint_limits_low
            dist_high = (self.joint_limits_high
                         - np.asarray(q, dtype=np.float64))
            dist_limit = np.minimum(dist_low, dist_high)
            results.append((dist_limit, qd, np.eye(n, dtype=np.float64)))
        else:
            # Dummy: always far from limits
            results.append((np.ones(n, dtype=np.float64) * 10.0,
                            qd, np.eye(n, dtype=np.float64) * 0.0))

        # 5. Velocity cap (uses xd, x ignored)
        results.append((np.zeros(n, dtype=np.float64), qd,
                        np.eye(n, dtype=np.float64)))

        # 6. Damping (uses xd, x ignored)
        results.append((np.zeros(n, dtype=np.float64), qd,
                        np.eye(n, dtype=np.float64)))

        # 7. Collision - analytic Jacobian: d(||eef-obs||)/dq = n_hat^T * J_pos
        if (self.collision_fn is not None and J_spatial is not None
                and len(obstacles) > 0):
            distances = self.collision_fn(q, eef_pos, obstacles)
            if distances is not None:
                for obs_prim, d in zip(obstacles, distances):
                    obs_pos, _ = obs_prim.get_world_pose()
                    diff = (np.asarray(eef_pos, dtype=np.float64)
                            - np.asarray(obs_pos, dtype=np.float64))
                    d_norm = float(np.linalg.norm(diff)) + 1e-8
                    n_hat = diff / d_norm  # unit normal from obstacle -> EEF
                    J_coll = (n_hat @ J_pos).reshape(1, n)  # (1, n)
                    xd_coll = float(J_coll[0] @ qd)
                    results.append((float(d), xd_coll, J_coll))

        return results

    # ---- Solve - aggregate, pullback, Cholesky ----
    def solve(self, q, qd=None, goals=None, goal_quat=None, obstacles=None):
        """Return optimal joint acceleration q_ddot [n_joints]."""
        q = np.asarray(q, dtype=np.float64)
        qd = (np.asarray(qd, dtype=np.float64) if qd is not None
              else np.zeros(self.n_joints, dtype=np.float64))
        obstacles = obstacles or []
        n = self.n_joints

        features = {
            "goal": (np.asarray(goals[0], dtype=np.float64)
                     if goals else np.zeros(3, dtype=np.float64)),
            "goal_quat": (np.asarray(goal_quat, dtype=np.float64)
                          if goal_quat is not None
                          else np.array([0, 0, 0, 1.0], dtype=np.float64)),
            "obstacles": obstacles,
        }

        # Build leaf list (fixed + cached obstacle leaves)
        n_obs = len(obstacles)
        if n_obs != len(self._obs_leaves):
            self._obs_leaves = [ObstacleAvoidance()] * n_obs
        all_leaves = self._fixed_leaves + self._obs_leaves
        leaf_names = (self._leaf_names
                      + [f"obs_{i}" for i in range(n_obs)])

        # Forward pass
        leaves_data = self.forward_pass(q, qd, obstacles)

        # Compute curvatures (matches rmpgraph.py solve_rmp2 curvature step)
        # crvs = self._compute_curvatures(q, qd, leaves_data)

        # Aggregate metrics and forces (with curvature correction)
        # Official formula: z = sum(M_i * a_i) - sum(M_i * crv_i)
        # so f = sum(J^T M a) - sum(J^T M crv)
        M_agg = np.zeros((n, n), dtype=np.float64)
        f_agg = np.zeros(n, dtype=np.float64)

        # DISABLED: curvature correction — too expensive (28 FK calls/frame)
        # crvs = self._compute_curvatures(q, qd, leaves_data)
        crvs = [np.zeros(J_leaf.shape[0], dtype=np.float64)
                for _, _, J_leaf in leaves_data]

        for leaf, name, (x_leaf, xd_leaf, J_leaf), crv in zip(
                all_leaves, leaf_names, leaves_data, crvs):
            metric_val, accel_val = leaf.eval(x_leaf, xd_leaf, **features)

            # Pullback: M_tau += J^T @ M @ J
            # f_tau += J^T @ M @ (a - crv)  [curvature correction]
            if J_leaf.shape[0] == 1 and metric_val.shape == (1, 1):
                # Scalar leaf (obstacle)
                m_scalar = float(metric_val[0, 0])
                a_scalar = float(accel_val[0])
                # Curvature negligible for 1-D obstacle leaves
                J = J_leaf
                M_agg += m_scalar * (J.T @ J)
                f_agg += m_scalar * a_scalar * J.T.flatten()
            else:
                M_agg += J_leaf.T @ metric_val @ J_leaf
                # Curvature correction: subtract M @ crv term
                acc_minus_crv = accel_val - metric_val @ crv
                f_agg += J_leaf.T @ acc_minus_crv

        # Metric normalization (matches rmpgraph.py:143-148)
        # Scale M and f by same factor so the solve is invariant
        # but numerically better-conditioned
        M_max = float(np.max(np.abs(M_agg)))
        if M_max > 1.0:
            scale = M_max * 0.01
            M_agg = M_agg / scale
            f_agg = f_agg / scale

        # Regularise and solve
        M_reg = M_agg + 1e-3 * np.eye(n, dtype=np.float64)
        try:
            L = np.linalg.cholesky(M_reg)
            z = np.linalg.solve(L, f_agg)
            q_ddot = np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            q_ddot = np.linalg.lstsq(M_reg, f_agg, rcond=None)[0]

        # Safety clamp
        if not np.all(np.isfinite(q_ddot)):
            q_ddot = np.clip(q_ddot, -100.0, 100.0)
        else:
            # Soft clamp to avoid exploding commands
            q_ddot = np.clip(q_ddot, -50.0, 50.0)

        return q_ddot

    def integrate(self, q, qd, q_ddot):
        """Semi-implicit Euler from (q, qd, q_ddot) -> (q_new, qd_new)."""
        q_ddot = np.asarray(q_ddot, dtype=np.float64)
        qd_new = qd + self.dt * q_ddot
        q_new = q + self.dt * qd_new
        return q_new, qd_new

    def apply_hard_limits(self, q_new):
        """Hard-clip joint positions to limits (safety net)."""
        q_new = np.asarray(q_new, dtype=np.float64)
        if (self.joint_limits_low is not None
                and self.joint_limits_high is not None):
            return np.clip(q_new, self.joint_limits_low, self.joint_limits_high)
        return q_new
