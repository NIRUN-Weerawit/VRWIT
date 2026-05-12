# -*- coding: utf-8 -*-
"""
RMP2 Bridge with RL-Augmented Gains (RMP2-RL)

Pure NumPy implementation of RMP2 with learned gains via RL.
This is the "RMP2" part of Li et al., RSS 2021 - where the
gainscalars and length scales of each RMP leaf are predicted by
a small neural network instead of hand-tuned constants.

Architecture:
    State s = [q, qd, goal_pos, goal_quat, obstacles...]
    Policy pi(s) -> {gainscale_i}
    Each leaf eval(s, gainscale_i) -> (metric, acceleration)
    Backward pass: M*q_ddot = sum(J^T @ M @ a)

The geometric decomposition is preserved. RL only learns
*how much* each behavior matters given the current state.

Usage:
    solver = RMP2RLSolver(n_joints=6, fk_fn=fk, ...)
    solver.load_policy("policy_weights.npz")  # pretrained
    q_ddot = solver.solve_rl(q, qd, goals=[vrgoal], goal_quat=[quat],
                              obstacles=[obs], features={...})
    q_new, qd_new = solver.integrate(q, qd, q_ddot)

Training (outside this file):
    - Collect trajectories with RMP1 (fixed gains)
    - Define reward: task success + smoothness + joint limit penalty
    - Train MLPPolicy to predict gains that maximize reward
    - Policy outputs: gainscale per leaf (scalar or vector)
"""

import numpy as np
import math


# ==============================================================================
# Learned Gainscale Policy (MLP)
# ==============================================================================

class MLPPolicy:
    """Multi-Layer Perceptron predicting RMP leaf gainscales.

    Input: concatenated state features
        - joint positions (n)
        - joint velocities (n)
        - goal position (3)
        - goal quaternion (4)
        - distance to nearest obstacle (1)
        - velocity toward obstacle (1)
    Output: gainscale per leaf
        - target_pos: 1 (metric_scalar)
        - target_orient: 1 (metric_scalar)
        - cspace: 1 (metric_scalar)
        - joint_limit: 2 (metric_scalar, accel_potential_gain)
        - vel_cap: 1 (damping_gain)
        - damping: 1 (accel_d_gain)
        - obstacle_i: 2 (metric_scalar, repulsion_gain)
    Total output: 1 + 1 + 1 + 2 + 1 + 1 + 2*n_obs = 8 + 2*n_obs
    """

    def __init__(self, n_joints, n_obstacles=3, hidden_dims=[64, 32]):
        self.n_joints = n_joints
        self.n_obstacles = n_obstacles

        # Input dim: q(n) + qd(n) + goal_pos(3) + goal_quat(4) + obs_dist(1) + obs_vel(1)
        self.input_dim = n_joints * 2 + 3 + 4 + 1 + 1

        # Output dim per leaf type
        self.output_dim = 8 + 2 * n_obstacles

        # Simple MLP: input -> hidden -> hidden -> output
        # Using Xavier init, no bias on first layer (centered data)
        self.W1 = np.random.randn(self.input_dim, hidden_dims[0]) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros(hidden_dims[0])
        self.W2 = np.random.randn(hidden_dims[0], hidden_dims[1]) * np.sqrt(2.0 / hidden_dims[0])
        self.b2 = np.zeros(hidden_dims[1])
        self.W3 = np.random.randn(hidden_dims[1], self.output_dim) * np.sqrt(2.0 / hidden_dims[1])
        self.b3 = np.zeros(self.output_dim)

    def forward(self, state):
        """Predict gainscales given state vector.

        Args:
            state: flattened state vector (input_dim,)
        Returns:
            gains: raw output from MLP (output_dim,)
        """
        state = np.asarray(state, dtype=np.float64)
        # Hidden 1: ReLU
        h1 = np.maximum(0, state.dot(self.W1) + self.b1)
        # Hidden 2: ReLU
        h2 = np.maximum(0, h1.dot(self.W2) + self.b2)
        # Output: linear (use ReLU in leaf to enforce positivity)
        gains = h2.dot(self.W3) + self.b3
        return gains

    def predict_scaled(self, state, gain_bounds):
        """Predict gainscales and clamp to plausible ranges.

        Args:
            state: state vector
            gain_bounds: dict of (min, max) per leaf type
        Returns:
            dict of leaf_name -> gainscale(s)
        """
        raw = self.forward(state)

        # Unpack: [target, orient, cspace, jl_scalar, jl_accel, vel_cap,
        #           damping, obs0_scale, obs0_repulse, obs1_scale, ...]
        idx = 0
        result = {}

        result['target'] = {'metric_scalar': max(gain_bounds['target'][0],
                                                  min(gain_bounds['target'][1],
                                                      raw[idx]))}
        idx += 1

        result['orientation'] = {'metric_scalar': max(gain_bounds['orientation'][0],
                                                       min(gain_bounds['orientation'][1],
                                                           raw[idx]))}
        idx += 1

        result['cspace'] = {'metric_scalar': max(gain_bounds['cspace'][0],
                                                 min(gain_bounds['cspace'][1],
                                                     raw[idx]))}
        idx += 1

        result['joint_limit'] = {
            'metric_scalar': max(gain_bounds['joint_limit_metric'][0],
                                  min(gain_bounds['joint_limit_metric'][1],
                                      raw[idx])),
            'accel_potential_gain': max(gain_bounds['joint_limit_accel'][0],
                                        min(gain_bounds['joint_limit_accel'][1],
                                            raw[idx + 1])),
        }
        idx += 2

        result['vel_cap'] = {'damping_gain': max(gain_bounds['vel_cap'][0],
                                                 min(gain_bounds['vel_cap'][1],
                                                     raw[idx]))}
        idx += 1

        result['damping'] = {'accel_d_gain': max(gain_bounds['damping'][0],
                                                 min(gain_bounds['damping'][1],
                                                     raw[idx]))}
        idx += 1

        # Obstacles: each has (metric_scalar, repulsion_gain)
        for obs_idx in range(self.n_obstacles):
            result[f'obstacle_{obs_idx}'] = {
                'metric_scalar': max(gain_bounds['obstacle_metric'][0],
                                      min(gain_bounds['obstacle_metric'][1],
                                          raw[idx])),
                'repulsion_gain': max(gain_bounds['obstacle_repulsion'][0],
                                       min(gain_bounds['obstacle_repulsion'][1],
                                           raw[idx + 1])),
            }
            idx += 2

        return result

    def save(self, path):
        """Save policy weights to .npz file."""
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def load(self, path):
        """Load policy weights from .npz file."""
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']


# ==============================================================================
# RMP Leaf Classes (accept learned gains)
# ==============================================================================

class TargetAttractorRL:
    """TargetAttractor with learnable gainscales."""

    name = "target_rl"

    def __init__(self, gain_bounds=None):
        # Default bounds for clamping RL output
        self.gain_bounds = gain_bounds or {
            'metric_scalar': (0.1, 5.0),
            'accel_p_gain': (20.0, 150.0),
            'accel_d_gain': (5.0, 30.0),
        }

    def eval(self, x, xd, gains=None, **features):
        """Eval with optional learned gains.

        Args:
            x: current EEF position (3,)
            xd: velocity in task space (3,)
            gains: dict with 'metric_scalar', 'accel_p_gain', 'accel_d_gain'
                   If None, uses defaults.
        """
        # Default gains
        if gains is None:
            gains = {
                'metric_scalar': 1.0,
                'accel_p_gain': 50.0,
                'accel_d_gain': 10.0,
            }

        metric_scalar = gains.get('metric_scalar', 1.0)
        accel_p_gain = gains.get('accel_p_gain', 50.0)
        accel_d_gain = gains.get('accel_d_gain', 10.0)

        goal = np.asarray(features.get("goal", np.zeros(3)), dtype=np.float64)
        delta = goal - np.asarray(x, dtype=np.float64)
        delta_norm = np.linalg.norm(delta) + 1e-3
        delta_hat = delta / delta_norm

        # PD-like acceleration
        if delta_norm < 0.05:
            accel_p = accel_p_gain * delta
        else:
            accel_p = accel_p_gain * delta_hat
        accel = accel_p - accel_d_gain * np.asarray(xd, dtype=np.float64)

        # Metric: distance-dependent weighting
        scaled_dist = delta_norm / 0.05
        alpha = (0.9 * np.exp(-0.5 * scaled_dist ** 2) + 0.1)
        I = np.eye(3, dtype=np.float64)
        S = np.outer(delta_hat, delta_hat)
        metric = (alpha * metric_scalar * I
                  + (1 - alpha) * 0.1 * S)

        return metric, accel


class OrientationAttractorRL:
    """OrientationAttractor with learnable gains."""

    name = "orientation_rl"

    def __init__(self, gain_bounds=None):
        self.gain_bounds = gain_bounds or {
            'metric_scalar': (0.1, 3.0),
            'accel_p_gain': (50.0, 200.0),
        }

    @staticmethod
    def quat_mul(qa, qb):
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
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    def quat_error(self, q_curr, q_goal):
        q_rel = self.quat_mul(self.quat_inv(q_curr), q_goal)
        vec_norm = np.linalg.norm(q_rel[:3]) + 1e-4
        angle = 2.0 * np.arctan2(vec_norm, abs(q_rel[3]))
        sign_w = 1.0 if q_rel[3] >= 0 else -1.0
        return sign_w * angle * q_rel[:3] / vec_norm

    def eval(self, quat_curr, omega, gains=None, **features):
        if gains is None:
            gains = {'metric_scalar': 0.5, 'accel_p_gain': 100.0, 'accel_d_gain': 10.0}

        metric_scalar = gains.get('metric_scalar', 0.5)
        accel_p_gain = gains.get('accel_p_gain', 100.0)
        accel_d_gain = gains.get('accel_d_gain', 10.0)

        quat_curr = np.asarray(quat_curr, dtype=np.float64)
        quat_goal = np.asarray(features.get("goal_quat", np.array([0, 0, 0, 1.0])),
                               dtype=np.float64)

        delta = self.quat_error(quat_curr, quat_goal)
        delta_norm = np.linalg.norm(delta) + 1e-4

        omega = np.asarray(omega, dtype=np.float64)
        delta_hat = delta / delta_norm

        if delta_norm < 0.2:
            accel_p = accel_p_gain * delta
        else:
            accel_p = accel_p_gain * delta_hat
        accel = accel_p - accel_d_gain * omega

        # Metric with adaptive scalar
        scaled_dist = delta_norm / 0.2
        boost = np.exp(-0.5 * scaled_dist ** 2)
        metric_val = metric_scalar + boost
        metric = metric_val * np.eye(3, dtype=np.float64)

        return metric, accel


class CSpaceTargetRL:
    """CSpaceTarget with learnable metric_scalar."""

    name = "cspace_rl"

    def __init__(self, gain_bounds=None):
        self.gain_bounds = gain_bounds or {'metric_scalar': (0.01, 0.5)}

    def eval(self, x, xd, gains=None, **features):
        if gains is None:
            gains = {'metric_scalar': 0.05, 'position_gain': 1.0, 'damping_gain': 0.2}

        metric_scalar = gains.get('metric_scalar', 0.05)
        position_gain = gains.get('position_gain', 1.0)
        damping_gain = gains.get('damping_gain', 0.2)

        x = np.asarray(x, dtype=np.float64)
        xd = np.asarray(xd, dtype=np.float64)
        x_norm = np.linalg.norm(x)

        if x_norm < 0.5:
            qdd_pos = -x * position_gain
        else:
            qdd_pos = -0.5 * (x / x_norm) * position_gain
        qdd_vel = -damping_gain * xd
        accel = qdd_pos + qdd_vel
        metric = np.eye(len(x), dtype=np.float64) * (metric_scalar + 1e-4)

        return metric, accel


class JointLimitRL:
    """JointLimit with learnable gains."""

    name = "joint_limit_rl"

    def __init__(self, gain_bounds=None):
        self.gain_bounds = gain_bounds or {
            'metric_scalar': (0.01, 1.0),
            'accel_potential_gain': (0.5, 5.0),
        }

    def eval(self, dist, vel, gains=None, **features):
        if gains is None:
            gains = {
                'metric_scalar': 0.1,
                'accel_potential_gain': 1.0,
                'accel_damper_gain': 200.0,
                'metric_velocity_gate_length_scale': 0.01,
                'accel_potential_exploder_length_scale': 0.1,
            }

        metric_scalar = gains.get('metric_scalar', 0.1)
        accel_potential_gain = gains.get('accel_potential_gain', 1.0)
        accel_damper_gain = gains.get('accel_damper_gain', 200.0)
        metric_velocity_gate_length_scale = gains.get('metric_velocity_gate_length_scale', 0.01)
        accel_potential_exploder_length_scale = gains.get('accel_potential_exploder_length_scale', 0.1)

        dist = np.maximum(np.asarray(dist, dtype=np.float64), 0.0)
        vel = np.asarray(vel, dtype=np.float64)

        # Metric: exponential barrier with velocity gate
        metric_before = metric_scalar / (dist / 0.01 + 0.001)
        sig = 1.0 / (1.0 + np.exp(-vel / metric_velocity_gate_length_scale))
        metric = (1 - sig) * metric_before

        # Acceleration
        scaled_x = dist / accel_potential_exploder_length_scale
        xdd_pos = accel_potential_gain / (scaled_x ** 2 + 0.01)
        xdd_vel = -accel_damper_gain * vel
        accel = xdd_pos + xdd_vel

        return np.diag(metric), accel


class JointVelocityCapRL:
    """JointVelocityCap with learnable damping_gain."""

    name = "vel_cap_rl"

    def __init__(self, max_velocity=2.0, velocity_damping_region=0.15, gain_bounds=None):
        self.max_velocity = max_velocity
        self.velocity_damping_region = velocity_damping_region
        self.damped_cutoff = max_velocity - velocity_damping_region
        self.gain_bounds = gain_bounds or {'damping_gain': (1.0, 20.0)}

    def eval(self, x, xd, gains=None, **features):
        if gains is None:
            gains = {'damping_gain': 5.0, 'metric_weight': 1.0}

        damping_gain = gains.get('damping_gain', 5.0)
        metric_weight = gains.get('metric_weight', 1.0)

        xd = np.asarray(xd, dtype=np.float64)
        n = len(xd)
        delta_vel = np.abs(xd) - self.damped_cutoff

        metric = np.zeros(n, dtype=np.float64)
        accel = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if delta_vel[i] > 0:
                clipped = min(delta_vel[i], self.velocity_damping_region - 1e-6)
                ratio = clipped / self.velocity_damping_region
                metric[i] = metric_weight / (1.0 - ratio ** 2 + 1e-8)
                accel[i] = -damping_gain * delta_vel[i] * np.sign(xd[i])

        return np.diag(metric), accel


class JointDampingRL:
    """JointDamping with learnable accel_d_gain."""

    name = "damping_rl"

    def __init__(self, gain_bounds=None):
        self.gain_bounds = gain_bounds or {'accel_d_gain': (1.0, 20.0)}

    def eval(self, x, xd, gains=None, **features):
        if gains is None:
            gains = {'accel_d_gain': 5.0, 'metric_scalar': 0.005}

        accel_d_gain = gains.get('accel_d_gain', 5.0)
        metric_scalar = gains.get('metric_scalar', 0.005)

        xd = np.asarray(xd, dtype=np.float64)
        n = len(xd)
        xd_norm = np.linalg.norm(xd)

        accel = -accel_d_gain * xd_norm * xd
        nonlinear_scalar = metric_scalar * xd_norm
        metric = np.eye(n, dtype=np.float64) * (nonlinear_scalar + 1e-4)

        return metric, accel


class ObstacleAvoidanceRL:
    """ObstacleAvoidance with learnable gains."""

    name = "obstacle_rl"

    def __init__(self, margin=0.1, gain_bounds=None):
        self.margin = margin
        self.gain_bounds = gain_bounds or {
            'metric_scalar': (0.5, 10.0),
            'repulsion_gain': (10.0, 100.0),
        }

    def eval(self, d, xd_coll, gains=None, **features):
        if gains is None:
            gains = {
                'metric_scalar': 3.5,
                'repulsion_gain': 40.0,
                'damping_gain': 100.0,
            }

        d = float(d)
        xd_coll = float(xd_coll)
        d = max(d - self.margin, 0.0)

        if d > 0.5 or d <= 0:
            return (np.array([[0.0]], dtype=np.float64),
                    np.array([0.0], dtype=np.float64))

        metric_scalar = gains.get('metric_scalar', 3.5)
        repulsion_gain = gains.get('repulsion_gain', 40.0)
        damping_gain = gains.get('damping_gain', 100.0)

        gate = ((d / 0.5) ** 2 - 2 * d / 0.5 + 1)
        base_metric = metric_scalar / (d / 0.3 + 1e-5)
        metric_val = base_metric * gate

        xdd_repel = repulsion_gain * np.exp(-d / 0.3)

        sig = 1.0 / (1.0 + np.exp(-xd_coll / 0.1))
        z = d / 0.1 + 1e-5
        xdd_damping = -(1 - sig) * damping_gain * xd_coll / z

        accel_val = xdd_repel + xdd_damping

        if xd_coll > 0:
            metric_val *= (1 - sig)

        return (np.array([[metric_val]], dtype=np.float64),
                np.array([accel_val], dtype=np.float64))


# ==============================================================================
# RMP2-RL Solver
# ==============================================================================

class RMP2RLSolver:
    """RMP2 solver with RL-predicted gainscales.

    Same geometric core as RMP2Solver, but leaves receive
    learned gains from MLPPolicy instead of fixed constants.

    Usage:
        solver = RMP2RLSolver(n_joints=6, fk_fn=fk, ...)
        solver.policy.load("policy.npz")
        q_ddot = solver.solve_rl(q, qd, goals=[...], goal_quat=[...],
                                  obstacles=[...], features={...})
    """

    def __init__(self, n_joints, fk_fn, collision_fn=None,
                 joint_limits_low=None, joint_limits_high=None,
                 default_q=None, dt=0.02, n_obstacles=3):
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
        self.n_obstacles = n_obstacles

        # RL Policy
        self.policy = MLPPolicy(n_joints=n_joints, n_obstacles=n_obstacles)

        # Default gain bounds (for eval when policy not used)
        self.gain_bounds = {
            'target': (0.1, 5.0),
            'orientation': (0.1, 3.0),
            'cspace': (0.01, 0.5),
            'joint_limit_metric': (0.01, 1.0),
            'joint_limit_accel': (0.5, 5.0),
            'vel_cap': (1.0, 20.0),
            'damping': (1.0, 20.0),
            'obstacle_metric': (0.5, 10.0),
            'obstacle_repulsion': (10.0, 100.0),
        }

        # RL-augmented leaves
        self._fixed_leaves = [
            TargetAttractorRL(),
            OrientationAttractorRL(),
            CSpaceTargetRL(),
            JointLimitRL(),
            JointVelocityCapRL(),
            JointDampingRL(),
        ]
        self._leaf_names = ["target", "orientation", "cspace", "joint_limit",
                            "vel_cap", "damping"]

        # Curvature state
        self._prev_q = None
        self._prev_J = None
        self._fd_eps = 1e-4

    def _build_state(self, q, qd, goal_pos, goal_quat, obs_dist, obs_vel):
        """Concatenate features into policy input vector."""
        state = np.concatenate([
            np.asarray(q, dtype=np.float64).flatten(),
            np.asarray(qd, dtype=np.float64).flatten(),
            np.asarray(goal_pos, dtype=np.float64).flatten(),
            np.asarray(goal_quat, dtype=np.float64).flatten(),
            [obs_dist],
            [obs_vel],
        ])
        return state

    def _compute_curvatures(self, q, qd, leaves_data):
        """Compute Christoffel curvature terms (same as RMP2Solver)."""
        n = self.n_joints
        eps = self._fd_eps
        crvs = []

        for i, (x_leaf, xd_leaf, J_leaf) in enumerate(leaves_data):
            leaf_dim = J_leaf.shape[0]
            if i < len(self._leaf_names):
                name = self._leaf_names[i]
            else:
                name = f"obs_{i - len(self._leaf_names)}"

            if name in ("cspace", "joint_limit", "vel_cap", "damping"):
                crvs.append(np.zeros(leaf_dim, dtype=np.float64))
                continue

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

                if Jp.shape == (n, 6):
                    Jp = Jp.T
                    Jm = Jm.T

                if name == "target":
                    dJ = (Jp[:3, :] - Jm[:3, :]) / (2.0 * eps)
                elif name == "orientation":
                    dJ = (Jp[3:6, :] - Jm[3:6, :]) / (2.0 * eps)
                else:
                    crvs.append(np.zeros(leaf_dim, dtype=np.float64))
                    break

                crv += (dJ @ qd) * qd[k]
            else:
                crvs.append(crv)
                continue

            crvs.append(np.zeros(leaf_dim, dtype=np.float64))

        return crvs

    def forward_pass(self, q, qd, obstacles=None):
        """Forward pass returns (state, velocity, Jacobian) per leaf."""
        obstacles = obstacles or []
        results = []
        n = self.n_joints

        fk_out = self.fk_fn(q)
        eef_pos = np.asarray(fk_out[0], dtype=np.float64)
        eef_quat = np.asarray(fk_out[1], dtype=np.float64)
        J_spatial = np.asarray(fk_out[2], dtype=np.float64)

        # Target position
        J_pos = J_spatial[:3, :]
        xd_pos = J_pos @ qd
        results.append((eef_pos, xd_pos, J_pos))

        # Orientation
        J_rot = J_spatial[3:6, :]
        omega = J_rot @ qd
        results.append((eef_quat, omega, J_rot))

        # CSpace target
        q_from_default = q - self.default_q
        results.append((q_from_default, qd, np.eye(n, dtype=np.float64)))

        # Joint limits
        if self.joint_limits_low is not None and self.joint_limits_high is not None:
            dist_low = q - self.joint_limits_low
            dist_high = self.joint_limits_high - q
            dist_limit = np.minimum(dist_low, dist_high)
            results.append((dist_limit, qd, np.eye(n, dtype=np.float64)))
        else:
            results.append((np.ones(n, dtype=np.float64) * 10.0,
                          qd, np.eye(n, dtype=np.float64) * 0.0))

        # Velocity cap
        results.append((np.zeros(n, dtype=np.float64), qd,
                      np.eye(n, dtype=np.float64)))

        # Damping
        results.append((np.zeros(n, dtype=np.float64), qd,
                      np.eye(n, dtype=np.float64)))

        # Obstacles
        if self.collision_fn is not None and J_spatial is not None and len(obstacles) > 0:
            distances = self.collision_fn(q, eef_pos, obstacles)
            if distances is not None:
                for obs_prim, d in zip(obstacles, distances):
                    obs_pos, _ = obs_prim.get_world_pose()
                    diff = eef_pos - np.asarray(obs_pos, dtype=np.float64)
                    d_norm = float(np.linalg.norm(diff)) + 1e-8
                    n_hat = diff / d_norm
                    J_coll = (n_hat @ J_pos).reshape(1, n)
                    xd_coll = float(J_coll @ qd)
                    results.append((float(d), xd_coll, J_coll))

        return results

    def solve_rl(self, q, qd=None, goals=None, goal_quat=None,
                 obstacles=None, use_rl=True):
        """Solve with RL-predicted gains.

        Args:
            q: joint positions (n,)
            qd: joint velocities (n,)
            goals: list of goal positions (len 1 for now)
            goal_quat: goal orientation quaternion
            obstacles: list of obstacles
            use_rl: if True, use policy; if False, use defaults
        """
        q = np.asarray(q, dtype=np.float64)
        qd = (np.asarray(qd, dtype=np.float64) if qd is not None
              else np.zeros(self.n_joints, dtype=np.float64))

        obstacles = obstacles or []
        n = self.n_joints

        # Default feature extraction
        goal_pos = (np.asarray(goals[0], dtype=np.float64)
                    if goals else np.zeros(3, dtype=np.float64))
        goal_quat = (np.asarray(goal_quat, dtype=np.float64)
                     if goal_quat is not None else np.array([0, 0, 0, 1.0]))

        # Extract obstacle features for state
        obs_dist = 1.0  # default far
        obs_vel = 0.0
        if self.collision_fn is not None and obstacles:
            fk_out = self.fk_fn(q)
            eef_pos = np.asarray(fk_out[0], dtype=np.float64)
            distances = self.collision_fn(q, eef_pos, obstacles)
            if distances and any(d < 10.0 for d in distances):
                obs_dist = min([d for d in distances if d < 10.0])
                # Approximate obs velocity (crude)
                obs_vel = 0.0

        # Build state and predict gains if RL enabled
        state = self._build_state(q, qd, goal_pos, goal_quat, obs_dist, obs_vel)
        if use_rl:
            gains = self.policy.predict_scaled(state, self.gain_bounds)
        else:
            gains = None  # use defaults

        features = {
            "goal": goal_pos,
            "goal_quat": goal_quat,
            "obstacles": obstacles,
        }

        # Build leaf list
        n_obs = len(obstacles)
        obs_leaves = [ObstacleAvoidanceRL() for _ in range(n_obs)]
        all_leaves = self._fixed_leaves + obs_leaves
        leaf_names = self._leaf_names + [f"obstacle_{i}" for i in range(n_obs)]

        # Forward pass
        leaves_data = self.forward_pass(q, qd, obstacles)

        # Curvature (disabled by default, expensive)
        crvs = [np.zeros(J.shape[0], dtype=np.float64)
                for _, _, J in leaves_data]

        # Aggregate with learned gains
        M_agg = np.zeros((n, n), dtype=np.float64)
        f_agg = np.zeros(n, dtype=np.float64)

        # Map gains to leaves
        gains_map = {
            'target': gains.get('target', {}) if gains else {},
            'orientation': gains.get('orientation', {}) if gains else {},
            'cspace': gains.get('cspace', {}) if gains else {},
            'joint_limit': gains.get('joint_limit', {}) if gains else {},
            'vel_cap': gains.get('vel_cap', {}) if gains else {},
            'damping': gains.get('damping', {}) if gains else {},
        }
        for i in range(n_obs):
            gains_map[f'obstacle_{i}'] = (gains.get(f'obstacle_{i}', {})
                                          if gains else {})

        for leaf, name, (x_leaf, xd_leaf, J_leaf), crv in zip(
                all_leaves, leaf_names, leaves_data, crvs):

            leaf_gains = gains_map.get(name, None)

            # Handle RL leaves
            if hasattr(leaf, 'eval') and 'RL' in leaf.__class__.__name__:
                metric_val, accel_val = leaf.eval(x_leaf, xd_leaf,
                                                  gains=leaf_gains, **features)
            else:
                # Default fallback
                metric_val, accel_val = leaf.eval(x_leaf, xd_leaf, **features)

            if J_leaf.shape[0] == 1 and metric_val.shape == (1, 1):
                m_scalar = float(metric_val[0, 0])
                a_scalar = float(accel_val[0])
                J = J_leaf
                M_agg += m_scalar * (J.T @ J)
                f_agg += m_scalar * a_scalar * J.T.flatten()
            else:
                M_agg += J_leaf.T @ metric_val @ J_leaf
                acc_minus_crv = accel_val - metric_val @ crv
                f_agg += J_leaf.T @ acc_minus_crv

        # Metric normalization
        M_max = float(np.max(np.abs(M_agg)))
        if M_max > 1.0:
            scale = M_max * 0.01
            M_agg = M_agg / scale
            f_agg = f_agg / scale

        # Solve
        M_reg = M_agg + 1e-3 * np.eye(n, dtype=np.float64)
        try:
            L = np.linalg.cholesky(M_reg)
            z = np.linalg.solve(L, f_agg)
            q_ddot = np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            q_ddot = np.linalg.lstsq(M_reg, f_agg, rcond=None)[0]

        # Clamp
        if not np.all(np.isfinite(q_ddot)):
            q_ddot = np.clip(q_ddot, -100.0, 100.0)
        else:
            q_ddot = np.clip(q_ddot, -50.0, 50.0)

        return q_ddot

    def integrate(self, q, qd, q_ddot):
        """Semi-implicit Euler integration."""
        q_ddot = np.asarray(q_ddot, dtype=np.float64)
        qd_new = qd + self.dt * q_ddot
        q_new = q + self.dt * qd_new
        return q_new, qd_new

    def apply_hard_limits(self, q_new):
        """Hard clip to joint limits."""
        q_new = np.asarray(q_new, dtype=np.float64)
        if self.joint_limits_low is not None and self.joint_limits_high is not None:
            return np.clip(q_new, self.joint_limits_low, self.joint_limits_high)
        return q_new


# ==============================================================================
# Simple test
# ==============================================================================

if __name__ == "__main__":
    # Dummy FK for testing
    def dummy_fk(q):
        n = len(q)
        # Simple: EEF = q scaled
        pos = q[:3] * 0.1
        quat = np.array([0, 0, 0, 1.0])
        # Jacobian: identity truncated
        J = np.zeros((6, n), dtype=np.float64)
        J[:3, :3] = np.eye(3)
        return pos, quat, J

    def dummy_collision(q, eef_pos, obstacles):
        return [0.5]  # always 0.5m away

    solver = RMP2RLSolver(
        n_joints=3,
        fk_fn=dummy_fk,
        collision_fn=dummy_collision,
        joint_limits_low=np.array([-1.0, -1.0, -1.0]),
        joint_limits_high=np.array([1.0, 1.0, 1.0]),
        default_q=np.array([0.0, 0.0, 0.0]),
    )

    q = np.array([0.1, 0.2, 0.0])
    qd = np.zeros(3)
    goals = [np.array([0.3, 0.3, 0.1])]
    goal_quat = np.array([0, 0, 0, 1])

    # Test with RL (no obstacles)
    q_ddot_rl = solver.solve_rl(q, qd, goals=goals, goal_quat=goal_quat,
                                 obstacles=[], use_rl=True)
    print("RL q_ddot:", q_ddot_rl)

    # Test without RL (defaults, no obstacles)
    q_ddot_no_rl = solver.solve_rl(q, qd, goals=goals, goal_quat=goal_quat,
                                    obstacles=[], use_rl=False)
    print("Default q_ddot:", q_ddot_no_rl)

    # Test policy forward
    state = solver._build_state(q, qd, goals[0], goal_quat, 0.5, 0.0)
    raw = solver.policy.forward(state)
    print("Policy raw output shape:", raw.shape)

    print("\nRL-augmented RMP2 solver ready.")