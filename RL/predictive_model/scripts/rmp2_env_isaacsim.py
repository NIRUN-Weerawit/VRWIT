# rmp2_env_isaacsim.py
# =============================================================================
# RMP2-RL Training Environment: IsaacSim-based Gym wrapper
# =============================================================================
# Gym-style environment for training RMP2 gain policy with PPO
# Uses IsaacSim headless simulation with Franka Panda robot
#
# Key components:
#   - IsaacSim simulation setup (adapted from rdt_rmp_vr_panda.py)
#   - RMP2 solver with configurable gains (from rmp2_bridge.py)
#   - Reward computation (from spec)
#   - Observation/action spaces

import os
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# IsaacSim imports
import omni.isaac.kit
from omni.isaac.kit import SimulationApp


# ==============================================================================
# Configuration
# ==============================================================================

# USD Stage path
STAGE_PATH = "/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description/franka_obs_1.usd"

# Panda robot config
PANDA_JOINT_COUNT = 7
JOINT_LIMITS_LOW = np.array([
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
])
JOINT_LIMITS_HIGH = np.array([
    2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
])
DEFAULT_JOINT = np.array([0.0, 0.0, 0.0, -0.1, 0.0, 0.1, 0.86])

# Goal space (from spec)
GOAL_BOUNDS = {
    'x': (0.1, 0.6),
    'y': (-0.4, 0.4),
    'z': (0.15, 0.45)
}

# Episode config
MAX_STEPS = 1000  # 20 seconds at 0.02s/step
DT = 0.02

# Success criteria (from spec)
SUCCESS_POS_THRESHOLD = 0.1  # 10cm
SUCCESS_ROT_THRESHOLD = 10.0  # 10 degrees
COLLISION_THRESHOLD = 0.03  # 3cm

# Action gain bounds (from spec)
GAIN_BOUNDS = {
    'target_metric': (0.1, 5.0),
    'orientation_metric': (0.1, 3.0),
    'cspace_metric': (0.01, 0.5),
    'joint_limit_metric': (0.01, 1.0),
    'joint_limit_accel': (0.5, 5.0),
    'vel_cap_damping': (1.0, 20.0),
    'damping_accel': (1.0, 20.0),
    'obstacle_metric': (0.5, 10.0),
    'obstacle_repulsion': (10.0, 100.0),
}

# Reward weights (from spec)
REWARD_WEIGHTS = {
    'pos': 1.0,
    'orient': 0.3,
    'smooth': 0.01,
    'joint_limit': 0.5,
    'obstacle': 10.0,
    'success': 10.0,
    'collision': 50.0,
    'collision_dist': 0.5,
}


# ==============================================================================
# RMP2 Solver (inline implementation, adapted from rmp2_bridge.py)
# ==============================================================================

class TargetAttractorLeaf:
    """Target position attractor with configurable gains."""
    
    def __init__(self):
        self.name = "target"
    
    def eval(self, x, xd, gains=None):
        if gains is None:
            gains = {'metric_scalar': 1.0, 'accel_p_gain': 50.0, 'accel_d_gain': 10.0}
        
        metric_scalar = gains.get('metric_scalar', 1.0)
        accel_p_gain = gains.get('accel_p_gain', 50.0)
        accel_d_gain = gains.get('accel_d_gain', 10.0)
        
        goal = np.asarray(gains.get('goal', np.zeros(3)), dtype=np.float64)
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
        alpha = 0.9 * np.exp(-0.5 * scaled_dist ** 2) + 0.1
        I = np.eye(3, dtype=np.float64)
        S = np.outer(delta_hat, delta_hat)
        metric = alpha * metric_scalar * I + (1 - alpha) * 0.1 * S
        
        return metric, accel


class OrientationAttractorLeaf:
    """Orientation attractor with configurable gains."""
    
    def __init__(self):
        self.name = "orientation"
    
    @staticmethod
    def quat_mul(qa, qb):
        x1, y1, z1, w1 = qa
        x2, y2, z2, w2 = qb
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=np.float64)
    
    @staticmethod
    def quat_inv(q):
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)
    
    def quat_error(self, q_curr, q_goal):
        """Axis-angle representation of quaternion error (matches reference)."""
        q_rel = self.quat_mul(self.quat_inv(q_curr), q_goal)
        vec_norm = np.linalg.norm(q_rel[:3]) + 1e-8
        # Match reference: use arctan2(vec_norm, |w|) * 2 for the angle
        angle = 2.0 * np.arctan2(vec_norm, abs(q_rel[3]) + 1e-8)
        # Match reference: sign based on w component
        sign_w = 1.0 if q_rel[3] >= 0 else -1.0
        return sign_w * angle * q_rel[:3] / vec_norm
    
    def eval(self, x, xd, gains=None):
        if gains is None:
            gains = {'metric_scalar': 0.5, 'accel_p_gain': 100.0, 'accel_d_gain': 10.0}
        
        metric_scalar = gains.get('metric_scalar', 0.5)
        accel_p_gain = gains.get('accel_p_gain', 100.0)
        accel_d_gain = gains.get('accel_d_gain', 10.0)
        
        goal_quat = np.asarray(gains.get('goal_quat', np.array([0, 0, 0, 1.0])), dtype=np.float64)
        quat_curr = np.asarray(x, dtype=np.float64)
        
        delta = self.quat_error(quat_curr, goal_quat)
        delta_norm = np.linalg.norm(delta) + 1e-4
        omega = np.asarray(xd, dtype=np.float64)
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


class CSpaceTargetLeaf:
    """Configuration space target (default pose)."""
    
    def __init__(self, default_q=None):
        self.default_q = default_q if default_q is not None else DEFAULT_JOINT
    
    def eval(self, x, xd, gains=None):
        if gains is None:
            gains = {'metric_scalar': 0.05, 'position_gain': 1.0, 'damping_gain': 0.2}
        
        metric_scalar = gains.get('metric_scalar', 0.05)
        position_gain = gains.get('position_gain', 1.0)
        damping_gain = gains.get('damping_gain', 0.2)
        
        q = np.asarray(x, dtype=np.float64)
        xd = np.asarray(xd, dtype=np.float64)
        
        delta = q - self.default_q
        delta_norm = np.linalg.norm(delta)
        
        if delta_norm < 0.5:
            qdd_pos = -delta * position_gain
        else:
            qdd_pos = -0.5 * (delta / delta_norm) * position_gain
        qdd_vel = -damping_gain * xd
        accel = qdd_pos + qdd_vel
        
        n = len(x)
        metric = np.eye(n, dtype=np.float64) * (metric_scalar + 1e-4)
        
        return metric, accel


class JointLimitLeaf:
    """Joint limit avoidance."""
    
    def __init__(self, joint_limits_low, joint_limits_high):
        self.joint_limits_low = joint_limits_low
        self.joint_limits_high = joint_limits_high
    
    def eval(self, x, xd, gains=None):
        if gains is None:
            gains = {
                'metric_scalar': 0.1,
                'accel_potential_gain': 1.0,
                'accel_damper_gain': 200.0,
            }
        
        metric_scalar = gains.get('metric_scalar', 0.1)
        accel_potential_gain = gains.get('accel_potential_gain', 1.0)
        accel_damper_gain = gains.get('accel_damper_gain', 200.0)
        
        dist = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
        vel = np.asarray(xd, dtype=np.float64)
        
        # Metric: exponential barrier with velocity gate
        metric_before = metric_scalar / (dist / 0.01 + 0.001)
        sig = 1.0 / (1.0 + np.exp(-vel / 0.01))
        metric = (1 - sig) * metric_before
        
        # Acceleration
        scaled_x = dist / 0.1
        xdd_pos = accel_potential_gain / (scaled_x ** 2 + 0.01)
        xdd_vel = -accel_damper_gain * vel
        accel = xdd_pos + xdd_vel
        
        return np.diag(metric), accel


class JointVelocityCapLeaf:
    """Joint velocity capping."""
    
    def __init__(self, max_velocity=2.0, velocity_damping_region=0.15):
        self.max_velocity = max_velocity
        self.velocity_damping_region = velocity_damping_region
        self.damped_cutoff = max_velocity - velocity_damping_region
    
    def eval(self, x, xd, gains=None):
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


class JointDampingLeaf:
    """Joint damping."""
    
    def __init__(self):
        self.name = "damping"
    
    def eval(self, x, xd, gains=None):
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


class ObstacleAvoidanceLeaf:
    """Obstacle avoidance."""
    
    def __init__(self, margin=0.1):
        self.margin = margin
    
    def eval(self, d, xd_coll, gains=None):
        if gains is None:
            gains = {
                'metric_scalar': 3.5,
                'repulsion_gain': 40.0,
                'damping_gain': 100.0,
            }
        
        d = d.item()
        xd_coll = xd_coll.item()
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


class RMP2SolverLearnedGains:
    """RMP2 solver that accepts learned gain parameters."""
    
    def __init__(self, n_joints, joint_limits_low, joint_limits_high, default_q, dt=0.02):
        self.n_joints = n_joints
        self.joint_limits_low = joint_limits_low
        self.joint_limits_high = joint_limits_high
        self.default_q = default_q
        self.dt = dt
        
        # Initialize leaves
        self.target_leaf = TargetAttractorLeaf()
        self.orient_leaf = OrientationAttractorLeaf()
        self.cspace_leaf = CSpaceTargetLeaf(default_q)
        self.joint_limit_leaf = JointLimitLeaf(joint_limits_low, joint_limits_high)
        self.vel_cap_leaf = JointVelocityCapLeaf()
        self.damping_leaf = JointDampingLeaf()
        self.obstacle_leaf = ObstacleAvoidanceLeaf(margin=0.1)
    
    def forward_pass(self, q, qd, goal_pos, goal_quat, eef_pos, eef_quat, obstacle_dist, obs_vel):
        """Compute forward pass to get (state, velocity, Jacobian) for each leaf."""
        results = []
        
        # Target (position) - use first 3
        results.append((eef_pos, np.zeros(3), np.eye(3, self.n_joints)))
        
        # Orientation - use next 3
        results.append((eef_quat, np.zeros(3), np.eye(3, self.n_joints)))
        
        # CSpace target
        q_from_default = q - self.default_q
        results.append((q_from_default, qd, np.eye(self.n_joints)))
        
        # Joint limits
        dist_low = q - self.joint_limits_low
        dist_high = self.joint_limits_high - q
        dist_limit = np.minimum(dist_low, dist_high)
        results.append((dist_limit, qd, np.eye(self.n_joints)))
        
        # Velocity cap
        results.append((np.zeros(self.n_joints), qd, np.eye(self.n_joints)))
        
        # Damping
        results.append((np.zeros(self.n_joints), qd, np.eye(self.n_joints)))
        
        # Obstacle (1D scalar)
        results.append((np.array([obstacle_dist]), np.array([obs_vel]), np.eye(1, self.n_joints)))
        
        return results
    
    def solve(self, q, qd, gains, goal_pos, goal_quat, eef_pos, eef_quat, obstacle_dist, obs_vel):
        """Solve for joint acceleration using learned gains."""
        
        # Forward pass
        leaves_data = self.forward_pass(q, qd, goal_pos, goal_quat, eef_pos, eef_quat, obstacle_dist, obs_vel)
        
        # Evaluate each leaf with gains
        M_total = np.zeros((self.n_joints, self.n_joints), dtype=np.float64)
        b_total = np.zeros(self.n_joints, dtype=np.float64)
        
        # Leaf 0: Target position
        target_gains = {
            'metric_scalar': gains[0],
            'accel_p_gain': 50.0,
            'accel_d_gain': 10.0,
            'goal': goal_pos,
        }
        metric_0, accel_0 = self.target_leaf.eval(leaves_data[0][0], leaves_data[0][1], target_gains)
        J_0 = leaves_data[0][2]
        M_total += J_0.T @ metric_0 @ J_0
        b_total += J_0.T @ metric_0 @ accel_0
        
        # Leaf 1: Orientation
        orient_gains = {
            'metric_scalar': gains[1],
            'accel_p_gain': 100.0,
            'accel_d_gain': 10.0,
            'goal_quat': goal_quat,
        }
        metric_1, accel_1 = self.orient_leaf.eval(leaves_data[1][0], leaves_data[1][1], orient_gains)
        J_1 = leaves_data[1][2]
        M_total += J_1.T @ metric_1 @ J_1
        b_total += J_1.T @ metric_1 @ accel_1
        
        # Leaf 2: CSpace
        cspace_gains = {'metric_scalar': gains[2], 'position_gain': 1.0, 'damping_gain': 0.2}
        metric_2, accel_2 = self.cspace_leaf.eval(leaves_data[2][0], leaves_data[2][1], cspace_gains)
        J_2 = leaves_data[2][2]
        M_total += J_2.T @ metric_2 @ J_2
        b_total += J_2.T @ metric_2 @ accel_2
        
        # Leaf 3: Joint limit
        jl_gains = {
            'metric_scalar': gains[3],
            'accel_potential_gain': gains[4],
            'accel_damper_gain': 200.0,
        }
        metric_3, accel_3 = self.joint_limit_leaf.eval(leaves_data[3][0], leaves_data[3][1], jl_gains)
        J_3 = leaves_data[3][2]
        M_total += J_3.T @ metric_3 @ J_3
        b_total += J_3.T @ metric_3 @ accel_3
        
        # Leaf 4: Velocity cap
        vc_gains = {'damping_gain': gains[5], 'metric_weight': 1.0}
        metric_4, accel_4 = self.vel_cap_leaf.eval(leaves_data[4][0], leaves_data[4][1], vc_gains)
        J_4 = leaves_data[4][2]
        M_total += J_4.T @ metric_4 @ J_4
        b_total += J_4.T @ metric_4 @ accel_4
        
        # Leaf 5: Damping
        damp_gains = {'accel_d_gain': gains[6], 'metric_scalar': 0.005}
        metric_5, accel_5 = self.damping_leaf.eval(leaves_data[5][0], leaves_data[5][1], damp_gains)
        J_5 = leaves_data[5][2]
        M_total += J_5.T @ metric_5 @ J_5
        b_total += J_5.T @ metric_5 @ accel_5
        
        # Leaf 6: Obstacle
        obs_gains = {
            'metric_scalar': gains[7],
            'repulsion_gain': gains[8],
            'damping_gain': 100.0,
        }
        metric_6, accel_6 = self.obstacle_leaf.eval(leaves_data[6][0], leaves_data[6][1], obs_gains)
        J_6 = leaves_data[6][2]
        M_total += J_6.T @ metric_6 @ J_6
        b_total += J_6.T @ metric_6 @ accel_6
        
        # Solve: M * q_ddot = b
        # Add regularization for numerical stability
        M_reg = M_total + np.eye(self.n_joints) * 1e-6
        
        try:
            q_ddot = np.linalg.solve(M_reg, b_total)
        except np.linalg.LinAlgError:
            q_ddot = np.zeros(self.n_joints)
        
        return q_ddot
    
    def integrate(self, q, qd, q_ddot):
        """Euler integration step."""
        dt = self.dt
        q_new = q + qd * dt + 0.5 * q_ddot * dt ** 2
        qd_new = qd + q_ddot * dt
        
        # Clip to joint limits
        q_new = np.clip(q_new, self.joint_limits_low, self.joint_limits_high)
        qd_new = np.clip(qd_new, -2.5, 2.5)  # Velocity limits
        
        return q_new, qd_new


# ==============================================================================
# RMP2 Training Environment
# ==============================================================================

class RMP2TrainingEnv:
    """Gym-style environment for RMP2-RL training."""
    
    def __init__(self, headless=True, stage_path=None):
        self.headless = headless
        self.stage_path = stage_path or STAGE_PATH
        
        # Initialize RMP2 solver
        self.rmp2_solver = RMP2SolverLearnedGains(
            n_joints=PANDA_JOINT_COUNT,
            joint_limits_low=JOINT_LIMITS_LOW,
            joint_limits_high=JOINT_LIMITS_HIGH,
            default_q=DEFAULT_JOINT,
            dt=DT,
        )
        
        # State tracking
        self.goal_pos = None
        self.goal_quat = None
        self.obstacle_pos = None
        self.prev_obs_dist = None
        self.step_count = 0
        self.episode_count = 0
        
        # Pre-episode state
        self.q = None
        self.qd = None
        
        # Observation stats for normalization
        self.obs_mean = None
        self.obs_std = None
        self.obs_count = 0
        
        # Initialize IsaacSim
        self._init_sim()
        
        # Initialize robot and objects
        self._init_robot()
    
    def _init_sim(self):
        """Initialize IsaacSim simulation app."""
        self.sim_app = SimulationApp({"headless": self.headless})
        
        from isaacsim.core.api.simulation_context import SimulationContext
        from isaacsim.core.utils.stage import open_stage
        open_stage(self.stage_path)
        self.sim = SimulationContext()
        self.dt = self.sim.get_physics_dt()
        print(f"Simulation initialized with dt={self.dt}")
        self.sim.reset()
        print("Simulation reset in _init_sim")
        self.sim.play()
        print("Simulation started")
    
    def _init_robot(self):
        """Initialize robot, EEF, and obstacle."""
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
        from isaacsim.core.utils.prims import get_prim_at_path
        from pxr import UsdPhysics
        # Robot
        self.robot = SingleArticulation("/World/franka")
        self.robot.initialize()
        print("Robot initialized")
        # End effector
        self.panda_hand = SingleRigidPrim("/World/franka/panda_hand")
        self.panda_hand.initialize()
        
        # Obstacle
        self.obstacle = SingleXFormPrim("/World/Xform_obstacle1")
        self.obstacle.initialize()
        
        # Get obstacle position
        obs_pos, _ = self.obstacle.get_world_pose()
        assert obs_pos is not None, "Failed to get obstacle position"
        self.obstacle_pos = np.array(obs_pos, dtype=np.float64)
        
        # Set up joint drive parameters
        robot_prim = get_prim_at_path("/World/franka")
        stage = robot_prim.GetStage()
        for prim in stage.Traverse():
            if prim.GetPath().HasPrefix(robot_prim.GetPath()):
                if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    drive.GetStiffnessAttr().Set(1e4)
                    drive.GetDampingAttr().Set(1e2)
    
    def _sample_goal(self):
        """Sample a goal position that's not too close to the obstacle."""
        max_attempts = 50
        assert self.obstacle_pos is not None, "Obstacle position must be initialized before sampling goals"
        for _ in range(max_attempts):
            x = np.random.uniform(*GOAL_BOUNDS['x'])
            y = np.random.uniform(*GOAL_BOUNDS['y'])
            z = np.random.uniform(*GOAL_BOUNDS['z'])
            goal_pos = np.array([x, y, z], dtype=np.float64)
            
            # Check distance to obstacle
            dist_to_obs = np.linalg.norm(goal_pos - self.obstacle_pos)
            if dist_to_obs >= 0.15:  # Keep at least 15cm from obstacle
                # Sample orientation (random rotation)
                angle = np.random.uniform(-np.pi, np.pi)
                quat = R.from_euler('z', angle).as_quat()  # Returns (x, y, z, w)
                return goal_pos, quat
        
        # Fallback: return a safe default goal
        return np.array([0.3, 0.0, 0.3], dtype=np.float64), np.array([0, 0, 0, 1])
    
    def _get_obs(self):
        """Build observation vector."""
        # Joint state
        q = np.array(self.robot.get_joint_positions()[:PANDA_JOINT_COUNT], dtype=np.float64)
        qd = np.array(self.robot.get_joint_velocities()[:PANDA_JOINT_COUNT], dtype=np.float64)
        
        # EEF state
        eef_pos, eef_quat = self.panda_hand.get_world_pose()
        eef_pos = np.asarray(eef_pos, dtype=np.float64)
        eef_quat = np.asarray(eef_quat, dtype=np.float64)
        
        # Obstacle distance
        obs_dist = np.linalg.norm(eef_pos - self.obstacle_pos)
        
        # Obstacle velocity (rate of change)
        if self.prev_obs_dist is not None:
            obs_vel = (self.prev_obs_dist - obs_dist) / DT
        else:
            obs_vel = 0.0
        self.prev_obs_dist = obs_dist
        
        # Normalize orientation (make sure it's unit quaternion)
        eef_quat = eef_quat / (np.linalg.norm(eef_quat) + 1e-8)
        
        obs = np.concatenate([
            q, qd,
            self.goal_pos, self.goal_quat,
            [obs_dist], [obs_vel],
        ])
        
        return obs.astype(np.float32)
    
    def _compute_reward(self, q_ddot):
        """Compute total reward from all components."""
        # Get current joint positions
        q = self.q
        
        # Get EEF position
        eef_pos, eef_quat = self.panda_hand.get_world_pose()
        eef_pos = np.asarray(eef_pos, dtype=np.float64)
        eef_quat = np.asarray(eef_quat, dtype=np.float64)
        eef_quat = eef_quat / (np.linalg.norm(eef_quat) + 1e-8)
        
        reward = 0.0
        info = {}
        
        # 1. Position reward
        pos_error = np.linalg.norm(eef_pos - self.goal_pos)
        r_pos = -np.exp(-pos_error / 0.05)
        reward += REWARD_WEIGHTS['pos'] * r_pos
        info['r_pos'] = r_pos
        
        # 2. Orientation reward
        quat_dot = np.abs(np.dot(eef_quat, self.goal_quat))
        r_orient = REWARD_WEIGHTS['orient'] * quat_dot
        reward += r_orient
        info['r_orient'] = r_orient
        
        # 3. Smoothness reward
        q_ddot_mag = np.linalg.norm(q_ddot)
        r_smooth = -REWARD_WEIGHTS['smooth'] * q_ddot_mag
        reward += r_smooth
        info['r_smooth'] = r_smooth
        
        # 4. Joint limit proximity reward
        r_joint_limit = 0.0
        for i in range(PANDA_JOINT_COUNT):
            dist_low = q[i] - JOINT_LIMITS_LOW[i]
            dist_high = JOINT_LIMITS_HIGH[i] - q[i]
            dist_limit = min(dist_low, dist_high)
            if dist_limit < 0.3:
                r_joint_limit += -REWARD_WEIGHTS['joint_limit'] * max(0, 1.0 - dist_limit / 0.3)
        reward += r_joint_limit
        info['r_joint_limit'] = r_joint_limit
        
        # 5. Obstacle proximity reward
        obs_dist = np.linalg.norm(eef_pos - self.obstacle_pos)
        if obs_dist < REWARD_WEIGHTS['collision_dist']:
            r_obstacle = -REWARD_WEIGHTS['obstacle'] * max(0, REWARD_WEIGHTS['collision_dist'] - obs_dist) ** 2
            reward += r_obstacle
            info['r_obstacle'] = r_obstacle
        else:
            info['r_obstacle'] = 0.0
        
        # 6. Success bonus
        angle_error = 2 * np.arccos(np.clip(np.abs(np.dot(eef_quat, self.goal_quat)), 0, 1))
        angle_deg = np.degrees(angle_error)
        
        if pos_error < SUCCESS_POS_THRESHOLD and angle_deg < SUCCESS_ROT_THRESHOLD:
            reward += REWARD_WEIGHTS['success']
            info['success'] = True
        else:
            info['success'] = False
        
        # 7. Collision penalty
        if obs_dist < COLLISION_THRESHOLD:
            reward -= REWARD_WEIGHTS['collision']
            info['collision'] = True
        else:
            info['collision'] = False
        
        info['pos_error'] = pos_error
        info['angle_deg'] = angle_deg
        info['obs_dist'] = obs_dist
        
        return reward, info
    
    def _check_termination(self):
        """Check if episode should terminate."""
        eef_pos, eef_quat = self.panda_hand.get_world_pose()
        eef_pos = np.asarray(eef_pos, dtype=np.float64)
        eef_quat = np.asarray(eef_quat, dtype=np.float64)
        
        # Success
        pos_error = np.linalg.norm(eef_pos - self.goal_pos)
        angle_error = 2 * np.arccos(np.clip(np.abs(np.dot(eef_quat, self.goal_quat)), 0, 1))
        angle_deg = np.degrees(angle_error)
        
        if pos_error < SUCCESS_POS_THRESHOLD and angle_deg < SUCCESS_ROT_THRESHOLD:
            return True, 'success'
        
        # Collision
        obs_dist = np.linalg.norm(eef_pos - self.obstacle_pos)
        if obs_dist < COLLISION_THRESHOLD:
            return True, 'collision'
        
        # Step limit
        if self.step_count >= MAX_STEPS:
            return True, 'max_steps'
        
        return False, None
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset IsaacSim
        self.sim.reset()
        print("Simulation reset")
        self.sim.play()
        self._init_robot()
        # Set robot to default pose
        self.robot.set_joint_positions(
            DEFAULT_JOINT.tolist(),
            joint_indices=list(range(PANDA_JOINT_COUNT))
        )
        self.sim.step(render=True)
        
        # Sample new goal
        self.goal_pos, self.goal_quat = self._sample_goal()
        
        # Get initial state
        print(f"self.robot.get_joint_positions(): {self.robot.get_joint_positions()}")
        self.q = np.array(self.robot.get_joint_positions()[:PANDA_JOINT_COUNT], dtype=np.float64)
        self.qd = np.array(self.robot.get_joint_velocities()[:PANDA_JOINT_COUNT], dtype=np.float64)
        
        self.prev_obs_dist = None
        self.step_count = 0
        self.episode_count += 1
        
        obs = self._get_obs()
        
        return obs, {}
    
    def step(self, action):
        """Execute one step."""
        # Unpack and clamp action (gains)
        gains = self._clamp_gains(action)
        
        # Get EEF state
        eef_pos, eef_quat = self.panda_hand.get_world_pose()
        eef_pos = np.asarray(eef_pos, dtype=np.float64)
        eef_quat = np.asarray(eef_quat, dtype=np.float64)
        
        # Get obstacle distance
        obs_dist = float(np.linalg.norm(eef_pos - self.obstacle_pos))
        obs_vel = 0.0  # Simplified
        
        # Solve RMP2 with learned gains
        q_ddot = self.rmp2_solver.solve(
            self.q, self.qd, gains,
            self.goal_pos, self.goal_quat,
            eef_pos, eef_quat,
            obs_dist, obs_vel
        )
        
        # Integrate
        self.q, self.qd = self.rmp2_solver.integrate(self.q, self.qd, q_ddot)
        
        # Apply to simulation
        self.robot.set_joint_positions(
            self.q.tolist(),
            joint_indices=list(range(PANDA_JOINT_COUNT))
        )
        self.sim.step(render=True)
        
        # Compute reward
        reward, info = self._compute_reward(q_ddot)
        
        # Check termination
        terminated, term_reason = self._check_termination()
        truncated = self.step_count >= MAX_STEPS
        
        self.step_count += 1
        
        # Get observation
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, info
    
    def _clamp_gains(self, action):
        """Clamp action values to gain bounds."""
        action = np.asarray(action, dtype=np.float64).flatten()
        
        # Map [0-10] action indices to gains
        gains = np.zeros(11, dtype=np.float64)
        
        # Index 0: target metric (0.1, 5.0)
        gains[0] = np.clip(action[0], GAIN_BOUNDS['target_metric'][0], GAIN_BOUNDS['target_metric'][1])
        
        # Index 1: orientation metric (0.1, 3.0)
        gains[1] = np.clip(action[1], GAIN_BOUNDS['orientation_metric'][0], GAIN_BOUNDS['orientation_metric'][1])
        
        # Index 2: cspace metric (0.01, 0.5)
        gains[2] = np.clip(action[2], GAIN_BOUNDS['cspace_metric'][0], GAIN_BOUNDS['cspace_metric'][1])
        
        # Index 3: joint limit metric (0.01, 1.0)
        gains[3] = np.clip(action[3], GAIN_BOUNDS['joint_limit_metric'][0], GAIN_BOUNDS['joint_limit_metric'][1])
        
        # Index 4: joint limit accel (0.5, 5.0)
        gains[4] = np.clip(action[4], GAIN_BOUNDS['joint_limit_accel'][0], GAIN_BOUNDS['joint_limit_accel'][1])
        
        # Index 5: vel cap damping (1.0, 20.0)
        gains[5] = np.clip(action[5], GAIN_BOUNDS['vel_cap_damping'][0], GAIN_BOUNDS['vel_cap_damping'][1])
        
        # Index 6: damping accel (1.0, 20.0)
        gains[6] = np.clip(action[6], GAIN_BOUNDS['damping_accel'][0], GAIN_BOUNDS['damping_accel'][1])
        
        # Index 7: obstacle 0 metric (0.5, 10.0)
        gains[7] = np.clip(action[7], GAIN_BOUNDS['obstacle_metric'][0], GAIN_BOUNDS['obstacle_metric'][1])
        
        # Index 8: obstacle 0 repulsion (10.0, 100.0)
        gains[8] = np.clip(action[8], GAIN_BOUNDS['obstacle_repulsion'][0], GAIN_BOUNDS['obstacle_repulsion'][1])
        
        # Index 9-10: duplicate for obstacle 1 (same obstacle)
        gains[9] = gains[7]
        gains[10] = gains[8]
        
        return gains
    
    def close(self):
        """Clean up simulation."""
        if hasattr(self, 'sim_app'):
            self.sim_app.close()
    
    @property
    def observation_space(self):
        """Return observation space (for gym compatibility)."""
        return {
            'q': (PANDA_JOINT_COUNT,),
            'qd': (PANDA_JOINT_COUNT,),
            'goal_pos': (3,),
            'goal_quat': (4,),
            'obs_dist': (1,),
            'obs_vel': (1,),
        }
    
    @property
    def action_space(self):
        """Return action space dimensions."""
        return 11


# ==============================================================================
# Testing
# ==============================================================================

if __name__ == "__main__":
    print("Testing RMP2TrainingEnv...")
    
    env = RMP2TrainingEnv(headless=True)
    
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    print("\nTaking random action step...")
    action = np.random.uniform(0, 1, 11)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print(f"Info: {info}")
    
    print("\nClosing environment...")
    env.close()
    
    print("Test complete!")