# RMP2-RL Training System: Complete Specification

## 1. Project Overview

### Goal
Build a reinforcement learning system that learns optimal gain parameters for a Riemannian Motion Policy (RMP2) solver controlling a Franka Panda 7-DOF robotic arm in IsaacSim simulation.

### What is RMP2-RL?
RMP2 decomposes robot motion into multiple "behavior leaves":
- Target position attractor
- Target orientation attractor
- Configuration space target
- Joint limit avoidance
- Joint velocity capping
- Joint damping
- Obstacle avoidance

**Standard RMP2**: Each behavior leaf has hand-tuned gain parameters (scalars).

**RMP2-RL**: A neural network policy predicts the gain parameters for each behavior leaf given the current robot state. The geometric RMP2 structure is preserved; RL only learns *how much* each behavior should matter in different situations.

### Architecture
```
State (q, qd, goal, obstacles) → Policy Network → Gain Parameters
                                                         ↓
RMP2 Solver (geometric, unchanged) → Joint Acceleration (q_ddot)
                                                         ↓
IsaacSim Simulation → Next State + Reward
```

### Key Insight
The RMP2 solver itself is NOT trained. Only the gain prediction policy is learned via PPO. The solver provides a physically meaningful motion primitive that the policy modulates.

---

## 2. Simulation Environment

### Platform
- **IsaacSim** (NVIDIA Omniverse)
- **Headless mode**: `SimulationApp({"headless": True})`
- **Existing reference**: Study `/home/ucluser/VRWIT/RL/predictive_model/scripts/rdt_rmp_vr_panda.py` for IsaacSim setup pattern (imports, robot initialization, FK, collision detection)

### Robot: Franka Panda
| Parameter | Value |
|-----------|-------|
| Robot prim | `/World/franka` |
| Base link | `/World/franka/panda_link0` |
| EEF link | `panda_hand` |
| Arm joints | 7 (panda_joint1 through panda_joint7) |
| Gripper joints | 2 (panda_finger_joint1, panda_finger_joint2, indices 7-8) |
| Arm joint indices | 0-6 (only these are controlled by RMP2) |
| USD stage | `/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description/franka_obs_1.usd` |
| Lula IK config | `/home/ucluser/isaacgym/assets/urdf/piper_description/config/franka_robot.yaml` |
| Lula IK URDF | `/home/ucluser/isaacgym/assets/urdf/franka_description/robots/franka_panda.urdf` |

### Joint Limits (radians)
```
Joint  Low      High
 1:   -2.8973   2.8973
 2:   -1.7628   1.7628
 3:   -2.8973   2.8973
 4:   -3.0718  -0.0698
 5:   -2.8973   2.8973
 6:   -0.0175   3.7525
 7:   -2.8973   2.8973
```

### Home Pose
```
DEFAULT_JOINT = [0.0, 0.0, 0.0, -0.1, 0.0, 0.1, 0.86]  # 7 DOF
```

### Simulation Parameters
- **Physics timestep (dt)**: 0.02 seconds (20 Hz physics)
- **Control frequency**: Match physics timestep (20 Hz)

### Obstacle
- **Prim path**: `/World/Xform_obstacle1`
- **Type**: Static obstacle in front of the robot arm
- **Characteristics**: Approximately 30cm high, very thin (narrow profile)
- **Behavior**: Does NOT move between episodes (fixed position)
- **Purpose**: Teach the robot to navigate around obstacles while reaching goals

### IsaacSim Imports (from reference script)
```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.api.objects import VisualCuboid
from omni.isaac.motion_generation import ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula import LulaKinematicsSolver
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
```

### IsaacSim Setup Pattern (from reference)
```python
# Load stage
open_stage(STAGE_PATH)

# Initialize simulation
sim = SimulationContext()
dt = sim.get_physics_dt()  # 0.02
sim.reset()
sim.play()

# Initialize robot
robot = SingleArticulation("/World/franka")
robot.initialize()

# Initialize EEF
panda_hand = SingleRigidPrim("/World/franka/panda_hand")
panda_hand.initialize()

# Initialize obstacle
obstacle = SingleXFormPrim("/World/Xform_obstacle1")

# Initialize IK solver
ik_solver = LulaKinematicsSolver(robot_description_path=PANDA_YAML, urdf_path=PANDA_URDF)
ik_solver.set_default_position_tolerance(0.02)
ik_solver.set_default_orientation_tolerance(0.02)

kin_solver = ArticulationKinematicsSolver(
    robot_articulation=robot,
    kinematics_solver=ik_solver,
    end_effector_frame_name="panda_hand",
)

# Joint drive parameters (PID gains)
robot_prim = get_prim_at_path("/World/franka")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.GetStiffnessAttr().Set(1e4)
        drive.GetDampingAttr().Set(1e2)
```

### State Accessors (from IsaacSim)
- **Joint positions**: `robot.dof_positions` (returns array of 9 values, take first 7)
- **Joint velocities**: `robot.dof_velocities` (returns array, take first 7)
- **EEF position**: `panda_hand.get_world_pose()[0]` (x, y, z in world frame)
- **EEF orientation**: `panda_hand.get_world_pose()[1]` (x, y, z, w quaternion)
- **Jacobian**: `robot.calculate_jacobian("panda_hand", "world", q)` (6×9, take first 7 columns)
- **Obstacle position**: `obstacle.get_world_pose()[0]`

### FK Function
The reference script has a working `_panda_fk(q7)` function that computes:
1. EEF position (3D numpy array)
2. EEF quaternion (x, y, z, w)
3. Spatial Jacobian (6×7 matrix)

This function uses `robot.calculate_jacobian()` with a finite-difference fallback. **Reuse this pattern** in the new environment file.

### Collision Detection Function
The reference script has `_panda_collision(q, eef, obstacles)` that:
1. Gets obstacle world positions from `obs_prim.get_world_pose()`
2. Computes Euclidean distance from EEF to each obstacle
3. Returns list of distances

**Reuse this pattern** in the new environment file.

---

## 3. Environment Wrapper (Gym-style API)

### Class: `RMP2TrainingEnv`

Exposes a Gym-compatible interface for RL training. Internally uses IsaacSim simulation.

### Observation Space: 26 dimensions (continuous)
```
Index  Field              Dim  Description
0-6    q                  7    Joint positions (radians, all arm joints)
7-13   qd                 7    Joint velocities (rad/s, all arm joints)
14-16  goal_pos           3    Target end-effector position (x, y, z in meters)
17-20  goal_quat          4    Target orientation (x, y, z, w quaternion)
21     obs_dist           1    Minimum distance from EEF to any obstacle (meters)
22     obs_vel            1    Rate of change of obstacle distance (rad/s, negative = approaching)
```

### Action Space: 11 dimensions (continuous)
The policy outputs raw values that are clamped to predefined ranges:

```
Index  Leaf Type        Parameter              Min    Max
0      Target           metric_scalar          0.1    5.0
1      Orientation      metric_scalar          0.1    3.0
2      CSpace           metric_scalar          0.01   0.5
3      Joint Limit      metric_scalar          0.01   1.0
4      Joint Limit      accel_potential_gain   0.5    5.0
5      Velocity Cap     damping_gain           1.0    20.0
6      Damping          accel_d_gain           1.0    20.0
7-8    Obstacle 0       metric_scalar          0.5    10.0
                          repulsion_gain       10.0   100.0
9-10   Obstacle 1       metric_scalar          0.5    10.0
                          repulsion_gain       10.0   100.0
```

### reset()
1. Reset IsaacSim simulation: `sim.reset()`
2. Set robot to default joint positions: `DEFAULT_JOINT`
3. Sample a new random goal position from goal space
4. Sample a new random goal orientation (any reasonable orientation)
5. Return the initial observation

### step(action)
1. Clamp action values to gain bounds
2. Unpack actions into gain dictionaries per leaf
3. Run RMP2 solver with these gains:
   - Get current state (q, qd)
   - Forward pass through RMP2 leaves with the given gains
   - Backward pass (transform to joint space via Jacobian)
   - Aggregate to get optimal q_ddot
4. Integrate q_ddot to get new q and qd (Euler integration):
   ```
   q_new = q + qd * dt + 0.5 * q_ddot * dt^2
   qd_new = qd + q_ddot * dt
   q_new = clip(q_new, joint_limits_low, joint_limits_high)
   ```
5. Apply joint positions to IsaacSim: `robot.set_joint_positions(q_new, joint_indices=list(range(7)))`
6. Step simulation: `sim.step(render=False)`
7. Compute reward
8. Check termination conditions
9. Return (observation, reward, terminated, truncated, info)

### close()
Shut down IsaacSim simulation app cleanly.

### Episode Termination
| Condition | Type | Details |
|-----------|-|---|
| Position + orientation success | `terminated=True` | \|\|eef_pos - goal_pos\|\| < 0.1m AND quaternion angle < 10° |
| Collision (too close to obstacle) | `terminated=True` | EEF-obstacle distance < 0.03m |
| Step limit | `truncated=True` | Maximum 1000 steps (20 seconds at 0.02s/step) |
| Joint limit violation | `terminated=True` | Any joint position exceeds joint limits (after clipping, if q_new is still at limit for multiple steps) |

### Goal Sampling (Goal Space)
```python
import numpy as np

x = np.random.uniform(0.1, 0.6)   # 10-60cm forward
y = np.random.uniform(-0.4, 0.4)  # ±40cm lateral
z = np.random.uniform(0.15, 0.45) # 15-45cm height

# Orientation: random reasonable orientation
# Sample a random rotation around z-axis (for tool approach direction)
angle_z = np.random.uniform(-np.pi, np.pi)
# Or use identity quaternion for simplicity initially
```

**IMPORTANT**: Goals should NOT be sampled inside or too close to the obstacle. Since the obstacle is in front of the arm at ~30cm height, consider checking distance from obstacle when sampling:
```python
obs_position = obstacle.get_world_pose()[0]
goal_distance_to_obs = np.linalg.norm(goal_pos - obs_position)
while goal_distance_to_obs < 0.15:  # Keep at least 15cm from obstacle
    goal_pos = sample_new_goal()
    goal_distance_to_obs = np.linalg.norm(goal_pos - obs_position)
```

---

## 4. Reward Function

### Components

Each component is computed at every step. Total reward is the sum.

#### 1. Position Reward: `r_pos`
Pull the EEF toward the goal position.
```python
pos_error = np.linalg.norm(eef_pos - goal_pos)
r_pos = -np.exp(-pos_error / 0.05)  # Sigmoid-like, saturates at -1 when far away
# When pos_error → 0, r_pos → -1 (negative penalty, minimal)
# When pos_error → ∞, r_pos → 0 (no reward)
# Equivalent: more negative = worse, closer to 0 = better
```

#### 2. Orientation Reward: `r_orient`
Encourage matching the goal orientation.
```python
# Quaternion cosine similarity (scaled to [-1, 1])
quat_dot = np.abs(np.dot(eef_quat, goal_quat))
r_orient = 0.3 * quat_dot  # Scale by 0.3
# Maximum: 0.3, Minimum: 0
```

#### 3. Smoothness Reward: `r_smooth`
Penalize jerky/high-acceleration motions.
```python
q_ddot_magnitude = np.linalg.norm(q_ddot)
r_smooth = -0.01 * q_ddot_magnitude  # Small penalty
# More negative = more jerky
```

#### 4. Joint Limit Proximity Reward: `r_joint_limit`
Encourage staying away from joint limits.
```python
dist_low = q - joint_limits_low  # Distance to lower limit (7,)
dist_high = joint_limits_high - q  # Distance to upper limit (7,)
dist_to_limit = np.minimum(dist_low, dist_high)  # (7,)

# For each joint, compute proximity penalty
for i in range(7):
    if dist_to_limit[i] < 0.3:  # Within 0.3 radians of a limit
        r_joint_limit += -0.5 * max(0, 1.0 - dist_to_limit[i] / 0.3)
    # When at limit: r = -0.5, at 0.3 rad away: r = 0
```

#### 5. Obstacle Proximity Reward: `r_obstacle`
Strongly penalize approaching the obstacle.
```python
obs_distance = np.linalg.norm(eef_pos - obstacle_pos)
if obs_distance < 0.5:  # Within 50cm of obstacle
    r_obstacle = -10.0 * max(0, 0.5 - obs_distance) ** 2
    # Quadratic penalty: stronger when closer
    # At 0.5m distance: r = 0
    # At 0.0m distance: r = -10.0 * 0.25 = -2.5
else:
    r_obstacle = 0
```

#### 6. Success Bonus: `r_success`
Large positive reward when reaching the goal.
```python
pos_error = np.linalg.norm(eef_pos - goal_pos)
# Orientation angle in degrees
angle_deg = quaternion_angle_degrees(eef_quat, goal_quat)

if pos_error < 0.1 and angle_deg < 10:
    r_success = 10.0
else:
    r_success = 0
```

#### 7. Collision Penalty: `r_collision`
Very large penalty for actual collision.
```python
obs_distance = np.linalg.norm(eef_pos - obstacle_pos)
if obs_distance < 0.03:  # Less than 3cm = collision
    r_collision = -50.0
else:
    r_collision = 0
```

### Total Reward
```python
total_reward = r_pos + r_orient + r_smooth + r_joint_limit + r_obstacle + r_success + r_collision
```

### Reward Summary Table
| Component | Range | Purpose |
|---|---|---|
| `r_pos` | [-1, 0] | Attract toward goal position |
| `r_orient` | [0, 0.3] | Match goal orientation |
| `r_smooth` | (-∞, 0] | Penalize jerky motion |
| `r_joint_limit` | [-0.5 × 7, 0] | Stay within joint limits |
| `r_obstacle` | [-2.5, 0] | Avoid obstacle vicinity |
| `r_success` | [0, 10] | Bonus for reaching goal |
| `r_collision` | [-50, 0] | Heavy penalty for collision |
| **Total** | **[-52+, +10.3]** | Combined signal |

### Note on Reward Tuning
These weights are starting points. During training, monitor reward shapes and adjust:
- If robot doesn't reach goals: increase `r_pos` magnitude or `r_success`
- If robot hits obstacles: increase `r_obstacle` magnitude
- If motion is too jerky: increase `r_smooth` weight
- If robot avoids joint limits too conservatively: decrease `r_joint_limit` weight

---

## 5. Policy Architecture (PyTorch)

### Network: `MLPPolicyNet`

This is the PyTorch equivalent of the existing NumPy `MLPPolicy` in `rmp2_bridge_rl.py`. Same input/output dimensions, same architecture.

#### Input: 26-dimensional state vector (float32 tensor)
Same as the observation space defined in Section 3.

#### Architecture
```python
class MLPPolicyNet(nn.Module):
    def __init__(self, input_dim=26, hidden_dims=[64, 32], output_dim=11):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.mean = nn.Linear(hidden_dims[1], output_dim)  # Policy mean
        self.log_std = nn.Parameter(torch.zeros(output_dim))  # Learnable log std dev for exploration
        
        # Initialize weights (Xavier/Glorot)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.mean.weight)
        
    def forward(self, x):
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        mean = self.mean(h2)
        std = torch.exp(self.log_std)
        return mean, std  # For sampling: action = mean + std * epsilon
        
    def get_action(self, x):
        mean, std = self.forward(x)
        dist = Normal(mean, std)
        action = dist.sample()
        return action, dist.entropy().mean(), dist.log_prob(action)
```

#### Output: 11-dimensional action vector (raw, continuous)
- Clamped to gain bounds AFTER network output, BEFORE passing to RMP2 solver
- For training: use softplus or tanh to constrain outputs to valid ranges during the forward pass

#### Value Network (Critic)
Separate value head for PPO's advantage estimation:
```python
class CriticNet(nn.Module):
    def __init__(self, input_dim=26, hidden_dims=[64, 32]):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.value = nn.Linear(hidden_dims[1], 1)  # Single value output
        
    def forward(self, x):
        return self.value(self.relu2(self.fc2(self.relu1(self.fc1(x)))))
```

---

## 6. PPO Algorithm

### Configuration
| Parameter | Value | Notes |
|---|---|---|
| `gamma` (discount) | 0.99 | Standard for episodic tasks |
| `gae_lambda` (advantage) | 0.95 | Generalized Advantage Estimation |
| `clip_epsilon` | 0.2 | PPO clipping parameter |
| `ent_coef` (entropy) | 0.01 | Exploration bonus |
| `vf_coef` (value loss) | 0.5 | Value function loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `batch_size` | 64 | Mini-batch gradient updates |
| `n_epochs` | 8 | Optimization epochs per rollout |
| `rollout_steps` | 2048 | Steps collected per epoch |
| `lr` (learning rate) | 3e-4 | Initial Adam learning rate |
| `lr_decay` | Linear to 0 | Over total training steps |
| `total_steps` | 1,000,000 | ~202 PPO epochs |
| `batch_size` | 128 | For data collection (128 steps × 1 env × 16 = 2048) |

### PPO Training Loop (Pseudocode)
```
Initialize policy_net and value_net
optimizer = Adam(policy.parameters() + value.parameters(), lr=3e-4)

For epoch in range(total_epochs):
    # -- Data Collection --
    For step in range(rollout_steps):
        obs = env.reset() if first_or_done else obs
        
        # Normalize observation
        obs_normalized = normalize(obs)
        
        # Get action from policy
        action, log_prob, entropy = policy(obs_normalized)
        action = clamp_action(action)  # To gain bounds
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store transition
        memory.append((obs, action, reward, terminated, log_prob, entropy))
        
        obs = next_obs
    
    # -- PPO Update --
    # Compute advantages using GAE
    advantages = compute_gae(rewards, values, terminated, gamma=0.99, lambda=0.95)
    
    For update_epoch in range(n_epochs=8):
        For mini_batch in shuffle_and_split(memory, batch_size=64):
            
            # Forward pass
            action_probs, values = policy_and_value(batch_obs)
            new_log_probs = get_log_probs(action_probs, batch_actions)
            
            # PPO clipped loss
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            advantages_clipped = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
            
            loss_actor = -torch.min(ratio * advantages, advantages_clipped * advantages).mean()
            loss_value = 0.5 * torch.nn.functional.mse_loss(values, batch_returns)
            loss_entropy = -0.01 * entropy.mean()
            
            total_loss = loss_actor + 0.5 * loss_value + loss_entropy
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
    
    # -- Logging (WandB) --
    log_metrics({
        'episode_returns': list_of_returns,
        'episode_lengths': list_of_lengths,
        'success_rate': success_count / total_episodes,
        'policy_loss': loss_actor.item(),
        'value_loss': loss_value.item(),
        'entropy_loss': loss_entropy.item(),
        'learning_rate': optimizer.param_groups[0]['lr'],
        'mean_reward_per_step': mean_reward,
    })
```

### Key Implementation Notes
1. **Observation normalization**: Use running mean/std for online normalization
2. **Advantage normalization**: Normalize advantages to zero-mean, unit-variance within each batch
3. **Gradient clipping**: Clip gradients by global norm (max_norm=0.5)
4. **Learning rate decay**: Linear decay from 3e-4 to 0 over total training steps
5. **Entropy bonus**: Encourages exploration, especially early in training

### Generalized Advantage Estimation (GAE)
```python
def compute_gae(rewards, values, terminated, gamma=0.99, gae_lambda=0.95):
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0 if terminated[t] else values[t + 1]
        else:
            next_value = 0 if terminated[t] else values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - terminated[t]) - values[t]
        gae = delta + gamma * gae_lambda * gae * (1 - terminated[t])
        advantages.insert(0, gae)
    
    returns = adv + values  # Actual returns
    return torch.tensor(advantages), torch.tensor(returns)
```

---

## 7. Training Script (`train_rmp2_rl.py`)

### Command-Line Arguments
```
--total_steps       Total training steps (default: 1000000)
--rollout_steps     Steps per rollout (default: 2048)
--n_epochs          PPO epochs per rollout (default: 8)
--lr                Learning rate (default: 3e-4)
--batch_size        Mini-batch size (default: 64)
--ent_coef          Entropy coefficient (default: 0.01)
--vf_coef           Value loss coefficient (default: 0.5)
--clip_epsilon      PPO clip parameter (default: 0.2)
--gamma             Discount factor (default: 0.99)
--gae_lambda        GAE lambda (default: 0.95)
--max_grad_norm     Max gradient norm (default: 0.5)
--wandb_project     WandB project name (default: "rmp2-rl")
--wandb_run_name    WandB run name (default: auto-generated)
--stage_path        Path to USD stage
--save_freq         Epochs between model saves (default: 50)
--save_dir          Directory to save models (default: "./trained_models")
--eval_freq         Epochs between evaluation runs (default: 10)
--seed              Random seed for reproducibility (default: 42)
```

### WandB Logging
Project name: `rmp2-rl`

Log at every PPO epoch:
| Metric | Description |
|---|---|
| `episode_return` | Return per completed episode |
| `episode_length` | Steps per completed episode |
| `success_rate` | % of successful episodes in last batch |
| `policy_loss` | Actor loss value |
| `value_loss` | Critic loss value |
| `entropy_loss` | Entropy bonus value |
| `learning_rate` | Current learning rate |
| `mean_reward_per_step` | Average reward per environment step |
| `advantage_mean` | Mean advantage (for debugging) |
| `advantage_std` | Std of advantages (for debugging) |
| `episode_success_count` | Running count of successful episodes |
| `episode_collision_count` | Running count of collision episodes |

### Checkpointing
Save model weights every `save_freq` epochs:
```python
torch.save({
    'policy_state_dict': policy_net.state_dict(),
    'value_state_dict': value_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'training_step': total_steps_so_far,
}, os.path.join(save_dir, f"policy_epoch_{epoch}.pt"))
```

### Evaluation
Every `eval_freq` epochs, run 10 evaluation episodes:
- Fixed random goals (for consistency)
- No exploration (use deterministic actions: mean, not sample)
- Report success rate, mean return, mean episode length
- Log to WandB with prefix `eval/`

---

## 8. File Structure

```
/home/ucluser/VRWIT/RL/predictive_model/
├── scripts/
│   ├── rmp2_env_isaacsim.py           # IsaacSim training environment
│   ├── rmp2_rl_policy.py              # PyTorch policy + PPO trainer
│   └── train_rmp2_rl.py               # Training loop + logging
│
│   # DO NOT EDIT:
│   ├── rdt_rmp_vr_panda.py            # Reference IsaacSim setup (DO NOT MODIFY)
│   ├── rmp2_bridge.py                  # Base RMP2 solver (read-only reference)
│   └── rmp2_bridge_rl.py              # RL-augmented RMP2 (read-only reference)
```

### File: `rmp2_env_isaacsim.py`
- Class: `RMP2TrainingEnv`
- Methods: `reset()`, `step(action)`, `close()`, `render()`
- Contains:
  - IsaacSim initialization (headless)
  - Robot setup (from reference script)
  - FK function (adapted from `_panda_fk`)
  - Collision detection (adapted from `_panda_collision`)
  - Reward computation (all 7 components)
  - Goal sampling with obstacle avoidance
  - State building for policy input
  - Observation normalization (running mean/std)

### File: `rmp2_rl_policy.py`
- Class: `MLPPolicyNet` (actor)
- Class: `CriticNet` (value)
- Class: `PPOAgent`
  - Methods: `select_action(obs)`, `update(batch)`, `save(path)`, `load(path)`
  - Contains: GAE computation, PPO loss computation, Adam optimizer, LR scheduler

### File: `train_rmp2_rl.py`
- Main training loop
- WandB integration
- Checkpointing
- Evaluation routine
- Command-line argument parsing

---

## 9. RMP2 Solver Integration

### How to Use the RMP2 Solver with Learned Gains

The RMP2 solver (`RMP2Solver` or `RMP2RLSolver`) computes optimal joint acceleration `q_ddot` given:
1. Current joint state: `q`, `qd`
2. Goal: `goal_pos`, `goal_quat`
3. Obstacles: `obstacles` list
4. Gains: Learned parameters from the policy

#### Flow:
```python
# 1. Get observation from environment
obs = env.reset()  # (26,)

# 2. Policy predicts gain parameters
gains = policy.predict_gains(obs)  # Raw predictions → clamped to bounds → dict per leaf

# 3. Environment runs RMP2 solver with these gains
action = gains  # The action IS the gains
next_obs, reward, done, info = env.step(action)

# Inside env.step():
# - Unpack gains into per-leaf dictionaries
# - Forward pass through RMP2 leaves: each leaf.eval(x, xd, gains=leaf_gains) → (metric, accel)
# - Backward pass: Transform to joint space via Jacobian: M*q_ddot = sum(J^T @ M @ a)
# - Integrate: q_new = q + qd*dt + 0.5*q_ddot*dt^2
# - Apply to IsaacSim: robot.set_joint_positions(q_new)
# - Step simulation: sim.step(render=False)
# - Compute reward
# - Return next observation
```

### Key Integration Point
The RMP2 solver is NOT a neural network -- it's a geometric controller. The RL policy outputs gain parameters that weight how much each behavior leaf contributes to the final motion command. The solver handles all the geometry, Jacobians, and metric computations.

### Solver Choice
Use the RMP2 solver from `rmp2_bridge.py` (base version) and pass the learned gains directly to each leaf's `eval()` method. The RL-augmented leaf classes in `rmp2_bridge_rl.py` show the pattern -- each leaf accepts an optional `gains` dict parameter that overrides fixed values.

**Implementation note**: Instead of importing from `rmp2_bridge_rl.py`, the new environment should contain its own RMP2 solver logic that accepts gain parameters. This keeps the training environment self-contained and avoids import dependencies on the existing scripts.

---

## 10. Training Workflow

### Phase 1: Curriculum Learning (Recommended)
1. **Start simple**: No obstacles, just point-to-point reaching
2. **Add obstacles**: After basic reaching works, enable obstacle avoidance
3. **Add orientation**: Start with position-only, add orientation matching later
4. **Full task**: Combined position + orientation + obstacle avoidance

### Phase 2: Hyperparameter Tuning
Monitor these metrics during training:
- Learning curve (episode return over time)
- Success rate over time
- Collision rate over time
- Reward signal per component (decompose rewards for debugging)
- Policy entropy (should decrease as training progresses)

### Phase 3: Evaluation & Deployment
- Test on novel goal positions
- Test with different obstacle positions
- Evaluate motion smoothness (jerk metrics)
- Export policy weights for deployment

---

## 11. Important Notes

### From Existing Reference Scripts
1. **RMP2 solver**: Study `scripts/rmp2_bridge.py` for the complete RMP2 implementation
2. **RL-augmented RMP2**: Study `scripts/rmp2_bridge_rl.py` for the gain-parameterized leaf pattern
3. **IsaacSim setup**: Study `scripts/rdt_rmp_vr_panda.py` for the IsaacSim initialization pattern
4. **FK function**: Adapt `_panda_fk()` from the reference script
5. **Collision detection**: Adapt `_panda_collision()` from the reference script

### Constraints
- **DO NOT modify** any existing scripts in the `scripts/` directory
- **DO NOT modify** `scripts/rdt_rmp_vr_panda.py`, `scripts/rmp2_bridge.py`, or `scripts/rmp2_bridge_rl.py`
- Use the existing scripts as **references only**
- All new code goes into **new files** in the same directory
- Must be **headless IsaacSim** compatible
- Must work with **PyTorch tensors** for the policy
- Use **WandB** for logging

### Common Issues to Watch For
1. **NaN gradients**: Can occur with extreme rewards. Use gradient clipping and reward clipping.
2. **Simulation stability**: Euler integration can be unstable. Use small dt (0.02s).
3. **Quaternion normalization**: Quaternions must be unit quaternions. Normalize before computing errors.
4. **Jacobian computation**: Can fail in singular configurations. Use finite-difference fallback.
5. **Obstacle detection**: Must handle edge cases where obstacle is far away.
6. **Memory leaks**: IsaacSim can accumulate memory. Reset simulation between epochs or use `sim.reset()`.

### Expected Performance
- **Initial phase**: Random gains, poor performance, many collisions
- **100K steps**: Should learn basic reaching
- **500K steps**: Should learn obstacle avoidance
- **1M steps**: Should achieve good success rate with smooth motions

---

## 12. Dependencies

| Package | Purpose |
|---|---|
| `torch` | PyTorch for policy network |
| `numpy` | Numerical computations |
| `gymnasium` | Environment interface |
| `wandb` | Experiment logging |
| `isaacsim` | Simulation (IsaacSim) |
| `omni.isaac.kit` | IsaacSim simulation app |
| `isaacsim.core.api` | IsaacSim core APIs |
| `omni.isaac.motion_generation` | IK solver |
| `paho-mqtt` | MQTT (used in reference, may not need for training) |
| `pyquaternion` or `scipy.spatial.transform` | Quaternion operations |

---

## 13. Final Checklist

Before starting training, verify:
- [ ] IsaacSim headless mode works correctly
- [ ] Robot can be initialized and moved
- [ ] EEF position/orientation can be read
- [ ] Obstacle position can be detected
- [ ] FK function returns correct Jacobian
- [ ] Collision detection works correctly
- [ ] Goal sampling avoids obstacles
- [ ] Reward function computes all components
- [ ] Policy network loads correctly
- [ ] PPO agent trains without NaN
- [ ] WandB logging captures all metrics
- [ ] Checkpointing saves/loads correctly
- [ ] Evaluation runs independently
