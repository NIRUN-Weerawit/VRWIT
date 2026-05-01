# rdt_sim_vr.py
# =============================================================================
# PiPER Robot VR Teleoperation + Predictive Model Integration (Isaac Sim)
# =============================================================================
#
# Overview
# --------
# This script runs an Isaac Sim simulation of the PiPER dual-arm robot and
# enables real-time VR-based teleoperation via an MQTT broker.  It combines
# several subsystems:
#
#   1. MQTT VR Bridge
#      • Subscribes to an MQTT topic ("robot/piper-wee") on a remote broker
#        to receive the VR controller's goal position, orientation, grip
#        state, and button/trigger inputs.
#      • Global variables (vr_goal_pos, vr_goal_rot, grip_flag, etc.) are
#        updated asynchronously by the MQTT callback.
#
#   2. Simulation Environment (Isaac Sim / Omniverse)
#      • Loads a USD stage containing the PiPER robot, cameras, and
#        manipulatable objects (a bin and a dex_cube).
#      • Exposes RGB cameras (currently one mid-mounted Realsense) for
#        image capture and optional video recording.
#      • Randomizes object poses at the start of each episode (Button A).
#
#   3. RMP2 Safe Motion Solver
#      • An instance of RMP2Solver (from rmp2_bridge) computes optimal
#        joint accelerations given an EEF goal and obstacle positions.
#      • Uses finite-difference forward kinematics and simple EEF-to-obstacle
#        distance metrics.
#      • Joint limits and a default rest pose are enforced.
#      • The resulting joint positions are applied to the robot each step.
#
#   4. Lula Inverse Kinematics
#      • A LulaKinematicsSolver is configured (via YAML + URDF) and wrapped
#        in an ArticulationKinematicsSolver for end-effector frame tracking.
#      • Currently used for reference; the main control loop relies on RMP2.
#
#   5. Predictive Model (Phase 2 — Currently Disabled)
#      • Infrastructure is present to load a trained transformer policy
#        (checkpoint + dataset statistics) and to run inference on captured
#        camera images.
#      • RDT-1B (Robotics Diffusion Transformer) config is also kept for
#        future re-enablement.
#      • At present, all policy inference code is inactive; the script
#        operates in pure RMP2 teleop mode.
#
#   6. Gripper Control
#      • Effort-based open/close driven by the VR grip flag.
#
#   7. Recording & Episode Management
#      • Pressing Button B starts video recording; pressing Button A resets
#        the environment, randomizes obstacles, and saves the recorded video
#        to dataset/ directories.
#      • Thumbstick inputs control alpha_pos / alpha_rot blending parameters
#        (reserved for future alpha-blending between VR and model outputs).
#
# Controls (VR → MQTT)
# --------------------
#   Trigger press (rising edge)  →  Start delta-tracking from current EEF pose
#   Trigger held                 →  Follow VR controller delta
#   Trigger release              →  Hold / save current pose
#   Button A                     →  Reset environment + new episode
#   Button B                     →  Toggle video recording
#   Thumbstick (1/3)             →  Increment / decrement alpha_pos
#   Thumbstick (0/2)             →  Increment / decrement alpha_rot
#
# Key Dependencies
# ----------------
#   • Isaac Sim / Omniverse Kit (SimulationApp, SimulationContext, prims)
#   • Lula IK Solver (omni.isaac.motion_generation)
#   • Custom RMP2Solver (rmp2_bridge.py)
#   • Paho MQTT Client
#   • OpenCV, NumPy, PyTorch, Gin Config
#
# Typical Usage
# -------------
#   python rdt_sim_vr.py
#
#   Ensure the MQTT broker (sora2.uclab.jp:1883) is reachable and that a
#   VR client is publishing to the "robot/piper-wee" topic.
# =============================================================================

import argparse
import math
import pickle
import shutil
import collections
import json
import os
import time
import warnings

import numpy as np
import cv2
import gin
import yaml
import torch
from paho.mqtt import client as mqtt_client
from PIL import Image as PImage
from scipy.spatial.transform import Rotation as R

from utils import get_image, create_video_writer
import sys

sys.path.insert(0, '/home/ucluser/VRWIT/RL/predictive_model')
from rmp2_bridge import RMP2Solver

warnings.filterwarnings("ignore", message=".*has been deprecated.*")

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.objects import VisualCuboid
from omni.isaac.motion_generation import ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula import LulaKinematicsSolver
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from pxr import UsdPhysics


# ---------------------------
# |  VR-server Connection   |
# ---------------------------
BROKER = "sora2.uclab.jp"
PORT = 1883
CLIENT_ID = 'PiPER-control-wee'
TOPIC = "control/piper-wee"


def connect_mqtt() -> mqtt_client:
    """Connect to the MQTT broker for VR controller data."""
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")

    client = mqtt_client.Client(
        client_id=CLIENT_ID,
        callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2,
    )
    client.on_connect = on_connect
    client.connect(BROKER, PORT)
    return client


# --- Global VR state (updated by MQTT callback) ---
vr_goal_pos = [0.2, 0.2, 0.2]
vr_goal_rot = [0.0, 0.0, 0.0, 1.0]  # Quaternion (x, y, z, w)
grip_flag = False
trigger_on = None
prev_trigger_on = False
controller_obj = [0.0, 0.0, 0.0]
buttonA = False
buttonB = False
thumbstick = None
mqtt_client = connect_mqtt()

# --- MQTT latency monitoring (used when DELAY=True) ---
recv_times = collections.deque(maxlen=20)
recv_messages = collections.deque(maxlen=1000)
old_time = time.time()
avg = 0.0
DELAY = False

# --- ANSI terminal helpers ---
SAVE = "\033[s"
RESTORE = "\033[u"
CLEAR = "\033[K"


def subscribe(client: mqtt_client):
    """Subscribe to VR controller MQTT topic and update global state."""
    def on_message(client, userdata, msg):
        global vr_goal_pos, vr_goal_rot, grip_flag, trigger_on
        global controller_obj, buttonA, buttonB, thumbstick
        global recv_times, recv_messages, old_time, avg

        now = time.monotonic()
        time_new = time.time()
        recv_times.append(now)
        data = msg.payload.decode()

        # Parse once, access multiple times
        data_json = json.loads(data)
        buttonA = data_json['buttonA']
        buttonB = data_json['buttonB']
        thumbstick = data_json['thumbstick']

        if DELAY:
            recv_messages.append((data, time_new))
            print(f"size of recv_messages: {len(recv_messages)}")
            print(SAVE + "\033[4A" + CLEAR +
                  f"size of recv_messages: {len(recv_messages)}" + RESTORE,
                  end='', flush=True)
            if len(recv_times) >= recv_times.maxlen:
                span = recv_times[-1] - recv_times[0]
                avg = (len(recv_times) - 1) / span
                print(SAVE + "\033[3A" + CLEAR +
                      f"Subscriber avg freq: {avg:.1f} Hz" + RESTORE,
                      end="", flush=True)
        else:
            vr_goal_pos = [data_json['goal_pos']['z'], data_json['goal_pos']['x'], data_json['goal_pos']['y']]
            # print(f"vr_goal_pose = {vr_goal_pos}")
            controller_obj = [
                data_json['controller_object']['_x'],
                data_json['controller_object']['_y'],
                data_json['controller_object']['_z'],
            ]
            vr_goal_rot = data_json['goal_rot']
            grip_flag = data_json['grip']
            trigger_on = data_json['sending']

    client.subscribe(TOPIC)
    client.on_message = on_message   



# ---------------------------
# |       Simulation        |
# ---------------------------
STAGE_PATH = "/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description/piper_env_warehouse_5.usd"
open_stage(STAGE_PATH)

sim = SimulationContext()
dt = sim.get_physics_dt()
sim.reset()
sim.play()

set_camera_view(
    eye=[2.0, 0.0, 4.0],
    target=[0.0, 0.0, 2.5],
)

# --- Camera config ---
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FREQ = 20

# --- Robot prims ---
robot = SingleArticulation("/World/piper_description")
robot.initialize()

base                = SingleRigidPrim("/World/piper_description/base_link")
piper_hand          = SingleRigidPrim("/World/piper_description/piper_hand")
linkEndEffector     = SingleXFormPrim("/World/piper_description/piper_hand/linkEndEffector")
# cube                = SingleXFormPrim("/World/Xform")
bin                 = SingleXFormPrim("/World/Xform_bin")
# teddy_bear          = SingleXFormPrim("/World/Xform_teddy_bear")
dex_cube            = SingleXFormPrim("/World/Xform_dex_cube")
# rubik               = SingleXFormPrim("/World/Xform_rubik")
# nvidia_cube         = SingleXFormPrim("/World/Xform_nvidia_cube")
# mug                 = SingleXFormPrim("/World/Xform_mug")

# body_rgb_cam        = Camera(
#                     prim_path="/World/piper_description/piper_hand/Xform_camera/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
#                     frequency=frequency,
#                     resolution=(width, height),)
mid_rgb_cam        = Camera(
                    prim_path="/World/Realsense_mid/RSD455/Camera_OmniVision_OV9782_Color",
                    frequency=CAM_FREQ,
                    resolution=(CAM_WIDTH, CAM_HEIGHT),)
# left_rgb_cam        = Camera(
#                     prim_path="/World/Realsense_left/RSD455/Camera_OmniVision_OV9782_Color",
#                     frequency=frequency,
#                     resolution=(width, height),)
# body_depth_cam      = Camera(
#                     prim_path="/World/piper_description/piper_hand/Xform_camera/Realsense/RSD455/Camera_Pseudo_Depth",
#                     frequency=frequency,
#                     resolution=(width, height),)
# mid_depth_cam        = Camera(
#                     prim_path="/World/Realsense_mid/RSD455/Camera_Pseudo_Depth",
#                     frequency=frequency,
#                     resolution=(width, height),)
# left_depth_cam        = Camera(
#                     prim_path="/World/Realsense_left/RSD455/Camera_Pseudo_Depth",
#                     frequency=frequency,
#                     resolution=(width, height),)

# rgb_cams            = [body_rgb_cam, mid_rgb_cam, left_rgb_cam]
rgb_cams            = [mid_rgb_cam]
depth_cams          = None
# depth_cams          = [body_depth_cam, mid_depth_cam, left_depth_cam]

base.initialize()
piper_hand.initialize()
linkEndEffector.initialize()
# body_rgb_cam.initialize()
mid_rgb_cam.initialize()
# left_rgb_cam.initialize()
# body_depth_cam.initialize()
# mid_depth_cam.initialize()
# left_depth_cam.initialize()
# body_depth_cam.add_distance_to_image_plane_to_frame()
# mid_depth_cam.add_distance_to_image_plane_to_frame()
# left_depth_cam.add_distance_to_image_plane_to_frame()
# body_depth_cam.get_annotator("distance_to_image_plane")


# body_rgb_cam.add_motion_vectors_to_frame()
robot_prim = get_prim_at_path("/World/piper_description")
stage = robot_prim.GetStage()

for prim in stage.Traverse():
    if not prim.GetPath().HasPrefix(robot_prim.GetPath()):
        continue
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.GetStiffnessAttr().Set(1e4)
        drive.GetDampingAttr().Set(1e2)

# ---------------------------
# | IK Target Visualization |
# ---------------------------
VR_target_marker = VisualCuboid(
    prim_path="/World/IK_VR_Target",
    position=[0.0, 0.0, 2.5],
    scale=[0.03, 0.03, 0.03],
    color=np.array([1.0, 0.0, 0.0]),  # red
)

Model_target_marker = VisualCuboid(
    prim_path="/World/IK_Model_Target",
    position=[0.0, 0.0, 2.6],
    scale=[0.03, 0.03, 0.03],
    color=np.array([0.0, 1.0, 0.0]),  # green
)

# ---------------------------
# |        Lula IK          |
# ---------------------------
PIPER_YAML = "/home/ucluser/isaacgym/assets/urdf/piper_description/config/piper_robot.yaml"
PIPER_URDF = "/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description.urdf"

ik_solver = LulaKinematicsSolver(
    robot_description_path=PIPER_YAML,
    urdf_path=PIPER_URDF,
)
ik_solver.set_default_position_tolerance(0.06)
ik_solver.set_default_orientation_tolerance(0.06)

kin_solver = ArticulationKinematicsSolver(
    robot_articulation=robot,
    kinematics_solver=ik_solver,
    end_effector_frame_name="linkEndEffector",
)

piper_hand_pos_world, _ = piper_hand.get_world_pose()
base_pos_world, base_rot_world = base.get_world_pose()
R_base = R.from_quat(base_rot_world)

# Initial reachable target
target_pos_world = np.array([0.2, 0.0, 0.25])
VR_target_marker.set_world_pose(position=target_pos_world)
Model_target_marker.set_world_pose(position=target_pos_world)

# --------------------------------------
# |    Predictive Model Config          |
# |    (Phase 2 — currently disabled)    |
# --------------------------------------
SAVE_DIR = "/home/ucluser/debug_images"
INPUT_SAVE_DIR = "/home/ucluser/input_images"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(INPUT_SAVE_DIR, exist_ok=True)

alpha_pos = 0.0
alpha_rot = 0.0

# Policy hyperparameters
STATE_DIM = 8
ACTION_DIM = 7
CHUNK_SIZE = 25
MAX_TIMESTEPS = 4000
TEMPORAL_AGG = True

POLICY_CONFIG = {
    'lr': 1e-4,
    'num_queries': CHUNK_SIZE,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 2048,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['cam1', 'cam2'],
    'vq': False,
    'action_dim': ACTION_DIM,
    'state_dim': STATE_DIM,
}

ROOT_DIR = "/home/ucluser/VRWIT/RL/predictive_model"
gin.parse_config_file(f"{ROOT_DIR}/configs/base_train_config.gin", skip_unknown=True)

camera_names = POLICY_CONFIG["camera_names"]
CKPT_DIR = f"{ROOT_DIR}/checkpoint_{CHUNK_SIZE}_{10}"
CKPT_NAME = 'policy_best.ckpt'
CKPT_PATH = os.path.join(CKPT_DIR, CKPT_NAME)

# Load dataset statistics for normalization
stats_path = os.path.join(CKPT_DIR, 'dataset_stats.pkl')
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)


def pre_process(s_obs):
    """Normalize observation using dataset statistics."""
    return (s_obs - stats['obs_mean']) / stats['obs_std']


def post_process(a):
    """Denormalize action using dataset statistics."""
    return a * stats['action_std'] + stats['action_mean']


query_frequency = POLICY_CONFIG['num_queries']

if TEMPORAL_AGG:
    end_time = 0.99 * MAX_TIMESTEPS
    query_frequency = 10
    num_queries = POLICY_CONFIG['num_queries']
    all_time_actions = torch.zeros(
        [MAX_TIMESTEPS, MAX_TIMESTEPS + num_queries, ACTION_DIM]
    ).cuda()
else:
    end_time = MAX_TIMESTEPS * 2

print(f"Query frequency: {query_frequency}")
print(f"Checkpoint path: {CKPT_PATH}")

# ---- RDT-1B config (Phase 2 — currently disabled) ----
# These are kept for future re-enablement
RDT_ARGS = {
    'max_publish_step': 300000,
    'chunk_size': 32,
    'arm_steps_length': [0.01] * 6 + [0.2],
    'use_actions_interpolation': False,
    'use_depth_image': False,
    'disable_puppet_arm': False,
    'config_path': "/home/ucluser/RoboticsDiffusionTransformer/configs/base.yaml",
    'pretrained_model_name_or_path': "/home/ucluser/RoboticsDiffusionTransformer/checkpoints",
    'lang_embeddings_path': "/home/ucluser/RoboticsDiffusionTransformer/outs/object_collection.pt",
    'ctrl_freq': 15,
    'use_robot_base': False,
}


# ---------------------------
# |     RMP2 Solver Init    |
# ---------------------------
PIPER_JOINT_COUNT = 6  # 6 actuated arm joints (not counting gripper pair)
DEFAULT_JOINT = np.array([0.0, 0.1, -0.2, 0.0, 0.0, 0.0])
JOINT_LIMITS_LOW = np.array([-2.6179, 0.0, -2.967, -1.745, -1.22, -2.09439])
JOINT_LIMITS_HIGH = np.array([2.6179, 3.14, 0.0, 1.745, 1.22, 2.09439])


def _piper_fk(q7):
    """Compute EEF position + 3×6 Jacobian analytically (NO sim.steps)."""
    q7 = np.asarray(q7, dtype=np.float64)
    ee_pos_world, _ = linkEndEffector.get_world_pose()
    eef = np.asarray(ee_pos_world, dtype=np.float64)

    try:
        ee_link = "linkEndEffector"
        spatial_J = robot.calculate_jacobian(ee_link, "world", q7)
        # Handle tensor/array formats from Isaac Sim
        if hasattr(spatial_J, 'numpy'):
            spatial_J = spatial_J.numpy()
        if spatial_J.ndim == 3:
            spatial_J = spatial_J[0]
        # spatial_J is 6×D (pos+rot), take top 3 rows for translational
        J_ee = np.asarray(spatial_J[:3, :PIPER_JOINT_COUNT], dtype=np.float64)
        if J_ee.shape != (3, PIPER_JOINT_COUNT):
            raise ValueError(f"Bad jacobian shape: {J_ee.shape}")
        return eef, J_ee
    except Exception as e:
        # Fallback: finite-diff FK (only if calculate_jacobian fails)
        eps = 1e-4
        J = np.zeros((3, PIPER_JOINT_COUNT), dtype=np.float64)
        for j in range(PIPER_JOINT_COUNT):
            q_plus = q7.copy()
            q_plus[j] += eps
            robot.set_joint_positions(q_plus.tolist(),
                                      joint_indices=list(range(PIPER_JOINT_COUNT)))
            sim.step(render=False)
            ej, _ = linkEndEffector.get_world_pose()
            J[:, j] = (np.asarray(ej, dtype=np.float64) - eef) / eps
        robot.set_joint_positions(q7.tolist(),
                                  joint_indices=list(range(PIPER_JOINT_COUNT)))
        return eef, J


def _piper_collision(q, eef, obstacles):
    """Return list of distances from EEF to each obstacle position."""
    if not obstacles:
        return []
    distances = []
    for obs_prim in obstacles:
        try:
            obs_pos_world, _ = obs_prim.get_world_pose()
            obs_pos = np.asarray(obs_pos_world, dtype=np.float64)
            distances.append(float(np.linalg.norm(eef - obs_pos)))
        except Exception:
            distances.append(10.0)
    return distances


# Build solver
rmp2_solver = RMP2Solver(
    n_joints=PIPER_JOINT_COUNT,
    fk_fn=_piper_fk,
    collision_fn=_piper_collision,
    joint_limits_low=JOINT_LIMITS_LOW,
    joint_limits_high=JOINT_LIMITS_HIGH,
    default_q=DEFAULT_JOINT,
    dt=0.02,  # matches Isaac Sim dt
)

print("[RMP2 Phase 1] Solver initialized — RDT disabled for now.")


# ---------------------------
# |     Utils functions     |
# ---------------------------

def count_files_in_directory(directory_path: str) -> int:
    """Count the number of files in a directory (non-recursive)."""
    try:
        return len([
            f for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ])
    except Exception as e:
        print(f"Error counting files in {directory_path}: {e}")
        return 0


def image_capture(writers, depth_stacks, rgb_cams, depth_cams,
                  width, height, record) -> list | None:
    """
    Capture and process RGB(+depth) images from camera list.

    Returns:
        List of RGBD images (numpy arrays [height, width, 4]) or None on failure.
    """
    rgbd_images = []
    if depth_cams is not None:
        assert len(rgb_cams) == len(depth_cams)

    for i in range(len(rgb_cams)):
        img = rgb_cams[i].get_rgba()
        if len(img) == 0:
            return None

        # Depth (dummy zeros if no depth cameras)
        if depth_cams is None:
            depth_image = np.full((height, width, 1), 0.0, dtype=np.float32)
        else:
            depth = depth_cams[i].get_depth()
            depth_image = np.clip(depth.copy(), 0.0, 5.0)
            depth_image = depth_image.reshape((height, width, 1))

        # Color
        color_image = img.copy().reshape((height, width, 4))
        rgb_image = color_image[:, :, :3]
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Write to video
        writers[i].write(bgr_image)
        if record:
            writers[i].write(bgr_image)
            depth_stacks[i].append(depth_image)

        rgbd_images.append(np.concatenate((rgb_image, depth_image), axis=2))

    return rgbd_images

def random_pose(is_tray: bool):
    """Generate a random pose for object placement."""
    z = 2.8 if is_tray else 2.45
    if is_tray:
        regions = [
            ((0.1, 0.5), (-0.6, -0.3)),
            ((0.1, 0.5), (0.3, 0.6)),
            ((0.5, 0.6), (-0.6, 0.6)),
        ]
    else:
        regions = [
            ((0.0, 0.5), (-1.0, -0.3)),
            ((0.0, 0.5), (0.3, 1.0)),
            ((0.5, 1.15), (-1.0, 1.0)),
        ]

    areas = [(x1 - x0) * (y1 - y0) for (x0, x1), (y0, y1) in regions]
    p = np.array(areas) / sum(areas)

    region = regions[np.random.choice(len(regions), p=p)]
    (x0, x1), (y0, y1) = region

    x = np.random.uniform(x0, x1)
    y = np.random.uniform(y0, y1)

    return np.array([x, y, z]), [0, 0, 0, 1]


def set_new_poses():
    """Randomize obstacle positions for a new episode."""
    pos_bin, q_bin = random_pose(is_tray=True)
    pos_cube, q_cube = random_pose(is_tray=False)

    bin.set_world_pose(position=pos_bin, orientation=q_bin)
    dex_cube.set_world_pose(position=pos_cube, orientation=q_cube)


def convert_6d_to_quaternion(rotation_6d):
    """
    Convert 6D rotation representation to quaternion (x, y, z, w).

    The 6D representation uses two orthonormal 3D vectors:
    - First 3 dims: x-axis basis vector
    - Last 3 dims: y-axis basis vector
    - z-axis is computed via cross product
    """
    is_batch = len(rotation_6d.shape) > 1
    if not is_batch:
        rotation_6d = rotation_6d[np.newaxis, :]

    quaternions = []
    for i in range(rotation_6d.shape[0]):
        vec1 = rotation_6d[i, :3].astype(np.float64)
        vec2 = rotation_6d[i, 3:6].astype(np.float64)

        # Normalize first vector
        norm1 = np.linalg.norm(vec1)
        vec1 = vec1 / norm1 if norm1 > 1e-8 else np.array([1.0, 0.0, 0.0])

        # Gram-Schmidt orthogonalization
        vec2 = vec2 - np.dot(vec1, vec2) * vec1
        norm2 = np.linalg.norm(vec2)
        if norm2 < 1e-8:
            vec2 = np.array([0.0, 1.0, 0.0]) if abs(vec1[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        else:
            vec2 = vec2 / norm2

        vec3 = np.cross(vec1, vec2)
        rotation_matrix = np.column_stack([vec1, vec2, vec3])
        quat = R.from_matrix(rotation_matrix).as_quat()  # (x, y, z, w)
        quaternions.append(quat)

    result = np.array(quaternions)
    return result[0] if not is_batch else result
# ---------------------------
# |        Main loop        |
# ---------------------------

# --- Create output directories ---
os.makedirs("dataset/states", exist_ok=True)
os.makedirs("dataset/actions", exist_ok=True)
for cam_idx in range(len(rgb_cams)):
    os.makedirs(f"videos/cam_{cam_idx}", exist_ok=True)

ep_tracker = count_files_in_directory("videos/cam_0")

# --- Teleop state ---
quat_save = euler_angles_to_quat(np.array([0.5 * math.pi, 0, 0]))
quat_start = None
pos_save = np.array([0.14, -0.03, 2.5]) - base_pos_world
_ee_pos_world, _current_rotation = linkEndEffector.get_world_pose()

# --- Loop variables ---
t = 0
record = False
writers = [
    create_video_writer(f"rgb_{i}", CAM_FREQ, CAM_WIDTH, CAM_HEIGHT)
    for i in range(len(rgb_cams))
]
depth_stacks = []

print(f"Number of video writers: {len(writers)}")

# --- Start MQTT listener ---
subscribe(mqtt_client)
mqtt_client.loop_start()

# --- Main control loop ---
with torch.inference_mode():
    while simulation_app.is_running():
        sim.step(render=True)
        t0 = time.perf_counter()

        # ================================================================
        # 1. Gripper control (effort-based)
        # ================================================================
        joint_efforts = robot.get_measured_joint_efforts()
        if grip_flag:
            joint_efforts[6] = -50.0
            joint_efforts[7] = 50.0
        else:
            joint_efforts[6] = 50.0
            joint_efforts[7] = -50.0

        grip_action = ArticulationAction(joint_efforts=joint_efforts)
        robot.apply_action(grip_action)

        # ================================================================
        # 2. Recording control
        # ================================================================
        if buttonB:
            record = True
            print("The video is being recorded!")

        # ================================================================
        # 3. Camera capture
        # ================================================================
        # Only capture when recording (avoids blocking get_rgba every frame)
        current_q = robot.get_joint_positions()
        if record:
            captured_img = image_capture(
                writers, depth_stacks, rgb_cams, depth_cams,
                CAM_WIDTH, CAM_HEIGHT, record=True,
            )

        # ================================================================
        # 4. Alpha blending control (thumbstick)
        # ================================================================
        if thumbstick == 1:
            alpha_pos = min(alpha_pos + 0.1, 1.0)
        elif thumbstick == 3:
            alpha_pos = max(alpha_pos - 0.1, 0.0)
        elif thumbstick == 0:
            alpha_rot = min(alpha_rot + 0.1, 1.0)
        elif thumbstick == 2:
            alpha_rot = max(alpha_rot - 0.1, 0.0)
        alpha_pos = round(alpha_pos, 1)
        alpha_rot = round(alpha_rot, 1)

        # ================================================================
        # 5. RMP2 Teleop Control
        # ================================================================
        t2 = time.perf_counter()

        # Build EEF goal from VR controller
        # print(f"vr_goal_pose = {vr_goal_pos}")
        
        pos_controller = np.array(vr_goal_pos, dtype=np.float64)
        quat_controller = euler_angles_to_quat(np.array([controller_obj[2], controller_obj[0],controller_obj[1]])) #
        # Grab-to-move delta tracking (rising edge)
        # print(f"trigger = {trigger_on} | previous trigger = {prev_trigger_on}")
        if trigger_on and not prev_trigger_on:
            ee_pos_world_now, _ = linkEndEffector.get_world_pose()
            pos_start = ee_pos_world_now - base_pos_world
            pos_start_ctrl = pos_controller.copy()
            
            quat_start = R.from_quat(np.asarray(quat_save, dtype=np.float64))       # w,x,y,z
            quat_start_ctrl = R.from_quat(np.asarray(quat_controller, dtype=np.float64))

        # On first trigger press, initialize start rotation
        if trigger_on and quat_start is None:
            quat_init = euler_angles_to_quat(np.array([0.5 * math.pi, 0, 0]))
            quat_start = R.from_quat(quat_init)
        
        # Save pose on trigger release (falling edge)
        if prev_trigger_on and not trigger_on:
            pos_save = target_pos_world if 'target_pos_world' in dir() else pos_save
            quat_save = target_rot_world if 'target_rot_world' in dir() else quat_save

        # Compute target pose if trigger has been pressed at least once
        # if quat_start is not None:
        if trigger_on:
            pos_ctrl_delta = pos_controller - pos_start_ctrl
            target_pos_world = pos_start + pos_ctrl_delta
            # print(f"targer_pos_world = {target_pos_world}")

            quat_ctrl = R.from_quat(np.asarray(quat_controller, dtype=np.float64))
            quat_ctrl_delta = quat_start_ctrl.inv() * quat_ctrl
            quat_current = quat_start * quat_ctrl_delta
            target_rot_world = quat_current.as_quat()

            VR_target_marker.set_world_pose(
                position=target_pos_world + base_pos_world,
                orientation=target_rot_world,
            )
        else:
            # Trigger released: hold saved pose
            target_pos_world = pos_save
            target_rot_world = (
                quat_save.as_quat() if hasattr(quat_save, 'as_quat')
                else quat_save
            )


        prev_trigger_on = trigger_on

        # RMP2 solve
        # obstacles_list = [bin, dex_cube]
        obstacles_list = None
        

        q_arm = np.array(current_q[:PIPER_JOINT_COUNT], dtype=np.float64)
        qd_arm = np.array(
            robot.get_joint_velocities()[:PIPER_JOINT_COUNT],
            dtype=np.float64,
        )

        # EEF goal for RMP2 Target RMP (world frame)
        eef_goal = (
            np.asarray(target_pos_world, dtype=np.float64)
            + np.asarray(base_pos_world, dtype=np.float64)
        )

        # Solve RMP2 → optimal joint acceleration
        q_ddot = rmp2_solver.solve(
            q=q_arm, qd=qd_arm,
            goals=[eef_goal],
            obstacles=obstacles_list,
        )

        # Integrate → new joint positions
        q_new, qd_new = rmp2_solver.integrate(q_arm, qd_arm, q_ddot)
        q_safe = rmp2_solver.apply_hard_limits(q_new)

        # Append gripper joints (unchanged, effort-controlled)
        full_q = list(q_safe) + list(current_q[PIPER_JOINT_COUNT:])
        robot_action = ArticulationAction(joint_positions=full_q)
        robot.apply_action(robot_action)

        t3 = time.perf_counter()
        Model_target_marker.set_world_pose(position=eef_goal)

        # ================================================================
        # 6. Console stats
        # ================================================================
        t4 = time.perf_counter()

        # Update console stats every 10 frames + print RMP2 timing
        t5 = time.perf_counter()
        # if t % 10 == 0:
        #     print(
        #         SAVE + "\033[1A" + CLEAR
        #         + f"RMP2 teleop | "
        #         + f"trigger={'ON' if trigger_on else 'OFF'} | "
        #         + f"alpha_pos={alpha_pos:.1f} alpha_rot={alpha_rot:.1f} | "
        #         + f"rec={record} | "
        #         + f"RMP2={((t5-t2)*1000):.1f}ms tot={((t5-t0)*1000):.1f}ms"
        #         + RESTORE,
        #         end="", flush=True,
        #     )

        t += 1

        # ================================================================
        # 7. Environment reset (Button A)
        # ================================================================
        if buttonA:
            t = 0
            sim.stop()
            sim.reset()
            set_new_poses()
            base.initialize()
            robot.initialize()
            piper_hand.initialize()
            linkEndEffector.initialize()
            target_pos_world = np.array([0.14, -0.03, 2.5]) - base_pos_world

            if record:
                print(f"Saving successful Ep {ep_tracker}")
                for cam_idx in range(len(rgb_cams)):
                    shutil.move(
                        f"rgb_{cam_idx}.avi",
                        f"videos/cam_{cam_idx}/rgb_ep_{ep_tracker}.avi",
                    )
                ep_tracker += 1
            else:
                for cam_idx in range(len(rgb_cams)):
                    try:
                        os.remove(f"rgb_{cam_idx}.avi")
                    except FileNotFoundError:
                        pass

            sim.play()
            sim.step(render=True)
            record = False
            continue


simulation_app.close()
mqtt_client.loop_stop()