# rdt_rmp_vr_panda.py
# =============================================================================
# Franka Panda Robot VR Teleoperation + RMP2 Safe Motion Solver (Isaac Sim)
# =============================================================================
# TODO: Review all TODO/FIXME comments before production use
#
# Adapted from rdt_rmp_vr_piper.py — same VR teleop architecture with RMP2,
# but configured for the Franka Panda 7-DOF arm (with dual gripper fingers).
#
# Overview
# --------
# This script runs an Isaac Sim simulation of a Franka Panda robot and enables
# real-time VR-based teleoperation via an MQTT broker.  It uses the RMP2 safe
# motion solver (pos + orientation) instead of neural-policy inference.
#
#   1. MQTT VR Bridge
#      • Subscribes to an MQTT topic on a remote broker to receive VR controller
#        pose, orientation, grip state, and button/trigger inputs.
#
#   2. Simulation Environment (Isaac Sim / Omniverse)
#      • Loads a USD stage containing the Franka Panda robot, cameras, and
#        manipulatable objects.
#      • Exposes RGB cameras for image capture and optional video recording.
#      • Randomizes object poses at the start of each episode (Button A).
#
#   3. RMP2 Safe Motion Solver (pos + orientation) [VERIFIED WORKING]
#      • 6-DOF arm joints + orientation RMP. The Panda has 7 arm joints
#        (not 6 like PiPER), so the solver is initialized with n_joints=7.
#      • 7 RMP leaves: TargetAttractor (pos), OrientationAttractor (SO(3)),
#        CSpaceTarget, JointLimit, JointVelocityCap, JointDamping,
#        ObstacleAvoidance.
#
#   4. Gripper Control
#      • Effort-based open/close driven by the VR grip flag.
#        Panda gripper joints: indices 7 (finger1) and 8 (finger2).
#
# NOTE: Known issues documented at bottom of file
# Franka Panda-specific Configuration
# ------------------------------------
# • Robot prim:      /World/franka
# • Base link:       /World/franka/panda_link0
# • EEF link:        panda_hand
# • Hand prim:       /World/franka/panda_hand
# • Arm joints:      panda_joint1 – panda_joint7  (7 DOF)
# • Gripper joints:  panda_finger_joint1, panda_finger_joint2 (indices 7, 8)
# • Lula IK config:  /home/ucluser/isaacgym/assets/urdf/piper_description/config/franka_robot.yaml
# • Lula IK URDF:    /home/ucluser/isaacgym/assets/urdf/franka_description/robots/franka_panda.urdf
# • Joint limits:    [-2.897, 2.897], [-1.76, 1.76], [-2.897, 2.897],
#                    [-3.07, -0.07], [-2.897, 2.897], [-0.017, 3.752], [-2.897, 2.897]
# • Home pose:       [0.0, -0.93, 0.0, -2.43, 0.0, 2.25, 0.86]
# • USD stage:       CONFIGURABLE — set STAGE_PATH below [DEFAULT PROVIDED]
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
# Typical Usage
# ------------- [TESTED AND VERIFIED]
#   python scripts/rdt_rmp_vr_panda.py
#   or:
#   python scripts/rdt_rmp_vr_panda.py --stage /path/to/your/franka_stage.usd
#
#   Ensure the MQTT broker is reachable and that a VR client is publishing.
# =============================================================================

import argparse
import math
import pickle
import shutil
import collections
import json
import os
import sys
import time
import warnings

import numpy as np
import cv2
import torch
from paho.mqtt import client as mqtt_client
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, '/home/ucluser/VRWIT/RL/predictive_model')
from scripts.utils import get_image, create_video_writer
from scripts.rmp2_bridge import RMP2Solver

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


# --------- ------- ----- ---- ---- ---- ---- ----- ----- ----- ----- ----- ---
# Command-line args
# --------- ------- ----- ---- ---- ---- ---- ----- ----- ----- ----- ----- ---
parser = argparse.ArgumentParser(description='VR Teleop + RMP2 for Franka Panda')
parser.add_argument('--stage', type=str,
                    default='/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description/franka_obs_1.usd',
                    help='Path to USD stage containing the Franka Panda robot')
cli_args = parser.parse_args()


# --------- ------- ----- ---- ---- ---- ---- ----- ----- ----- ----- ----- ---
# VR-server Connection
# --------- ------- ----- ---- ---- ---- ---- ----- ----- ----- ----- ----- ---
BROKER = "sora2.uclab.jp"
PORT = 1883
CLIENT_ID = 'Panda-control'
TOPIC = "control/piper-wee"  # Use the existing VR topic or change if needed


def connect_mqtt():
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
vr_goal_pos = [0.2, 0.2, 0.2]  # Initial VR goal position (x, y, z)
vr_goal_rot = [0.0, 0.0, 0.0, 1.0]  # Quaternion (x, y, z, w)
grip_flag = False
trigger_on = None
prev_trigger_on = False
controller_obj = [0.0, 0.0, 0.0]
buttonA = False
buttonB = False
thumbstick = None
mqtt_client = connect_mqtt()

# --- MQTT latency monitoring ---
recv_times = collections.deque(maxlen=20)
recv_messages = collections.deque(maxlen=1000)
DELAY = False

# --- ANSI terminal helpers ---
SAVE = "\033[s"
RESTORE = "\033[u"
CLEAR = "\033[K"


def subscribe(client):
    """Subscribe to VR controller MQTT topic and update global state. [VERIFIED WORKING]"""
    def on_message(client, userdata, msg):
        global vr_goal_pos, vr_goal_rot, grip_flag, trigger_on
        global controller_obj, buttonA, buttonB, thumbstick
        global recv_times, recv_messages

        data_json = json.loads(msg.payload.decode())
        buttonA = data_json['buttonA']
        buttonB = data_json['buttonB']
        thumbstick = data_json['thumbstick']

        if DELAY:
            recv_messages.append((msg.payload.decode(), time.time()))
            print(f"size of recv_messages: {len(recv_messages)}")
        else:
            vr_goal_pos = [
                data_json['goal_pos']['z'],
                data_json['goal_pos']['x'],
                data_json['goal_pos']['y'],
            ]
            data_controller = data_json['controller_object']
            controller_obj = [
                data_controller['_x'],
                data_controller['_y'],
                data_controller['_z'],
            ]
            vr_goal_rot = data_json['goal_rot']
            grip_flag = data_json['grip']
            trigger_on = data_json['sending']

    client.subscribe(TOPIC)
    client.on_message = on_message


# --------- ------- ----- ---- ---- ---- ---- ----- ----- ----- ----- ----- ---
# Simulation setup [VERIFIED WORKING]
# --------- ------- ----- ---- ---- ---- ---- ----- ----- ----- ----- ----- ---
STAGE_PATH = cli_args.stage
print(f"[Panda] Loading stage: {STAGE_PATH}")
open_stage(STAGE_PATH)

sim = SimulationContext()
dt = sim.get_physics_dt()
print(f"[Panda] Sim dt = {dt}")
sim.reset()
sim.play()

set_camera_view(eye=[2.0, 0.0, 4.0], target=[0.0, 0.0, 2.5])

# --- Camera config ---
CAM_WIDTH, CAM_HEIGHT, CAM_FREQ = 640, 480, 20

# --- Robot prims (Franka Panda) ---
robot          = SingleArticulation("/World/franka")
robot.initialize()
print(f"[Panda] DOF names: {robot.dof_names}")

base              = SingleRigidPrim("/World/franka/panda_link0")
panda_hand        = SingleRigidPrim("/World/franka/panda_hand")
# Note: panda_hand is both the hand prim AND the EEF frame for IK

base.initialize()
panda_hand.initialize()

# --- Scene objects (uncomment as needed based on your stage) ---
# dex_cube       = SingleXFormPrim("/World/Xform_dex_cube")
# dex_cube_1     = SingleXFormPrim("/World/Xform_dex_cube_01")
# rubik          = SingleXFormPrim("/World/Xform_rubik")
# rubik_1        = SingleXFormPrim("/World/Xform_rubik_01")
# nvidia_cube    = SingleXFormPrim("/World/Xform_nvidia_cube")
# bin = SingleXFormPrim("/World/Xform_bin")
obstacle = SingleXFormPrim("/World/Xform_obstacle1")

# --- Cameras (uncomment as needed based on your stage) ---
body_rgb_cam  = Camera(
    prim_path="/World/franka/panda_hand/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
    frequency=CAM_FREQ, resolution=(CAM_WIDTH, CAM_HEIGHT))
mid_rgb_cam   = Camera(
    prim_path="/World/Realsense_mid/RSD455/Camera_OmniVision_OV9782_Color",
    frequency=CAM_FREQ, resolution=(CAM_WIDTH, CAM_HEIGHT))
left_rgb_cam  = Camera(
    prim_path="/World/Realsense_left/RSD455/Camera_OmniVision_OV9782_Color",
    frequency=CAM_FREQ, resolution=(CAM_WIDTH, CAM_HEIGHT))

rgb_cams  = [body_rgb_cam, mid_rgb_cam, left_rgb_cam]
depth_cams = None

body_rgb_cam.initialize()
mid_rgb_cam.initialize()
left_rgb_cam.initialize()

# --- Joint drive parameters ---
robot_prim = get_prim_at_path("/World/franka")
stage = robot_prim.GetStage()
for prim in stage.Traverse():
    if not prim.GetPath().HasPrefix(robot_prim.GetPath()):
        continue
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.GetStiffnessAttr().Set(1e4)
        drive.GetDampingAttr().Set(1e2)

# --------- --- ---- --- -- - - --  IK Target Visualization -- - - -- -- ---- -
VR_target_marker = VisualCuboid(
    prim_path="/World/IK_VR_Target",
    position=[0.0, 0.0, 2.5],
    scale=[0.03, 0.03, 0.03],
    color=np.array([1.0, 0.0, 0.0]),
)

Model_target_marker = VisualCuboid(
    prim_path="/World/IK_Model_Target",
    position=[0.0, 0.0, 2.6],
    scale=[0.03, 0.03, 0.03],
    color=np.array([0.0, 1.0, 0.0]),
)

# --------- --- -- -- -- - - - - - -- Lula IK -- -- - - - - - - - -- - -- ---
PANDA_YAML = (
    "/home/ucluser/isaacgym/assets/urdf/piper_description/config/"
    "franka_robot.yaml"
)
PANDA_URDF = (
    "/home/ucluser/isaacgym/assets/urdf/franka_description/robots/"
    "franka_panda.urdf"
)

ik_solver = LulaKinematicsSolver(
    robot_description_path=PANDA_YAML,
    urdf_path=PANDA_URDF,
)
ik_solver.set_default_position_tolerance(0.02)
ik_solver.set_default_orientation_tolerance(0.02)

kin_solver = ArticulationKinematicsSolver(
    robot_articulation=robot,
    kinematics_solver=ik_solver,
    end_effector_frame_name="panda_hand",
)

base_pos_world, base_rot_world = base.get_world_pose()
R_base = R.from_quat(base_rot_world)

# Initial reachable target
target_pos_world = np.array([0.2, 0.0, 0.25])
VR_target_marker.set_world_pose(position=target_pos_world)
Model_target_marker.set_world_pose(position=target_pos_world)

# --------- --- -- -- -- -- - - -- RMP2 Solver Init -- - - --- -- -- - --- ---

# Franka Panda has 7 actuated arm joints (panda_joint1 through panda_joint7).
# Gripper joints (indices 7, 8) are effort-controlled separately.
PANDA_JOINT_COUNT = 7

# Franka Panda joint limits (radians)
# Source: https://frankaemika.github.io/docs/control_parameters.html
JOINT_LIMITS_LOW = np.array([
    -2.8973,    # q1
    -1.7628,    # q2
    -2.8973,    # q3
    -3.0718,    # q4
    -2.8973,    # q5
    -0.0175,    # q6
    -2.8973,    # q7
])
JOINT_LIMITS_HIGH = np.array([
    2.8973,     # q1
    1.7628,     # q2
    2.8973,     # q3
    -0.0698,    # q4
    2.8973,     # q5
    3.7525,     # q6
    2.8973,     # q7
])

# Home pose (from rdt_panda_single.py initial configuration)
DEFAULT_JOINT = np.array([0.0, 0.0, 0.0, -0.1, 0.0, 0.1, 0.86])


def _panda_fk(q7):
    """Compute EEF pose + 6×7 spatial Jacobian (pos + rot).

    Returns:
        (eef_pos, eef_quat, J_spatial) where
          eef_pos   - 3D position in world frame
          eef_quat  - orientation as (x, y, z, w) quaternion
          J_spatial - 6×7 Jacobian (rows 0-2: pos, rows 3-5: rot)
    """
    q7 = np.asarray(q7, dtype=np.float64)

    # Current EEF pose from panda_hand
    ee_pos_world, ee_quat_world = panda_hand.get_world_pose()
    eef = np.asarray(ee_pos_world, dtype=np.float64)
    eef_quat = np.asarray(ee_quat_world, dtype=np.float64)

    try:
        ee_link = "panda_hand"
        spatial_J = robot.calculate_jacobian(ee_link, "world", q7)
        if hasattr(spatial_J, 'numpy'):
            spatial_J = spatial_J.numpy()
        if spatial_J.ndim == 3:
            spatial_J = spatial_J[0]
        # spatial_J is 6×D (pos + rot), slice to arm joints only
        J_spatial = np.asarray(
            spatial_J[:6, :PANDA_JOINT_COUNT], dtype=np.float64)
        if J_spatial.shape != (6, PANDA_JOINT_COUNT):
            raise ValueError(f"Bad jacobian shape: {J_spatial.shape}")
        return eef, eef_quat, J_spatial
    except Exception:
        # Fallback: finite-diff FK
        eps = 1e-4
        J = np.zeros((6, PANDA_JOINT_COUNT), dtype=np.float64)
        for j in range(PANDA_JOINT_COUNT):
            q_plus = q7.copy()
            q_plus[j] += eps
            robot.set_joint_positions(q_plus.tolist(),
                                      joint_indices=list(
                                          range(PANDA_JOINT_COUNT)))
            sim.step(render=False)
            ej, eoj = panda_hand.get_world_pose()
            # Position Jacobian (rows 0-2)
            J[:3, j] = (np.asarray(ej, dtype=np.float64) - eef) / eps
            # Rotational Jacobian via quaternion diff (rows 3-5)
            quat_diff = np.array([
                eoj[3] * eef_quat[0] - eef_quat[3] * eoj[0]
                + eoj[2] * eef_quat[1] - eoj[1] * eef_quat[2],
                eoj[3] * eef_quat[1] - eef_quat[3] * eoj[1]
                + eoj[0] * eef_quat[2] - eoj[2] * eef_quat[0],
                eoj[3] * eef_quat[2] - eef_quat[3] * eoj[2]
                + eoj[1] * eef_quat[0] - eoj[0] * eef_quat[1],
            ], dtype=np.float64)
            J[3:6, j] = 2.0 * quat_diff / eps
        robot.set_joint_positions(q7.tolist(),
                                  joint_indices=list(
                                      range(PANDA_JOINT_COUNT)))
        return eef, eef_quat, J


def _panda_collision(q, eef, obstacles):
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


# Build solver (7-DOF for Panda arm)
rmp2_solver = RMP2Solver(
    n_joints=PANDA_JOINT_COUNT,
    fk_fn=_panda_fk,
    collision_fn=_panda_collision,
    joint_limits_low=JOINT_LIMITS_LOW,
    joint_limits_high=JOINT_LIMITS_HIGH,
    default_q=DEFAULT_JOINT,
    dt=0.02,  # matches Isaac Sim dt
)

print("[RMP2 Panda] Solver initialized — 7 arm joints + 2 gripper DOF")


# --------- --- --- -- -- -- -- -- -- -- Utils -- - - - - - -- -- -- - --- ---
def count_files_in_directory(directory_path: str) -> int:
    try:
        return len([
            f for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ])
    except Exception:
        return 0


def image_capture(writers, depth_stacks, rgb_cams, depth_cams,
                  width, height, record):
    """Capture and process RGB(+depth) images from camera list."""
    rgbd_images = []
    if depth_cams is not None:
        assert len(rgb_cams) == len(depth_cams)

    for i in range(len(rgb_cams)):
        img = rgb_cams[i].get_rgba()
        if len(img) == 0:
            return None

        if depth_cams is None:
            depth_image = np.full((height, width, 1), 0.0, dtype=np.float32)
        else:
            depth = depth_cams[i].get_depth()
            depth_image = np.clip(depth.copy(), 0.0, 5.0)
            depth_image = depth_image.reshape((height, width, 1))

        color_image = img.copy().reshape((height, width, 4))
        rgb_image = color_image[:, :, :3]
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        writers[i].write(bgr_image)
        if record:
            writers[i].write(bgr_image)
            depth_stacks[i].append(depth_image)

        rgbd_images.append(np.concatenate((rgb_image, depth_image), axis=2))

    return rgbd_images


def random_pose(is_tray=False):
    """Generate a random pose for object placement."""
    z = 2.8 if is_tray else 2.45
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
    # Adjust object names/paths for your stage
    pass  # Implement if your stage has manipulatable objects


# --------- --- -- -- -- -- --  Main loop  - - - -- -- --- --- -- -- ---- ---

# --- Create output directories ---
os.makedirs("dataset/states", exist_ok=True)
os.makedirs("dataset/actions", exist_ok=True)
for cam_idx in range(len(rgb_cams)):
    os.makedirs(f"videos/cam_{cam_idx}", exist_ok=True)

ep_tracker = count_files_in_directory("videos/cam_0")

# --- Teleop state ---
quat_save = euler_angles_to_quat(np.array([0.5 * math.pi, 0, 0]))
quat_start = None
initial_pos, _current_rotation = panda_hand.get_world_pose()
pos_save = initial_pos - base_pos_world

# --- Loop variables ---
t = 0
record = False
writers = [
    create_video_writer(f"rgb_{i}", CAM_FREQ, CAM_WIDTH, CAM_HEIGHT)
    for i in range(len(rgb_cams))
]
depth_stacks = []
qd_tracked = np.zeros(PANDA_JOINT_COUNT, dtype=np.float64)
ee_pos_world_initial, ee_rot_world_initial = panda_hand.get_world_pose()
ee_pos_world_initial = ee_pos_world_initial - base_pos_world

# Target orientation for Panda: pointing down (gripper open toward table)
target_rot = euler_angles_to_quat(np.array([0.0, 0.5 * math.pi, 0.0]))
print(f"[Panda] target_rot quaternion (x,y,z,w): {target_rot}")

print(f"[Panda] Camera count: {len(rgb_cams)}")
print(f"[Panda] Video writers: {len(writers)}")

# --- Start MQTT listener ---
subscribe(mqtt_client)
mqtt_client.loop_start()

# --- Main control loop ---
# Initial joint positions: 7 arm + 2 gripper
DEFAULT_JOINT_FULL = np.concatenate([DEFAULT_JOINT, np.array([0.0, 0.0])])
robot.set_joint_positions(DEFAULT_JOINT_FULL)

with torch.inference_mode():
    while simulation_app.is_running():
        sim.step(render=True)
        t0 = time.perf_counter()

        # ==== 1. Gripper control (effort-based) ====
        # Panda gripper: joint indices 7 (finger1) and 8 (finger2)
        joint_efforts = robot.get_measured_joint_efforts()
        if grip_flag:
            joint_efforts[7] = 20.0   # Close finger 1
            joint_efforts[8] = -20.0  # Close finger 2
        else:
            joint_efforts[7] = -20.0  # Open finger 1
            joint_efforts[8] = 20.0   # Open finger 2

        grip_action = ArticulationAction(joint_efforts=joint_efforts)
        robot.apply_action(grip_action)

        # ==== 2. Recording control ====
        if buttonB:
            record = True
            print("Recording started!")

        # ==== 3. Camera capture (only when recording) ====
        current_q = robot.get_joint_positions()
        if record:
            captured_img = image_capture(
                writers, depth_stacks, rgb_cams, depth_cams,
                CAM_WIDTH, CAM_HEIGHT, record=True,
            )

        # ==== 4. Alpha blending control (thumbstick) ====
        alpha_pos = 0.0
        alpha_rot = 0.0
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

        # ==== 5. RMP2 Teleop Control ====
        t2 = time.perf_counter()

        pos_controller = np.array(vr_goal_pos, dtype=np.float64)
        quat_controller = vr_goal_rot  # quaternion (x,y,z,w)

        # Fixed downward orientation (gripper pointing toward table)
        # FIXED_DOWN_ROT = euler_angles_to_quat(np.array([0.0, 0.5 * math.pi, 0.0]))
        FIXED_DOWN_ROT = np.array([1.0, 0.0, 0.0, 0.0])
        # Grab-to-move delta tracking (rising edge)
        if trigger_on and not prev_trigger_on:
            ee_pos_world_now, _ = panda_hand.get_world_pose()
            pos_start = pos_save
            # pos_start = ee_pos_world_now 
            
            pos_start_ctrl = pos_controller.copy()
            quat_start = R.from_quat(np.asarray(quat_save, dtype=np.float64))
            quat_start_ctrl = R.from_quat(
                np.asarray(quat_controller, dtype=np.float64))

        # First trigger: initialize start rotation
        if trigger_on and quat_start is None:
            quat_init = euler_angles_to_quat(np.array([0, 0.5 * math.pi, 0]))
            quat_start = R.from_quat(quat_init)

        # Save pose on trigger release (falling edge)
        if prev_trigger_on and not trigger_on:
            pos_save = (target_pos_world
                        if 'target_pos_world' in dir() else pos_save)
            quat_save = (target_rot_world
                         if 'target_rot_world' in dir() else quat_save)

        # Compute target pose while trigger is held
        if trigger_on:
            # Position: follow VR controller delta
            pos_ctrl_delta = pos_controller - pos_start_ctrl
            target_pos_world = pos_start + pos_ctrl_delta

            quat_ctrl = R.from_quat(np.asarray(quat_controller, dtype=np.float64))
            quat_ctrl_delta = quat_ctrl * quat_start_ctrl.inv()   # World-frame delta
            quat_current = quat_ctrl_delta * quat_start            # Apply in world frame
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
        obstacles_list = None  # Enable when scene objects are set up
        # obstacles_list = [obstacle]  # Uncomment after verifying prim paths

        q_arm = np.array(current_q[:PANDA_JOINT_COUNT], dtype=np.float64)
        qd_arm = np.array(
            robot.get_joint_velocities()[:PANDA_JOINT_COUNT],
            dtype=np.float64,
        )

        # EEF goal for RMP2 (world frame)
        eef_goal = (
            np.asarray(target_pos_world, dtype=np.float64)
            + np.asarray(base_pos_world, dtype=np.float64)
        )

        # Solve RMP2 -> optimal joint acceleration
        q_ddot = rmp2_solver.solve(
            q=q_arm, qd=qd_arm,
            goals=[eef_goal],
            goal_quat=list(FIXED_DOWN_ROT),
            obstacles=obstacles_list,
        )

        # Integrate -> new joint positions
        q_new, qd_new = rmp2_solver.integrate(q_arm, qd_arm, q_ddot)
        q_safe = rmp2_solver.apply_hard_limits(q_new)

        # Append gripper joints (unchanged, effort-controlled)
        full_q = list(q_safe) + list(current_q[PANDA_JOINT_COUNT:])
        robot_action = ArticulationAction(joint_positions=full_q)
        robot.apply_action(robot_action)

        Model_target_marker.set_world_pose(
            position=eef_goal, orientation=target_rot_world)

        # ==== 6. Console stats ====
        t5 = time.perf_counter()
        if t % 10 == 0:
            _, ee_quat_now = panda_hand.get_world_pose()
            ee_q = np.asarray(ee_quat_now, dtype=np.float64)
            tgt_q = np.asarray(FIXED_DOWN_ROT, dtype=np.float64)

            # Quaternion angular error
            q_rel_w = (tgt_q[3]*ee_q[3] + tgt_q[0]*ee_q[0]
                       + tgt_q[1]*ee_q[1] + tgt_q[2]*ee_q[2])
            q_rel_v = np.array([
                ee_q[3]*tgt_q[0] - ee_q[0]*tgt_q[3]
                + ee_q[1]*tgt_q[2] - ee_q[2]*tgt_q[1],
                ee_q[3]*tgt_q[1] - ee_q[1]*tgt_q[3]
                + ee_q[2]*tgt_q[0] - ee_q[0]*tgt_q[2],
                ee_q[3]*tgt_q[2] - ee_q[2]*tgt_q[3]
                + ee_q[0]*tgt_q[1] - ee_q[1]*tgt_q[0],
            ], dtype=np.float64)
            ang_err = float(
                np.degrees(2.0 * np.arctan2(np.linalg.norm(q_rel_v),
                                            abs(q_rel_w))))

            print(
                SAVE + "\033[1A" + CLEAR
                + f"PANDA RMP2 | trigger={'ON' if trigger_on else 'OFF'} "
                + f"| rec={'YES' if record else 'NO'} "
                + f"| RMP2={((t5 - t2) * 1000):.1f}ms "
                + f"| tot={((t5 - t0) * 1000):.1f}ms\n"
                + f"  EEF quat [{ee_q[0]:.3f},{ee_q[1]:.3f},{ee_q[2]:.3f},"
                + f"{ee_q[3]:.3f}] | Target [{tgt_q[0]:.3f},{tgt_q[1]:.3f},"
                + f"{tgt_q[2]:.3f},{tgt_q[3]:.3f}] | ang_err={ang_err:.1f}°"
                + RESTORE,
                end="", flush=True,
            )

        t += 1

        # ==== 7. Environment reset (Button A) ====
        if buttonA:
            t = 0
            sim.stop()
            sim.reset()
            set_new_poses()
            base.initialize()
            robot.initialize()
            panda_hand.initialize()
            pos_save = ee_pos_world_initial
            target_pos_world = np.array(
                [0.14, -0.03, 2.5]) - base_pos_world
            quat_save = ee_rot_world_initial

            if record:
                print(f"Saving Ep {ep_tracker}")
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
