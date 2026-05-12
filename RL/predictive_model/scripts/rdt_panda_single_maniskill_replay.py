"""
rdt_panda_single.py — Isaac Sim Replay & RDT-1B Inference for Franka Panda (Single-Arm)
=============================================================================

OVERVIEW
--------
This script runs a physics simulation of a single Franka Panda robot arm in
NVIDIA Isaac Sim and replays pre-recorded demonstration trajectories from a
ManiSkill HDF5 dataset (PickCube-v1).  It captures multi-camera RGB frames,
records video episodes, and resets the scene (re-randomising object poses) at
each episode boundary.

The script also contains commented-out infrastructure for closed-loop control
using the Robotics Diffusion Transformer (RDT-1B) policy, including an
observation-window buffer, action interpolation, and joint-limit scaling.

MAIN WORKFLOW
-------------
1.  Start Isaac Sim (non-headless) and load the USD stage
    (`franka_simple_1.usd`).
2.  Initialise the robot articulation, four RGB cameras (wrist, mid, left,
    right), and a Lula-based inverse-kinematics solver.
3.  Load 1 000 demonstration trajectories from an HDF5 file
    (`trajectory.state.pd_ee_pose.physx_cpu.h5`), extracting joint positions
    (`eps`) and cube poses (`cube_pos`), and computing episode-length
    boundaries (`lengths`).
4.  Randomise object positions using the first trajectory's cube pose.
5.  Main simulation loop:
    a. Step the physics simulator.
    b. Read the next joint configuration from the trajectory dataset and
       apply it via `ArticulationAction`.
    c. Capture RGB frames from a randomly-chosen camera and write to video.
    d. When an episode boundary is reached, move completed `.avi` files to
       `/media/ucluser/PortableSSD/videos/cam_<N>/`, create fresh video
       writers, re-randomise object poses, and advance to the next episode.

COMMENTED-OUT RDT-1B INFERENCE PIPELINE
---------------------------------------
The block marked `#-----RDT-1B (start)` contains the full inference loop:
  • Builds a 2-step observation window (JPEG-compressed images + qpos +
    gripper state).
  • Every `rdt_chunk_size` (64) steps, calls `policy.step()` with proprio,
    images, and pre-computed language embeddings → returns a chunk of 64
    actions.
  • Optionally interpolates between consecutive actions
    (`interpolate_action`) to respect per-joint step limits.
  • Scales normalised model outputs [-1, 1] to real joint limits via
    `scale_action_to_joint_limits`.
  • Applies the resulting joint positions to the robot.

VR / MQTT SECTION
-----------------
MQTT client imports are present but the VR-telemetry path is not actively
used in the current main loop (it lives in `rdt_sim_vr_panda.py`).

KEY CONFIGURATION
-----------------
  chunk_size          : 64  (action chunk length for RDT-1B)
  ctrl_freq           : 25 Hz
  camera resolution   : 1280 × 720 @ 30 Hz
  cameras             : wrist (body), mid, left, right  (RGB only)
  state_dim           : 7  (6 joint angles + 1 gripper)
  joint_ranges        : Franka Panda limits (j1–j6 + gripper pair)
  pretrained model    : /home/ucluser/RoboticsDiffusionTransformer/checkpoints
  vision encoder      : SigLIP-so400m-patch14-384
  language embeddings : /home/ucluser/RoboticsDiffusionTransformer/outs/object_collection_2.pt

DIRECTORY STRUCTURE (runtime)
-----------------------------
  videos/cam_<N>/              — recorded RGB episodes
  dataset/states/              — (reserved for state logging)
  dataset/actions/             — (reserved for action logging)
  dataset/depths/cam_<N>/      — (reserved for depth stacks)
  /media/.../videos/cam_<N>/   — final video archive location

DEPENDENCIES
------------
  Isaac Sim (omni.isaac.*, isaacsim.*, pxr.*)
  PyTorch, NumPy, OpenCV, Pillow, SciPy, Matplotlib
  Gin Config, PyYAML, h5py, paho-mqtt
  RoboticsDiffusionTransformer (external repo on sys.path)

AUTHOR / PROVENANCE
-------------------
Derived from the Robotics Diffusion Transformer (RDT-1B) evaluation scripts,
adapted for Isaac Sim + Franka Panda single-arm replay.
"""

import argparse
import math
import pickle
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gin
import yaml
import collections
import json
import os
import torch
import time
import warnings
import h5py
from paho.mqtt  import client as mqtt_client
from threading  import Lock
from PIL        import Image as PImage

from scripts.utils import slerp, get_observations, get_image, create_video_writer
import sys
sys.path.insert(0, '/home/ucluser/RoboticsDiffusionTransformer')
sys.path.insert(0, '/home/ucluser/RoboticsDiffusionTransformer/scripts')
from models.rdt_runner import RDTRunner
from piper_model import create_model

warnings.filterwarnings(
    "ignore",
    message=".*has been deprecated.*",
)
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# from isaacsim import SimulationApp

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})  # or True

from isaacsim.core.api.simulation_context import SimulationContext

from isaacsim.core.prims import (
    SingleArticulation,
    SingleRigidPrim,
    SingleXFormPrim,
    
)

from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.objects import VisualCuboid

# from omni.isaac.core.simulation_context import SimulationContext
# from omni.isaac.core.articulations import Articulation
# from omni.isaac.core.utils.types import ArticulationAction
# from omni.isaac.core.prims import RigidPrim, XFormPrim
# from omni.isaac.core.objects import VisualCuboid
from omni.isaac.motion_generation import ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula import LulaKinematicsSolver
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from scipy.spatial.transform import Rotation as R


from pxr import UsdPhysics
"""import inspect
from omni.isaac.motion_generation.lula import LulaKinematicsSolver

print(inspect.signature(LulaKinematicsSolver))"""

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# ---------------------------
# |       Simulation        |
# ---------------------------
open_stage("/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description/franka_simple_1.usd")
sim = SimulationContext()
dt = sim.get_physics_dt()
sim.reset()
sim.play()
set_camera_view(
    eye=[2.0, 0.0, 4.0],      # camera position
    target=[0.0, 0.0, 2.5],   # look-at point
)
# Camera width height, frequency
width  = 1280
height = 720
frequency = 30
robot = SingleArticulation("/World/franka")
robot.initialize()

print(f"Robot dof names : {robot.dof_names}")
for name in robot.dof_names:
    print(robot.get_dof_index(name), name)
    
base                    = SingleRigidPrim("/World/franka/panda_link0")
table                   = SingleXFormPrim("/World/Xform_table")
franka_hand             = SingleRigidPrim("/World/franka/panda_hand")
# linkEndEffector     = SingleXFormPrim("/World/piper_description/piper_hand/linkEndEffector")
# cube_xform                = SingleXFormPrim("/World/Xform")
# bin_xform                 = SingleXFormPrim("/World/Xform_bin")
# teddy_bear_xform          = SingleXFormPrim("/World/Xform_teddy_bear")
dex_cube_xform          = SingleXFormPrim("/World/Xform_dex_cube")
dex_cube_1_xform        = SingleXFormPrim("/World/Xform_dex_cube_01")
rubik_xform             = SingleXFormPrim("/World/Xform_rubik")
rubik_1_xform           = SingleXFormPrim("/World/Xform_rubik_01")
nvidia_cube_xform       = SingleXFormPrim("/World/Xform_nvidia_cube")



# # cube_prim                 = SingleRigidPrim("/World/Xform")
# bin_prim                  = SingleRigidPrim("/World/Xform_bin/small_KLT")
# # teddy_bear_prim           = SingleRigidPrim("/World/Xform_teddy_bear")
# dex_cube_prim             = SingleRigidPrim("/World/Xform_dex_cube/dex_cube_instanceable")
# rubik_prim                = SingleRigidPrim("/World/Xform_rubik/rubiks_cube")
# nvidia_cube_prim          = SingleRigidPrim("/World/Xform_nvidia_cube/nvidia_cube")


# # mug                 = SingleXFormPrim("/World/Xform_mug")
objects             = {#"cube":cube_xform, 
#                     #    "teddy_bear": teddy_bear_xform, 
                       "dex_cube": dex_cube_xform,
                       "dex_cube_1": dex_cube_1_xform, 
                       "rubik":rubik_xform, 
                       "rubik_1":rubik_1_xform, 
                       "nvidia_cube": nvidia_cube_xform,
                    #    "bin": bin_xform
                       }


# objects_prim        = {"dex_cube": dex_cube_prim, 
#                        "rubik":rubik_prim, 
#                        "nvidia_cube": nvidia_cube_prim,
#                        "bin": bin_prim}
prim_paths = ["/World/Xform_bin/small_KLT", "/World/Xform_dex_cube/dex_cube_instanceable" , "/World/Xform_rubik/rubiks_cube", "/World/Xform_nvidia_cube/nvidia_cube"]


body_rgb_cam        = Camera(
                    prim_path="/World/franka/panda_hand/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
                    frequency=frequency,
                    resolution=(width, height),)
mid_rgb_cam        = Camera(
                    prim_path="/World/Realsense_mid/RSD455/Camera_OmniVision_OV9782_Color",
                    frequency=frequency,
                    resolution=(width, height),)
left_rgb_cam        = Camera(
                    prim_path="/World/Realsense_left/RSD455/Camera_OmniVision_OV9782_Color",
                    frequency=frequency,
                    resolution=(width, height),)
right_rgb_cam        = Camera(
                    prim_path="/World/Realsense_right/RSD455/Camera_OmniVision_OV9782_Color",
                    frequency=frequency,
                    resolution=(width, height),)


rgb_cams            = [body_rgb_cam, mid_rgb_cam, left_rgb_cam, right_rgb_cam]
depth_cams          = None
# depth_cams          = [body_depth_cam, mid_depth_cam, left_depth_cam]

base.initialize()
franka_hand.initialize()
# linkEndEffector.initialize()
body_rgb_cam.initialize()
mid_rgb_cam.initialize()
left_rgb_cam.initialize()
right_rgb_cam.initialize()

robot_prim = get_prim_at_path("/World/franka")
stage = get_prim_at_path("/World/franka").GetStage()

for prim in stage.Traverse():
    if not prim.GetPath().HasPrefix(robot_prim.GetPath()):
        continue

    # Revolute / prismatic joints only
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")

        drive.GetStiffnessAttr().Set(1e4)
        drive.GetDampingAttr().Set(1e2)
        

# ---------------------------
# | IK Target Visualization |
# ---------------------------
VR_target_marker       = VisualCuboid(prim_path="/World/IK_VR_Target",
    position=[0.0, 0.0, 2.5],
    scale=[0.03, 0.03, 0.03],
    color=np.array([1.0, 0.0, 0.0])  # red
)
Model_target_marker       = VisualCuboid(prim_path="/World/IK_Model_Target",
    position=[0.0, 0.0, 2.6],
    scale=[0.03, 0.03, 0.03],
    color=np.array([0.0, 1.0, 0.0])  # red
)

# ---------------------------
# |        Lula IK          |
# ---------------------------
ik_solver = LulaKinematicsSolver(
    robot_description_path="/home/ucluser/isaacgym/assets/urdf/piper_description/config/franka_robot.yaml",
    urdf_path="/home/ucluser/isaacgym/assets/urdf/franka_description/robots/franka_panda.urdf"
)
ik_solver.set_default_position_tolerance(0.02)
ik_solver.set_default_orientation_tolerance(0.02)

# print(f"#-------------- get_default_position_tolerance() = {ik_solver.get_default_position_tolerance()}")
# print(f"#-------------- get_default_orientation_tolerance() = {ik_solver.get_default_orientation_tolerance()}")

#-------------- get_default_position_tolerance() = 0.001
#-------------- get_default_orientation_tolerance() = 0.010000041667134873

kin_solver = ArticulationKinematicsSolver(
    robot_articulation=robot,
    kinematics_solver=ik_solver,
    end_effector_frame_name="panda_hand" #piper_hand 
)

base_pos_world, base_rot_world = base.get_world_pose()
R_base = R.from_quat(base_rot_world)

# small reachable offset
target_pos = np.array([0.2, 0.0, 0.25])
VR_target_marker.set_world_pose(position=target_pos)
Model_target_marker.set_world_pose(position=target_pos)

# target_rot = euler_angles_to_quat(np.array([0.5 * math.pi, controller_obj[1], 0]))

# grip_action = ArticulationAction(joint_positions=robot.get_joint_positions())

# ---------------------------
# |         RDT-1B          |
# ---------------------------


args = {
    'max_publish_step': 300000,
    'chunk_size': 64,
    'arm_steps_length': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
    'use_actions_interpolation': False,
    'use_depth_image': False,
    'disable_puppet_arm': False,
    'config_path': "/home/ucluser/RoboticsDiffusionTransformer/configs/base.yaml",
    'pretrained_model_name_or_path': "/home/ucluser/RoboticsDiffusionTransformer/checkpoints_large",
    'lang_embeddings_path': "/home/ucluser/RoboticsDiffusionTransformer/outs/object_collection_2.pt",
    'ctrl_freq': 25,
    'use_robot_base' : False
}
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

args = Args(**args)
print("args setup completed")
CAMERA_NAMES = ['exterior-cam', 'right-wrist-cam', 'left-wrist-cam']

observation_window = None

lang_embeddings = None


def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config
    
    # pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "/home/ucluser/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384"
    # pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config, 
        dtype=torch.bfloat16,
        pretrained="/home/ucluser/RoboticsDiffusionTransformer/checkpoints",
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )
    return model

def get_config(args):
    config = {
        'episode_len': 300000,
        'state_dim': 7,  # position (3) + orientation (6)
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config

# Update the observation window buffer
def update_observation_window(config, imgs, joint_state, EEF_position, grippers):
    # JPEG transformation
    # Align with training
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB
    global observation_window
    if observation_window is None:
        observation_window = collections.deque(maxlen=2)

        # Append the first dummy image
        observation_window.append(
            {
                'qpos': None,
                'EEF_position': None,
                'gripper': None,
                'images':
                    {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                        config["camera_names"][2]: None,
                    },
            }
        )

    for i in range(len(imgs)):
        imgs[i] = imgs[i][:,:, :3]
        assert imgs[i].shape[2] == 3
        imgs[i] = jpeg_mapping(imgs[i])

    qpos = joint_state
    qpos = torch.from_numpy(qpos).float().cuda()
    if EEF_position is not None:
        EEF_pos = EEF_position
        EEF_pos = torch.from_numpy(EEF_pos).float().cuda()
        
    gripper = grippers
    gripper = torch.from_numpy(gripper).float().cuda()
     
    observation_window.append(
        {
            'qpos': qpos,
            'EEF_position': None,
            'gripper': gripper,
            'images':
                {
                    config["camera_names"][0]: imgs[0],
                    config["camera_names"][1]: imgs[1],
                    config["camera_names"][2]: imgs[2],
                },
        }
    )
    
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


config = get_config(args)
# print("Model is being created")
policy = make_policy(args)
# policy = policy.cuda()
# print("policy is completely made")

lang_dict = torch.load(args.lang_embeddings_path, weights_only=False)
print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
lang_embeddings = lang_dict["embeddings"].cuda()

max_publish_step = config['episode_len']
rdt_chunk_size = config['chunk_size']

print(f"rdt_chunk_size = {rdt_chunk_size}")
left_dof_count = robot.get_joint_positions().shape[0]
print(f"left_dof_count = {left_dof_count}")

pre_action = np.zeros(config['state_dim'])
pre_action[:7] = np.array(
    [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.00133514404296875]
)
action = None
action_buffer = np.zeros([rdt_chunk_size, config['state_dim']])

joint_ranges = np.array([
    [-2.6179, 2.6179],    # joint1
    [0, 3.14],            # joint2
    [-2.967, 0],          # joint3
    [-1.745, 1.745],      # joint4
    [-1.22, 1.22],        # joint5
    [-2.09439, 2.09439],  # joint6
    [0, 0.04],            # joint7 (gripper)
    [-0.04, 0]            # joint8 (gripper)
])

joint_mins = joint_ranges[:, 0]
joint_ranges_vals = (joint_ranges[:, 1] - joint_ranges[:, 0])


# ---------------------------
# |     Utils functions     |
# ---------------------------

def timed_inference(policy, obs, img):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.no_grad():
        action, rgb_reconstructed = policy(obs, img)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    return action, (t1 - t0)

def count_files_in_directory(directory_path: str) -> int:
    """
    Counts the number of files in a directory (non-recursive).
    
    Args:
        directory_path (str): Path to the directory to scan.
    
    Returns:
        int: Number of files in the directory.
    
    Example:
        num_files = count_files_in_directory('/path/to/directory')
    """
    try:
        return len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
    except Exception as e:
        print(f"Error counting files in {directory_path}: {e}")
        return 0

def image_capture(writers: list | None, rgb_cam: int, width: int, height: int, record: bool) -> list | None:
    """
    Captures and processes RGBA and depth images from multiple camera pairs.
    
    This function retrieves raw image data from multiple RGB and depth cameras,
    processes the data by reshaping and normalizing, and combines them into RGBD images.
    Optionally records the processed RGB frames and depth data to disk.
    
    Args:
        writers (list): List of video writers for recording RGB frames, one per camera. [ cam_1 writer, cam_2 writer, ..]
        depth_stacks (list): List to accumulate depth image frames for later processing. [ [cam_1 frames], [cam_2 frames], ..]
        rgb_cams (list): List of RGB camera objects with get_rgba() method.
        depth_cams (list): List of depth camera objects with get_depth() method.
        width (int): Width of the camera images.
        height (int): Height of the camera images (note: typo in parameter name).
        record (bool): If True, writes RGB frames to video files and appends depth to depth_stacks.
    
    Returns:
        RGBD_image: List of RGBD images per camera (numpy arrays with shape [height, width, 4]) concatenating
              RGB (3 channels) and depth (1 channel), or None if image capture fails.
    
    Note:
        - Depth values are clipped to [0.0, 5.0] range
        - Returns None if any camera fails to capture image data
        - Requires equal number of RGB and depth cameras
    """

    for i in range(len(rgb_cams)):
        img     = rgb_cams[i].get_rgba()
        if len(img) == 0:
            return None
        color_image = img.copy()
        # Reshape to (height, width, 4) - note: height comes first!
        color_image = color_image.reshape((height, width, 4))
        rgb_image   = color_image[:, :, :3]
        bgr_image   = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        writers[i].write(bgr_image)        

def random_pose(is_tray: bool):
    z = 2.8 if is_tray else 2.45
    if is_tray:
        regions = [
            ((0.1, 0.5), (-0.6, -0.3)),
            ((0.1, 0.5), (0.3, 0.6)),
            ((0.5, 0.6), (-0.6, 0.6))
        ]
    else:
        regions = [
            ((0.0, 0.5), (-1.0, -0.3)),
            ((0.0, 0.5), (0.3, 1.0)),
            ((0.5, 1.15),(-1.0, 1.0))
        ]

    # Compute areas
    areas = [(x1-x0)*(y1-y0) for (x0,x1),(y0,y1) in regions]
    p = np.array(areas) / sum(areas)

    region = regions[np.random.choice(len(regions), p=p)]
    (x0,x1),(y0,y1) = region

    x = np.random.uniform(x0, x1)
    y = np.random.uniform(y0, y1)
    
    yaw = np.random.uniform(-np.pi, np.pi)

    # q = R.from_euler("z", yaw).as_quat()
    # q1 = R.   ("x", 0.0).as_quat()
    # total_q = q.inv() * q1


    q = [0,0,0,1]
    return np.array([x,y,z]),q

def set_new_poses():   
    for object in objects:
        if object == "bin": 
            pos, q = random_pose(is_tray=True)
        else:
            pos, q = random_pose(is_tray=False)
            
        objects[object].set_world_pose(
            position=pos,
            orientation=q
        )

def scale_action_to_joint_limits(action, joint_ranges):
    """
    Maps normalized action values from RDT-1B model to actual joint limits.
    
    The RDT-1B model typically outputs actions in normalized range [-1, 1].
    This function scales them to the actual joint position limits.
    
    Args:
        action: Array of shape [8,] or [n, 8] containing normalized action values in range [-1, 1]
        joint_ranges: Array of shape [8, 2] with [min, max] limits for each joint
    
    Returns:
        Scaled action array with same shape as input, mapped to joint limits
    
    Example:
        >>> action = np.array([0.5, -0.3, 0.1, 0.0, -0.5, 0.8, 1.0, -1.0])
        >>> scaled_action = scale_action_to_joint_limits(action, joint_ranges)
    """
    is_batch = len(action.shape) > 1
    
    if not is_batch:
        action = action[np.newaxis, :]
    
    batch_size = action.shape[0]
    num_joints = action.shape[1]
    
    scaled_actions = np.zeros_like(action)
    clamped_actions = np.zeros_like(action)
    
    for i in range(num_joints):
        joint_min = joint_ranges[i, 0]
        joint_max = joint_ranges[i, 1]
        joint_range = joint_max - joint_min
        
        # Map from [-1, 1] to [joint_min, joint_max]
        scaled_actions[:, i] = joint_min + (action[:, i] + 1) / 2 * joint_range
        clamped_actions[:, i] = np.clip(scaled_actions[:, i], joint_min, joint_max)
    
    if not is_batch:
        clamped_actions = clamped_actions[0]
    
    return clamped_actions


def prnt(obj, name: str | None, attr: str | None):
    if attr is None:
        if isinstance(obj, list):
            print(f"{name} : {item:.2f}" for item in obj)
        elif isinstance(obj, set):
            print(f"{name} : {obj}")
        elif isinstance(obj, np.ndarray):
            print(f"{name} of {type(obj)} : {obj}")
        elif isinstance(obj, int) or isinstance(obj, float):
            print(f"{obj=}")
        else:
            print(f"{name}: {obj}")
    elif attr == "type":
        print(f"Type of {name} is {type(obj)}")
    elif attr == "size":
        if isinstance(obj, list) or isinstance(obj, set):
            print(f"Size of {name}: {len(obj)}")
        elif isinstance(obj, np.ndarray):
            print(f"Shape of {name} : {obj.shape}")
        elif isinstance(obj, int) or isinstance(obj, float):
            print(f"{obj=} has no shape")

def random_new_pose(new_pos):
    obj_list = list(objects.values())
    np.random.shuffle(obj_list)
    
    # prnt(obj_list, "obj_last", None)
    obj_list[0].set_world_pose(position= new_pos[0:3] + [0.825, -0.135, base_pos_world[2]],
                                        orientation= new_pos[3:7])     
    obj_list[1].set_world_pose(position=[0.5, -0.2, base_pos_world[2]],
                                        orientation= new_pos[3:7])   
    obj_list[2].set_world_pose(position=[0.4, -0.3, base_pos_world[2]],
                                        orientation= new_pos[3:7])   
    obj_list[3].set_world_pose(position=[0.3, -0.25, base_pos_world[2]],
                                        orientation= new_pos[3:7])   
    obj_list[4].set_world_pose(position=[0.2, -0.35, base_pos_world[2]],
                                        orientation= new_pos[3:7])   
        

with h5py.File("/home/ucluser/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_pose.physx_cpu.h5", "r") as f:
    eps = np.zeros((1,9))
    cube_pos = np.zeros((1,7))
    terminated = np.zeros(1)
    success = np.zeros(1)
    lengths = [0]
    
    for i in range(1000):
        data        = f[f"traj_{i}/env_states/articulations/panda"][:]
        length      = data.shape[0]
        pos         = f[f"traj_{i}/env_states/actors/cube"][0]
        pos         = pos[np.newaxis, :]
        # term  = f[f"traj_{i}/terminated"][:]
        lengths.append(length + lengths[-1])
        # succ  = f[f"traj_{i}/success"][:]
        # prnt(succ, "suc", None)
        eps = np.concatenate((eps, data[:, 13:22]), axis=0)
        cube_pos = np.concatenate((cube_pos, pos[:, 0:7]), axis=0)
        # terminated = np.concatenate((terminated, term), axis=None)
        # success = np.concatenate((success, succ), axis=None)
        # prnt(terminated, "terminated", "size")
    cube_pos = np.delete(cube_pos, 0, 0)
    eps = np.delete(eps, 0, 0)
    
    all_joints = eps
# prnt(cube_pos, "cube_pos", "size" )
print(lengths)
lengths.pop(0)
# prnt(lengths, "lengths", "size")
lengths = set(lengths)  
def next_step(t):
    return all_joints[t]
# ---------------------------
# |        Main loop        |
# ---------------------------
try:
    os.makedirs(f"dataset/states" ,  exist_ok=True)
    os.makedirs(f"dataset/actions" , exist_ok=True)
    for cam in range(len(rgb_cams)):
        os.makedirs(f"/media/ucluser/PortableSSD/videos/cam_{cam}", exist_ok=True)
        os.makedirs(f"dataset/depths/cam_{cam}" ,                   exist_ok=True)
except FileExistsError:
    pass

ep_tracker = count_files_in_directory("videos/cam_0")

quat_save   = euler_angles_to_quat(np.array([0.5 * math.pi, 0, 0]))
quat_start  = None
pos_save    = np.array([0.14,-0.03, 2.5]) - base_pos_world

# pos_marker_save    = np.array([0.2,0.0,0.2])
t  = 0
t1 = 0
t2 = 0
t3 = 0
t4 = 0
t5 = 0
all_actions = None
target_action = None
record = False
writers = []
depth_stacks = []
ts = 0
term_counter = 0
new_pos = cube_pos[term_counter]
# prnt(new_pos, "new_pose", None)
random_new_pose(new_pos)

for cam in range(len(rgb_cams)):
    writers.append(create_video_writer(f"rgb_{cam}", frequency, width, height))

with torch.inference_mode():
    while simulation_app.is_running():
        
        sim.step(render=True)
        # t0 = time.perf_counter()
                        
        # joint_states   = robot.get_joint_positions()
        
        # print(len(joint_states))
        # print(f"Left  states: {', '.join(f'{joint * 180 / math.pi:.4f}' for joint in joint_states)}")

        
        
        #-----RDT-1B (start)--------------#
        """
        t2 = time.perf_counter()
        if captured_img:
            

            grippers        = np.array([left_joint_states[6]])
            # print(f"Dimension joint_states : {joint_states.shape} ")
            print(f"Dimension grippers : {grippers.shape} ")
            
            update_observation_window(config=config,imgs=captured_img, joint_state=left_joint_states[0:6], EEF_position=None, grippers=grippers)
            if t % rdt_chunk_size == 0:
                # Start inference
                image_arrs = [
                    observation_window[-2]['images'][config['camera_names'][0]],
                    observation_window[-2]['images'][config['camera_names'][1]],
                    observation_window[-2]['images'][config['camera_names'][2]],
                    
                    observation_window[-1]['images'][config['camera_names'][0]],
                    observation_window[-1]['images'][config['camera_names'][1]],
                    observation_window[-1]['images'][config['camera_names'][2]],
                ]
                images = [PImage.fromarray(arr) if arr is not None else None
                        for arr in image_arrs]
                
                # get last qpos in shape [8, ] and EEF position in shape [3, ]  
                proprio = torch.cat([observation_window[-1]['qpos'], observation_window[-1]['gripper']], dim=0)
                proprio = proprio.unsqueeze(0)
                actions = policy.step(
                    proprio=proprio,
                    images=images,
                    text_embeds=lang_embeddings 
                ).squeeze(0).cpu().numpy()
                
                action_buffer = actions.copy()
            t4 = time.perf_counter()
            raw_action = action_buffer[t % rdt_chunk_size]
            action = raw_action
            if args.use_actions_interpolation:
                # print(f"Time {t}, pre {pre_action}, act {action}")
                interp_actions = interpolate_action(args, pre_action, action)
                # interp_actions = interpolate_joints(args, pre_action, action)
                # interp_actions = interpolate_EEF(args, pre_action[:3], action[:3])
            else:
                interp_actions = action[np.newaxis, :]
            # Execute the interpolated actions one by one
            for act in interp_actions:
                print(f"output action (deg): {act * 180 / math.pi}")
                print(f"output action (rad): {act}")
                # print(f"type of target pos : {type(target_pos)}")
                left_actions    = act[0:6]

                left_grip       = act[6]


                # print(f"target action shape: {act.shape}")
                # print(f"left_actions type = {type(left_actions)}")
                # print(f"left_actions = {left_actions}")
                # print(f"left_grip = {left_grip}")
                # print(f"left_grip shape = {left_grip.shape}")
                # print(f"left_grip type = {type(left_grip)}")

                scaled_left_action  = scale_action_to_joint_limits(np.concatenate((left_actions,  np.array([left_grip])), axis=0),  joint_ranges)
                scaled_left_action  = np.concatenate((scaled_left_action,  np.array([-1 * scaled_left_action[-1]])), axis=0)
            
                # robot_pos           = np.concatenate((scaled_left_action, np.array([-1 * scaled_left_action[-1]]), scaled_right_action, np.array([-1 * scaled_right_action[-1]])) , axis=0)
                
                print(f"scaled_left_action pos : {scaled_left_action  * 180 / math.pi}")

    
                # left_action   = ArticulationAction(joint_positions=np.concatenate((left_actions, np.array([left_grip, left_grip])), axis=0))
             
                print(f"reordered robot position : {scaled_left_action  * 180 / math.pi}")
                # robot_position      = np.zeros(16, dtype=float)
                # robot_position[5]   = -0.25039473   * math.pi / 180
                # robot_position[10]  =  20.04485302   * math.pi / 180
                # robot_position[11]  = -5.81892053  * math.pi / 180
                
                # left_position = np.concatenate((left_actions, np.array([left_grip, -1 * left_grip])), axis=0)
        

                robot_action   = ArticulationAction(joint_positions=scaled_left_action)
                
                left_robot.apply_action(robot_action)
            
            pre_action = action.copy()
            
            t+=1
        """
        ts += 1
        joint = next_step(ts)
        
        joint = np.array(joint)
        # print(f"shape = {joint.shape}")
        robot_action   = ArticulationAction(joint_positions=joint)
                
        robot.apply_action(robot_action)
        image_capture(writers=writers, rgb_cam= np.random.choice(4), width=width, height=height, record=False)
        
        if ts in lengths:
            print(f"Saving successful Ep {ep_tracker}")
            for cam in range(len(rgb_cams)):
                shutil.move(f"rgb_{cam}.avi", f"/media/ucluser/PortableSSD/videos/cam_{cam}/rgb_ep_{ep_tracker}.avi")
                writers[cam]= create_video_writer(f"rgb_{cam}", frequency, width, height)
            ep_tracker += 1
            new_pos = cube_pos[term_counter+1]
            # print(terminated[ts])
            # prnt(new_pos, "new_pos", None)
            random_new_pose(new_pos)
            
            term_counter += 1
        # if t % 2 == 0:
        #     ts+=1
        #     prnt(joint_states, "joint_states", None)
        #     prnt(joint, "joint_action", None)
        # t+=1
        
simulation_app.close()
