"""
rdt_panda_single.py — RDT-1B Closed-Loop Inference for Franka Panda in Isaac Sim
===================================================================================

OVERVIEW
--------
Runs a physics simulation of a single Franka Panda robot arm in NVIDIA Isaac
Sim and executes closed-loop control using the Robotics Diffusion Transformer
(RDT-1B) policy.  The model receives multi-camera RGB observations and
proprioceptive state (joint angles + gripper), then outputs action chunks that
are scaled to real joint limits and applied to the simulated robot.

WORKFLOW
--------
1.  Start Isaac Sim (non-headless) and load the USD stage
    (`franka_simple_1.usd`).
2.  Initialise the robot articulation, four RGB cameras (wrist, mid, left,
    right), and a Lula-based inverse-kinematics solver.
3.  Load the RDT-1B policy with a SigLIP vision encoder and pre-computed
    language embeddings.
4.  Main simulation loop:
    a.  Capture RGB frames from all cameras.
    b.  Update a 2-step observation window (JPEG-compressed images + qpos +
        gripper state).
    c.  Every `chunk_size` (64) steps, run `policy.step()` → returns a chunk
        of 64 actions shaped [64, state_dim].
    d.  Optionally interpolate between consecutive actions to respect
        per-joint step limits (`interpolate_action`).
    e.  Scale normalised model outputs [-1, 1] to real joint limits via
        `scale_action_to_joint_limits`.
    f.  Apply the resulting joint positions to the robot via
        `ArticulationAction`.
    g.  Record RGB video frames per camera.

KEY CONFIGURATION
-----------------
  chunk_size          : 64  (action chunk length for RDT-1B)
  ctrl_freq           : 25 Hz
  camera resolution   : 1280 x 720 @ 30 Hz
  cameras             : wrist (body), mid, left, right  (RGB only)
  state_dim           : 8  (7 joint angles + 1 gripper)
  joint_ranges        : Franka Panda limits (j1-j7 + gripper pair)
  pretrained model    : /home/ucluser/RoboticsDiffusionTransformer/checkpoints
  vision encoder      : SigLIP-so400m-patch14-384
  language embeddings : /home/ucluser/RoboticsDiffusionTransformer/outs/object_collection_2.pt

DIRECTORY STRUCTURE (runtime)
-----------------------------
  videos/cam_<N>/              — recorded RGB episodes
  dataset/states/              — (reserved for state logging)
  dataset/actions/             — (reserved for action logging)
  dataset/depths/cam_<N>/      — (reserved for depth stacks)

DEPENDENCIES
------------
  Isaac Sim (omni.isaac.*, isaacsim.*, pxr.*)
  PyTorch, NumPy, OpenCV, Pillow, SciPy, Matplotlib
  Gin Config, PyYAML
  RoboticsDiffusionTransformer (external repo on sys.path)
"""

import argparse
import math
import pickle
import shutil
import numpy as np
import cv2
import gin
import yaml
import collections
import os
import torch
import time
import warnings
from PIL import Image as PImage

import sys
sys.path.insert(0, '/home/ucluser/VRWIT/RL/predictive_model')
# sys.path.insert(0, '/home/ucluser/RoboticsDiffusionTransformer/scripts')
from scripts.utils import slerp, get_observations, get_image, create_video_writer

sys.path.insert(0, '/home/ucluser/RoboticsDiffusionTransformer')
from models.rdt_runner import RDTRunner
# from piper_model import create_model
from scripts.maniskill_model import create_model

warnings.filterwarnings("ignore", message=".*has been deprecated.*")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False

# ---------------------------------------------------------------------------
# Simulation setup
# ---------------------------------------------------------------------------
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
from scipy.spatial.transform import Rotation as R
from pxr import UsdPhysics

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# ---------------------------------------------------------------------------
# Simulation setup
# ---------------------------------------------------------------------------
open_stage("/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description/franka_simple_1.usd")
sim = SimulationContext()
dt = sim.get_physics_dt()
sim.reset()
sim.play()
set_camera_view(
    eye=[2.0, 0.0, 4.0],
    target=[0.0, 0.0, 2.5],
)

width     = 720
height    = 480
frequency = 20

robot = SingleArticulation("/World/franka")
robot.initialize()
print(f"Robot dof names : {robot.dof_names}")
for name in robot.dof_names:
    print(robot.get_dof_index(name), name)

base          = SingleRigidPrim("/World/franka/panda_link0")
table         = SingleXFormPrim("/World/Xform_table")
franka_hand   = SingleRigidPrim("/World/franka/panda_hand")

# Manipulatable objects in the scene
dex_cube_xform    = SingleXFormPrim("/World/Xform_dex_cube")
dex_cube_1_xform  = SingleXFormPrim("/World/Xform_dex_cube_01")
rubik_xform       = SingleXFormPrim("/World/Xform_rubik")
rubik_1_xform     = SingleXFormPrim("/World/Xform_rubik_01")
nvidia_cube_xform = SingleXFormPrim("/World/Xform_nvidia_cube")

objects = {
    "dex_cube": dex_cube_xform,
    "dex_cube_1": dex_cube_1_xform,
    "rubik": rubik_xform,
    "rubik_1": rubik_1_xform,
    "nvidia_cube": nvidia_cube_xform,
}

# Cameras
body_rgb_cam  = Camera(prim_path="/World/franka/panda_hand/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
                       frequency=frequency, resolution=(width, height))
mid_rgb_cam   = Camera(prim_path="/World/Realsense_mid/RSD455/Camera_OmniVision_OV9782_Color",
                       frequency=frequency, resolution=(width, height))
left_rgb_cam  = Camera(prim_path="/World/Realsense_left/RSD455/Camera_OmniVision_OV9782_Color",
                       frequency=frequency, resolution=(width, height))
right_rgb_cam = Camera(prim_path="/World/Realsense_right/RSD455/Camera_OmniVision_OV9782_Color",
                       frequency=frequency, resolution=(width, height))

rgb_cams = [mid_rgb_cam, body_rgb_cam, right_rgb_cam]

base.initialize()
franka_hand.initialize()
body_rgb_cam.initialize()
mid_rgb_cam.initialize()
# left_rgb_cam.initialize()
right_rgb_cam.initialize()

# Joint drive parameters
robot_prim = get_prim_at_path("/World/franka")
stage = robot_prim.GetStage()
for prim in stage.Traverse():
    if not prim.GetPath().HasPrefix(robot_prim.GetPath()):
        continue
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.GetStiffnessAttr().Set(1e4)
        drive.GetDampingAttr().Set(1e2)

# IK target visualisation markers
VR_target_marker    = VisualCuboid(prim_path="/World/IK_VR_Target",
    position=[0.0, 0.0, 2.5], scale=[0.03, 0.03, 0.03], color=np.array([1.0, 0.0, 0.0]))
Model_target_marker = VisualCuboid(prim_path="/World/IK_Model_Target",
    position=[0.0, 0.0, 2.6], scale=[0.03, 0.03, 0.03], color=np.array([0.0, 1.0, 0.0]))

# Lula IK solver
ik_solver = LulaKinematicsSolver(
    robot_description_path="/home/ucluser/isaacgym/assets/urdf/piper_description/config/franka_robot.yaml",
    urdf_path="/home/ucluser/isaacgym/assets/urdf/franka_description/robots/franka_panda.urdf",
)
ik_solver.set_default_position_tolerance(0.02)
ik_solver.set_default_orientation_tolerance(0.02)

kin_solver = ArticulationKinematicsSolver(
    robot_articulation=robot,
    kinematics_solver=ik_solver,
    end_effector_frame_name="panda_hand",
)

base_pos_world, base_rot_world = base.get_world_pose()

target_pos = np.array([0.2, 0.0, 0.25])
VR_target_marker.set_world_pose(position=target_pos)
Model_target_marker.set_world_pose(position=target_pos)

# ---------------------------------------------------------------------------
# RDT-1B policy configuration
# ---------------------------------------------------------------------------
args = {
    'max_publish_step': 300000,
    'chunk_size': 32,
    'arm_steps_length': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
    'use_actions_interpolation': False,
    'use_depth_image': False,
    'disable_puppet_arm': False,
    'config_path': "/home/ucluser/RoboticsDiffusionTransformer/configs/base.yaml",
    'pretrained_model_name_or_path': "/home/ucluser/RoboticsDiffusionTransformer/checkpoint-170b-30000",
    'pretrained_vision_encoder_name_or_path' : "/home/ucluser/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384",
    'lang_embeddings_path': "/home/ucluser/RoboticsDiffusionTransformer/data/empty_lang_embed.pt",
    'ctrl_freq': 25,
    'use_robot_base': False,
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
    """Create and return the RDT-1B policy model."""
    # Free up CUDA memory before loading the model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA memory before model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config

    
    model = create_model(
        args=args.config,
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_vision_encoder_name_or_path=args.pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )
    return model


def get_config(args):
    """Return runtime config dict for the RDT policy."""
    return {
        'episode_len': 300000,
        'state_dim': 8,   # 7 joint angles + 1 gripper (matches model output)
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }


def update_observation_window(config, imgs, joint_state, grippers):
    """Append a timestep to the rolling observation window (maxlen=2).

    Images are JPEG-compressed to match training-time preprocessing.
    """
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    global observation_window
    if observation_window is None:
        observation_window = collections.deque(maxlen=2)
        # Dummy first entry
        observation_window.append({
            'qpos': None,
            'gripper': None,
            'images': {cam: None for cam in config["camera_names"]},
        })

    for i in range(len(imgs)):
        imgs[i] = imgs[i][:, :, :3]
        assert imgs[i].shape[2] == 3
        imgs[i] = jpeg_mapping(imgs[i])

    qpos = torch.from_numpy(joint_state).float().cuda()
    gripper = torch.from_numpy(grippers).float().cuda()

    observation_window.append({
        'qpos': qpos,
        'gripper': gripper,
        'images': {
            config["camera_names"][0]: imgs[0],
            config["camera_names"][1]: imgs[1],
            config["camera_names"][2]: imgs[2],
        },
    })


def interpolate_action(args, prev_action, cur_action):
    """Linearly interpolate between two actions if the jump is too large."""
    steps = np.concatenate(
        (np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0
    )
    diff = np.abs(cur_action - prev_action)
    step = int(np.ceil(diff / steps).astype(int).max())
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def scale_action_to_joint_limits(action, joint_ranges):
    """Map normalised action values [-1, 1] to actual joint limits."""
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
        scaled_actions[:, i] = joint_min + (action[:, i] + 1) / 2 * joint_range
        clamped_actions[:, i] = np.clip(scaled_actions[:, i], joint_min, joint_max)

    return clamped_actions[0] if not is_batch else clamped_actions


def count_files_in_directory(directory_path: str) -> int:
    """Count files in a directory (non-recursive)."""
    try:
        return len([
            f for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ])
    except Exception as e:
        print(f"Error counting files in {directory_path}: {e}")
        return 0


def image_capture(writers, width, height):
    """Capture RGB frames from all cameras and write to video writers.

    Returns a list of RGB images (one per camera) or None if capture fails.
    """
    rgb_images = []
    for i in range(len(rgb_cams)):
        img = rgb_cams[i].get_rgba()
        if len(img) == 0:
            return None
        color_image = img.copy().reshape((height, width, 4))
        rgb_image = color_image[:, :, :3]
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        writers[i].write(bgr_image)
        rgb_images.append(rgb_image)
    return rgb_images


def random_pose(is_tray=False):
    """Sample a random reachable pose for scene object randomisation."""
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
    """Randomise positions of all manipulatable objects in the scene."""
    for obj_name, obj_prim in objects.items():
        pos, q = random_pose(is_tray=(obj_name == "bin"))
        obj_prim.set_world_pose(position=pos, orientation=q)


# ---------------------------------------------------------------------------
# Initialise policy and language embeddings
# ---------------------------------------------------------------------------
config = get_config(args)
policy = make_policy(args)

lang_dict = torch.load(args.lang_embeddings_path, weights_only=False)
print(f'Running with instruction: "{lang_dict["instruction"]}" from "{lang_dict["name"]}"')
lang_embeddings = lang_dict["embeddings"].cuda()

max_publish_step = config['episode_len']
rdt_chunk_size = config['chunk_size']
print(f"rdt_chunk_size = {rdt_chunk_size}")

left_dof_count = robot.get_joint_positions().shape[0]
print(f"left_dof_count = {left_dof_count}")

pre_action = np.zeros(config['state_dim'])
action = None
action_buffer = np.zeros([rdt_chunk_size, config['state_dim']])

joint_ranges = np.array([
    [-2.8973,  2.8973],   # joint1
    [-1.7628,  1.7628],   # joint2
    [-2.8973,  2.8973],   # joint3
    [-3.0718,  -0.0698],   # joint4
    [-2.8973,  2.8973],   # joint5
    [-0.0175,  3.7525],   # joint6
    [-2.8973,  2.8973],    # joint7
    [ 0.0,     0.04],     # joint8 (gripper left)
    [-0.04,    0.0],      # joint9 (gripper right)
])

# ---------------------------------------------------------------------------
# Main simulation loop — RDT-1B closed-loop inference
# ---------------------------------------------------------------------------
try:
    os.makedirs("dataset/states", exist_ok=True)
    os.makedirs("dataset/actions", exist_ok=True)
    for cam_idx in range(len(rgb_cams)):
        os.makedirs(f"/media/ucluser/PortableSSD/videos/cam_{cam_idx}", exist_ok=True)
        os.makedirs(f"dataset/depths/cam_{cam_idx}", exist_ok=True)
except FileExistsError:
    pass

ep_tracker = count_files_in_directory("/media/ucluser/PortableSSD/videos/cam_0")

# Video writers — one per camera
writers = [
    create_video_writer(f"rgb_{cam_idx}", frequency, width, height)
    for cam_idx in range(len(rgb_cams))
]

# Reset scene objects to random positions
set_new_poses()

t = 0
record = True

empty = [0.0, -0.93, 0.0, -2.43, 0.0, 2.25, 0.86]
# Set initial robot pose
initial_joint_positions = np.array([
    0.0,    # joint1
    -0.93,   # joint2
    0.0,    # joint3
   -2.43,   # joint4
    0.0,    # joint5
    2.25,   # joint6
    0.86,   # joint7
    0.0,    # joint8 (gripper left — slightly open)
   -0.0,    # joint9 (gripper right — slightly open)
])
robot.set_joint_positions(initial_joint_positions)
# sim.step(render=True)
print(f"Robot initial pose set to: {initial_joint_positions}")
# robot_action = ArticulationAction(joint_positions=initial_joint_positions)
# robot.apply_action(robot_action)
# sim.step(render=True)
# time.sleep(5)
with torch.inference_mode():
    while simulation_app.is_running():
        sim.step(render=True)

        # --- Capture images ---
        captured_img = image_capture(writers, width, height)

        if captured_img is not None:
            # --- Get current joint state ---
            joint_states = robot.get_joint_positions()
            # print(f"Current joint states at step {t}: {joint_states}")
            grippers = np.array([joint_states[7]])

            # --- Update observation window ---
            update_observation_window(
                config=config,
                imgs=captured_img[:3],  # first 3 cameras match CAMERA_NAMES
                joint_state=joint_states[:7],
                grippers=grippers,
            )

            # --- Run inference every chunk_size steps ---
            if t % rdt_chunk_size == 0:
                image_arrs = [
                    observation_window[-2]['images'][config['camera_names'][0]],
                    observation_window[-2]['images'][config['camera_names'][1]],
                    observation_window[-2]['images'][config['camera_names'][2]],
                    observation_window[-1]['images'][config['camera_names'][0]],
                    observation_window[-1]['images'][config['camera_names'][1]],
                    observation_window[-1]['images'][config['camera_names'][2]],
                ]
                images = [
                    PImage.fromarray(arr) if arr is not None else None
                    for arr in image_arrs
                ]

                # Proprioceptive input: qpos [7] + gripper [1] = [8]
                proprio = torch.cat(
                    [observation_window[-1]['qpos'], observation_window[-1]['gripper']],
                    dim=0,
                ).unsqueeze(0)
                
                # print(f"Proprioceptive input at step {t}: {proprio.cpu().numpy()}")
                # Model outputs actions shaped [1, chunk_size, state_dim]
                actions = policy.step(
                    proprio=proprio,
                    images=images,
                    text_embeds=lang_embeddings,
                ).squeeze(0).cpu().numpy()
                
                # print(f"Model output actions at step {t}: {actions}")

                action_buffer = actions.copy()

            # --- Execute current action from buffer ---
            raw_action = action_buffer[t % rdt_chunk_size]
            action = raw_action
            # print(f"Raw model output action at step {t}: {action}")

            if args.use_actions_interpolation:
                interp_actions = interpolate_action(args, pre_action, action)
            else:
                interp_actions = action[np.newaxis, :]

            for act in interp_actions:
                left_actions = act[0:7]  # 7 joint angles
                left_grip = act[7]       # 1 gripper value
                print(f"left_actions = {left_actions}")
                # Scale from [-1, 1] to real joint limits
                # scaled_left_action = scale_action_to_joint_limits(
                #     np.concatenate((left_actions, np.array([left_grip])), axis=0),
                #     joint_ranges,
                # )
                # # Mirror gripper for opposing finger
                # scaled_left_action = np.concatenate(
                #     (scaled_left_action, np.array([-scaled_left_action[-1]])), axis=0
                # )
                # --- No scaling: use raw model output directly ---
                raw_action_for_robot = np.concatenate(
                    (left_actions, np.array([left_grip]), np.array([-left_grip])), axis=0
                )
                # print(f"Applying raw action (no scaling): {raw_action_for_robot}")
                robot_action = ArticulationAction(joint_positions=raw_action_for_robot)
                robot.apply_action(robot_action)

            pre_action = action.copy()

        t += 1

simulation_app.close()
