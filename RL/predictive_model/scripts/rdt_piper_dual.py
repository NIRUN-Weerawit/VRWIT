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



# ---------------------------
# |       Simulation        |
# ---------------------------
open_stage("/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description/piper_dual_2.usd")
sim = SimulationContext()
dt = sim.get_physics_dt()
sim.reset()
sim.play()
set_camera_view(
    eye=[2.0, 0.0, 4.0],      # camera position
    target=[0.0, 0.0, 2.5],   # look-at point
)
# Camera width height, frequency
width  = 640 
height = 480
frequency = 20
left_robot  = SingleArticulation("/World/piper_description")
right_robot = SingleArticulation("/World/piper_description_01")

# print(f"left dof names : {left_robot.dof_names}")
# for name in left_robot.dof_names:
#     print(left_robot.get_dof_index(name), name)
    
# print(f"right dof names : {right_robot.dof_names}")
# for name in right_robot.dof_names:
#     print(right_robot.get_dof_index(name), name)

# dof names : ['joint1', 'joint2', 'joint3', 'joint1_right', 'joint4', 'joint2_right', 'joint5', 'joint3_right', 'joint6', 'joint4_right', 'joint7', 'joint8', 'joint5_right', 'joint6_right', 'joint7_right', 'joint8_right']
# 0 joint1
# 1 joint2
# 2 joint3
# 3 joint1_right
# 4 joint4
# 5 joint2_right
# 6 joint5
# 7 joint3_right
# 8 joint6
# 9 joint4_right
# 10 joint7
# 11 joint8
# 12 joint5_right
# 13 joint6_right
# 14 joint7_right
# 15 joint8_right
left_index  = np.array([0,1,2,4,6,8,10,11])
right_index = np.array([3,5,7,9,12,13,14,15])


left_robot.initialize()
right_robot.initialize()
print(f"Are they the same object? {left_robot is right_robot}")

left_base               = SingleRigidPrim("/World/piper_description/base_link")
right_base              = SingleRigidPrim("/World/piper_description_01/base_link")
left_piper_hand         = SingleRigidPrim("/World/piper_description/piper_hand")
right_piper_hand        = SingleRigidPrim("/World/piper_description_01/piper_hand")
left_linkEndEffector    = SingleXFormPrim("/World/piper_description/piper_hand/linkEndEffector")
right_linkEndEffector   = SingleXFormPrim("/World/piper_description_01/piper_hand/linkEndEffector")
# cube                = SingleXFormPrim("/World/Xform")
# bin                 = SingleXFormPrim("/World/Xform_bin")
# teddy_bear          = SingleXFormPrim("/World/Xform_teddy_bear")
dex_cube                = SingleXFormPrim("/World/Xform_dex_cube")
# rubik               = SingleXFormPrim("/World/Xform_rubik")
# nvidia_cube         = SingleXFormPrim("/World/Xform_nvidia_cube")
# mug                 = SingleXFormPrim("/World/Xform_mug")

objects             = { 
                       "dex_cube": dex_cube, 
                       }

ext_rgb_cam        = Camera(
                    prim_path="/World/Realsense_mid/RSD455/Camera_OmniVision_OV9782_Color",
                    frequency=frequency,
                    resolution=(width, height),)
left_wrist_rgb_cam        = Camera(
                    prim_path="/World/piper_description/piper_hand/Xform_camera/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
                    frequency=frequency,
                    resolution=(width, height),)
right_wrist_rgb_cam        = Camera(
                    prim_path="/World/piper_description_01/piper_hand/Xform_camera/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
                    frequency=frequency,
                    resolution=(width, height),)
rgb_cams            = [ext_rgb_cam, right_wrist_rgb_cam, left_wrist_rgb_cam]
depth_cams          = None
# depth_cams          = [body_depth_cam, mid_depth_cam, left_depth_cam]

left_base.initialize()
right_base.initialize()
left_piper_hand.initialize()
left_linkEndEffector.initialize()
right_linkEndEffector.initialize()
# left_wrist_rgb_cam.initialize()
ext_rgb_cam.initialize()
left_wrist_rgb_cam.initialize()
right_wrist_rgb_cam.initialize()
# left_rgb_cam.initialize()
# body_depth_cam.initialize()
# mid_depth_cam.initialize()
# left_depth_cam.initialize()
# body_depth_cam.add_distance_to_image_plane_to_frame()
# mid_depth_cam.add_distance_to_image_plane_to_frame()
# left_depth_cam.add_distance_to_image_plane_to_frame()
# body_depth_cam.get_annotator("distance_to_image_plane")


# left_wrist_rgb_cam.add_motion_vectors_to_frame()
left_robot_prim = get_prim_at_path("/World/piper_description")
right_robot_prim = get_prim_at_path("/World/piper_description_01")
left_stage = get_prim_at_path("/World/piper_description").GetStage()
right_stage = get_prim_at_path("/World/piper_description_01").GetStage()

for prim in left_stage.Traverse():
    if not prim.GetPath().HasPrefix(left_robot_prim.GetPath()):
        continue

    if prim.IsA(UsdPhysics.RevoluteJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.GetStiffnessAttr().Set(1e4)
        drive.GetDampingAttr().Set(1e2)
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
        drive.GetStiffnessAttr().Set(1e4)
        drive.GetDampingAttr().Set(1e2)
        
for prim in right_stage.Traverse():
    if not prim.GetPath().HasPrefix(right_robot_prim.GetPath()):
        continue

    if prim.IsA(UsdPhysics.RevoluteJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.GetStiffnessAttr().Set(1e4)
        drive.GetDampingAttr().Set(1e2)
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
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
    robot_description_path="/home/ucluser/isaacgym/assets/urdf/piper_description/config/piper_robot.yaml",
    urdf_path="/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description.urdf"
)
ik_solver.set_default_position_tolerance(0.06)
ik_solver.set_default_orientation_tolerance(0.06)

print(f"#-------------- get_default_position_tolerance() = {ik_solver.get_default_position_tolerance()}")
print(f"#-------------- get_default_orientation_tolerance() = {ik_solver.get_default_orientation_tolerance()}")

#-------------- get_default_position_tolerance() = 0.001
#-------------- get_default_orientation_tolerance() = 0.010000041667134873

kin_solver = ArticulationKinematicsSolver(
    robot_articulation=left_robot,
    kinematics_solver=ik_solver,
    end_effector_frame_name="linkEndEffector" #piper_hand 
)
piper_hand_pos_world, _ = left_piper_hand.get_world_pose()
base_pos_world, base_rot_world = left_base.get_world_pose()
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
    'chunk_size': 32,
    'arm_steps_length': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
    'use_actions_interpolation': False,
    'use_depth_image': False,
    'disable_puppet_arm': False,
    'config_path': "/home/ucluser/RoboticsDiffusionTransformer/configs/base.yaml",
    'pretrained_model_name_or_path': "/home/ucluser/RoboticsDiffusionTransformer/checkpoints",
    'lang_embeddings_path': "/home/ucluser/RoboticsDiffusionTransformer/outs/group_object_collection.pt",
    'ctrl_freq': 15,
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
        'state_dim': 14,  # position (3) + orientation (6)
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
left_dof_count = left_robot.get_joint_positions().shape[0]
right_dof_count = right_robot.get_joint_positions().shape[0]
print(f"left_dof_count = {left_dof_count}, right_dof_count = {right_dof_count}")

pre_action = np.zeros(config['state_dim'])
pre_action[:14] = np.array(
    [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.00133514404296875] + 
    [0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625,-0.3393220901489258, -0.3397035598754883]
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

def image_capture(writers: list | None, depth_stacks: list, rgb_cams: list, depth_cams: list | None, width: int, height: int, record: bool) -> list | None:
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
    
    rgbd_images = []
    if depth_cams is not None:
        assert len(rgb_cams) == len(depth_cams)
    for i in range(len(rgb_cams)):
        
        img     = rgb_cams[i].get_rgba()
        if len(img) == 0:
            return None
        
        if depth_cams is None:
            depth_image = np.full((height, width, 1), fill_value=0.0, dtype=np.float32)
        else:
            depth   = depth_cams[i].get_depth()
            depth_image = depth.copy()
            depth_image = np.clip(depth_image, 0.0, 5.0)
            depth_image = depth_image[:, :, np.newaxis] 
            depth_image = depth_image.reshape((height, width, 1))
            
        color_image = img.copy()
        # Reshape to (height, width, 4) - note: height comes first!
        color_image = color_image.reshape((height, width, 4))
        rgb_image   = color_image[:, :, :3]
        bgr_image   = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # cv2.imwrite(os.path.join(input_save_dir, f"pred_cam{i}_{t:06d}.png"),bgr_image)

        if record:
            writers[i].write(bgr_image)
            depth_stacks[i].append(depth_image)
        # rgbd_image  = np.concatenate((rgb_image, depth_image), axis=2)
        rgbd_images.append(np.concatenate((rgb_image, depth_image), axis=2))
        
    return rgbd_images
        

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


# ---------------------------
# |        Main loop        |
# ---------------------------
try:
    os.makedirs(f"dataset/states" ,  exist_ok=True)
    os.makedirs(f"dataset/actions" , exist_ok=True)
    for cam in range(len(rgb_cams)):
        os.makedirs(f"videos/cam_{cam}",            exist_ok=True)
        os.makedirs(f"dataset/depths/cam_{cam}" ,   exist_ok=True)
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


with torch.inference_mode():
    while simulation_app.is_running():
        
        sim.step(render=True)
        t0 = time.perf_counter()
                        
        left_joint_states   = left_robot.get_joint_positions()
        joint_states  = np.zeros(16, dtype=float)
        # print(f"Joint states: {', '.join(f'{joint * 180 / math.pi:.4f}' for joint in left_joint_states)}")
        
        joint_states[0:8]  = left_joint_states[left_index]
        # print(f"Left  states: {', '.join(f'{joint * 180 / math.pi:.4f}' for joint in joint_states)}")
        joint_states[8:16] = left_joint_states[right_index]
        # right_joint_states  = right_robot.get_joint_positions()
        print(f"Right states: {', '.join(f'{joint * 180 / math.pi:.4f}' for joint in joint_states)}")
        # print(f"Joint states: {', '.join(f'{joint:.2f}' for joint in right_joint_states[0:16])}")
        # print(f"dimension of left joint states: {left_joint_states.shape}")
        # print(f"dimension of right joint states: {right_joint_states.shape}")
        
        #-----RDT-1B (start)--------------#
        captured_img = image_capture(None, depth_stacks, rgb_cams , depth_cams, width, height, record=False)
        
        t2 = time.perf_counter()
        if captured_img:
            
            # left_EEF_pos, _ = left_linkEndEffector.get_world_pose()
            # right_EEF_pos, _ = right_linkEndEffector.get_world_pose()
            fixed_joint_states    = np.concatenate((joint_states[0:6] , joint_states[8:14]), axis=0)
            grippers        = np.array((joint_states[6],joint_states[14]))
            # print(f"Dimension joint_states : {joint_states.shape} ")
            # print(f"Dimension grippers : {grippers.shape} ")
            
            update_observation_window(config=config,imgs=captured_img, joint_state=fixed_joint_states, EEF_position=None, grippers=grippers)
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
                # print(f"proprio = {proprio}")
                # unsqueeze to [1, 11]
                proprio = proprio.unsqueeze(0)
                # actions shaped as [1, 64, 8] in format [right]
                # print("PRODUCING ACTIONS")
                # print(f"devices: proprio={proprio.device}")
                # print(f"devices: images={images[0].device}")
                # print(f"devices: lang_emb={lang_embeddings.device}")
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
                right_actions   = act[6:12]
                # left_grip       = act[6]
                left_grip       = act[12]
                right_grip      = act[13]
                # print(f"target action shape: {act.shape}")
                # print(f"left_actions type = {type(left_actions)}")
                # print(f"left_actions = {left_actions}")
                # print(f"left_grip = {left_grip}")
                # print(f"left_grip shape = {left_grip.shape}")
                # print(f"left_grip type = {type(left_grip)}")
                # print(f"right_grip = {right_grip}")
                # print(f"right_grip shape = {right_grip.shape}")
                # print(f"left_action shape = {left_actions.shape}")
                # print(f"right_action shape = {right_actions.shape}")
                # print(f"dimension of total left action = {np.array([left_actions + left_grip + left_grip]).shape}")
                
                scaled_left_action  = scale_action_to_joint_limits(np.concatenate((left_actions,  np.array([left_grip])), axis=0),  joint_ranges)
                scaled_left_action  = np.concatenate((scaled_left_action,  np.array([-1 * scaled_left_action[-1]])), axis=0)
                scaled_right_action = scale_action_to_joint_limits(np.concatenate((right_actions, np.array([right_grip])), axis=0), joint_ranges)
                scaled_right_action  = np.concatenate((scaled_right_action,  np.array([-1 * scaled_right_action[-1]])), axis=0)
                # robot_pos           = np.concatenate((scaled_left_action, np.array([-1 * scaled_left_action[-1]]), scaled_right_action, np.array([-1 * scaled_right_action[-1]])) , axis=0)
                
                print(f"scaled_left_action pos : {scaled_left_action  * 180 / math.pi}")
                print(f"scaled_right_action pos : {scaled_right_action  * 180 / math.pi}")
                robot_position = np.zeros(16, dtype=float)
                robot_position[left_index]  = scaled_left_action
                robot_position[right_index] = scaled_right_action
    
                # left_action   = ArticulationAction(joint_positions=np.concatenate((left_actions, np.array([left_grip, left_grip])), axis=0))
                # right_action   = ArticulationAction(joint_positions=np.concatenate((right_actions, np.array([right_grip, right_grip])), axis=0))
                # robot_position = np.concatenate((left_actions, np.array([left_grip, -1 * left_grip])), axis=0)
                # robot_position = np.concatenate((left_actions, np.array([left_grip, -1 * left_grip]), right_actions, np.array([right_grip, -1 * right_grip])), axis=0)
                # robot_position = np.concatenate((right_actions, np.array([right_grip, -1 * right_grip]), left_actions, np.array([left_grip, -1 * left_grip])), axis=0)
                print(f"reordered robot position : {robot_position  * 180 / math.pi}")
                # robot_position      = np.zeros(16, dtype=float)
                # robot_position[5]   = -0.25039473   * math.pi / 180
                # robot_position[10]  =  20.04485302   * math.pi / 180
                # robot_position[11]  = -5.81892053  * math.pi / 180
                
                # left_position = np.concatenate((left_actions, np.array([left_grip, -1 * left_grip])), axis=0)
                # right_position = np.concatenate((right_actions, np.array([right_grip, -1 * right_grip])), axis=0)
                # left_action = ArticulationAction(joint_positions=left_position)
                # right_action = ArticulationAction(joint_positions=right_position)

                robot_action   = ArticulationAction(joint_positions=robot_position)
                
                left_robot.apply_action(robot_action)
                # left_robot.apply_action(left_action)
                # right_robot.apply_action(robot_action)
            
            pre_action = action.copy()
            
            t+=1
        
    
simulation_app.close()
