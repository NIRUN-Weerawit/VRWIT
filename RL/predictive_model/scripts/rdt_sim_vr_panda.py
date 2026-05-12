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
from policy import ACTPolicy
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


from pxr import UsdPhysics,UsdShade, PhysxSchema
"""import inspect
from omni.isaac.motion_generation.lula import LulaKinematicsSolver

print(inspect.signature(LulaKinematicsSolver))"""


# ---------------------------
# |  VR-server Connection   |
# ---------------------------
broker = "sora2.uclab.jp"
port = 1883
client_id = 'PiPER-control-wee'
# topic = "control/d0fee136-7315-49ed-9b53-a3973fe4128e-14mewk9"
topic = "control/piper-wee"

def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    client = mqtt_client.Client(client_id=client_id, callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2)

    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

vr_joints = [0]*7
vr_goal_pos = [0.2, 0.2, 0.2]
vr_goal_rot = [0.0, 0.0, 0.0, 1.0]  # Quaternion (x, y, z, w)
# vr_goal_rot = [0.0, 0.0, 0.0]  # euler (x, y, z)
grip_flag = None
vr_joints_lock = Lock()
trigger_on = None
prev_trigger_on = False
data_rot = [0.0, 0.0, 0.0, 1.0]
controller_obj = [0.0, 0.0, 0.0]
buttonA = False
buttonB = False
thumbstick = None
client = connect_mqtt()
recv_times = collections.deque(maxlen=20)
recv_messages = collections.deque(maxlen=1000)
old_time = time.time()
avg = 0.0
latency = 0
acc_latency = 0
avg_latency = 0
DELAY = False
# ANSI shorthands
SAVE = "\033[s"
RESTORE = "\033[u"
CLEAR = "\033[K"

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        global vr_joints, vr_goal_pos, vr_goal_rot, grip_flag, trigger_on, controller_obj, buttonA, buttonB, thumbstick, recv_times, recv_messages, old_time, avg
        now = time.monotonic()
        time_new = time.time()
        recv_times.append(now)
        data = msg.payload.decode()
        """if time_new - old_time >= 0.00 and json.loads(data)['sending']:
            recv_messages.append(data)
            old_time = time_new"""
        buttonA = json.loads(data)['buttonA']
        buttonB = json.loads(data)['buttonB']
        thumbstick = json.loads(data)['thumbstick']
        if DELAY:
            # if json.loads(data)['sending']:
            recv_messages.append((data, time_new))
            # trigger_on = json.loads(data)['sending']
            print(f"size of the recv_messages :{len(recv_messages)}" )
            print(SAVE + "\033[4A" + CLEAR + f"size of the recv_messages :{len(recv_messages)}" + RESTORE, end='', flush=True)
            if len(recv_times) >= recv_times.maxlen:
                span = recv_times[-1] - recv_times[0]
                avg = (len(recv_times)-1) / span
                print(SAVE            # remember current cursor
                    + "\033[3A"        # up 2 lines, now at line 1
                    + CLEAR            # clear that entire line
                    + f"Subscriber avg freq: {avg:.1f} Hz"
                    + RESTORE         # go back to saved spot
                    , end="", flush=True
                )
        else:
            # data_joints = json.loads(data)['joints']
            data_pos = json.loads(data)['goal_pos']
            data_rot = json.loads(data)['goal_rot']
            grip_flag = json.loads(data)['grip']
            trigger_on = json.loads(data)['sending']
            data_controller = json.loads(data)['controller_object']
            # vr_joints[:]    = data_joints
            vr_goal_pos     = [data_pos['z'], data_pos['x'], data_pos['y']]
            # controller_obj  = [0,0,0]
            controller_obj  = [data_controller['_x'], data_controller['_y'], data_controller['_z']]
            vr_goal_rot     = data_rot       
            # print(f"data_controller: {data_controller} ")
        # print("gripper", gripper)
        # print("type", type(vr_joints))
        # print("vr_joint", vr_joints)
        # print("vr_goal_pos", vr_goal_pos)
        # print("vr_goal_rot", vr_goal_rot)
        # print('len(vr):', len(vr_joints))
        # print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

    client.subscribe(topic)
    client.on_message = on_message   



# ---------------------------
# |       Simulation        |
# ---------------------------
open_stage("/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description/franka_simple.usd")
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
frequency = 20
robot = SingleArticulation("/World/franka")
robot.initialize()

base                = SingleRigidPrim("/World/franka/panda_link0")
table               = SingleXFormPrim("/World/Xform_table")
franka_hand          = SingleRigidPrim("/World/franka/panda_hand")
# linkEndEffector     = SingleXFormPrim("/World/piper_description/piper_hand/linkEndEffector")
cube_xform                = SingleXFormPrim("/World/Xform")
bin_xform                 = SingleXFormPrim("/World/Xform_bin")
teddy_bear_xform          = SingleXFormPrim("/World/Xform_teddy_bear")
dex_cube_xform            = SingleXFormPrim("/World/Xform_dex_cube")
rubik_xform               = SingleXFormPrim("/World/Xform_rubik")
nvidia_cube_xform         = SingleXFormPrim("/World/Xform_nvidia_cube")



# cube_prim                 = SingleRigidPrim("/World/Xform")
bin_prim                  = SingleRigidPrim("/World/Xform_bin/small_KLT")
# teddy_bear_prim           = SingleRigidPrim("/World/Xform_teddy_bear")
dex_cube_prim             = SingleRigidPrim("/World/Xform_dex_cube/dex_cube_instanceable")
rubik_prim                = SingleRigidPrim("/World/Xform_rubik/rubiks_cube")
nvidia_cube_prim          = SingleRigidPrim("/World/Xform_nvidia_cube/nvidia_cube")


# mug                 = SingleXFormPrim("/World/Xform_mug")
objects             = {"cube":cube_xform, 
                    #    "teddy_bear": teddy_bear_xform, 
                       "dex_cube": dex_cube_xform, 
                       "rubik":rubik_xform, 
                       "nvidia_cube": nvidia_cube_xform,
                       "bin": bin_xform}

objects_prim        = {"dex_cube": dex_cube_prim, 
                       "rubik":rubik_prim, 
                       "nvidia_cube": nvidia_cube_prim,
                       "bin": bin_prim}
prim_paths = ["/World/Xform_bin/small_KLT", "/World/Xform_dex_cube/dex_cube_instanceable" , "/World/Xform_rubik/rubiks_cube", "/World/Xform_nvidia_cube/nvidia_cube"]

# mass_api = UsdPhysics.MassAPI.Apply(table.prim)
# mass_api.CreateMassAttr(30.2) 

# for object in objects_prim:
    # print(f"Mass of {object} is {objects[object].prim.get_mass()}")
    # mass_api = UsdPhysics.MassAPI.Apply(objects[object].prim)
    # mass_api.CreateMassAttr(1.0)  # kilograms
    
    # Find the bound material
    # binding = UsdShade.MaterialBindingAPI(objects_prim[object].prim)
    # mat_path = binding.GetDirectBinding().GetMaterialPath()
    # mat_prim = get_prim_at_path(str(mat_path))

    # Read PhysX friction
    # physx_mat = PhysxSchema.PhysxMaterialAPI(mat_prim)
    # static_friction = physx_mat.GetStaticFrictionAttr().Get()
    # dynamic_friction = physx_mat.GetDynamicFrictionAttr().Get()

    # print(f"static={static_friction}, dynamic={dynamic_friction}")
    # print(f"Coeff of friction of {object} is {objects_prim[object].prim.get_friction_coefficients()}")


# for object in objects_prim:
#     objects_prim[object].initialize()
#     print(f"Mass of {object} is {objects_prim[object].get_mass()}")

# Mass of dex_cube is 0.2160000056028366
# Mass of rubik is 1.0
# Mass of nvidia_cube is 0.2746249735355377
# Mass of bin is 0.5600000023841858

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
body_depth_cam      = Camera(
                    prim_path="/World/franka/panda_hand/Realsense/RSD455/Camera_Pseudo_Depth",
                    frequency=frequency,
                    resolution=(width, height),)
mid_depth_cam        = Camera(
                    prim_path="/World/Realsense_mid/RSD455/Camera_Pseudo_Depth",
                    frequency=frequency,
                    resolution=(width, height),)
left_depth_cam        = Camera(
                    prim_path="/World/Realsense_left/RSD455/Camera_Pseudo_Depth",
                    frequency=frequency,
                    resolution=(width, height),)

rgb_cams            = [mid_rgb_cam, body_rgb_cam, body_rgb_cam]
depth_cams          = None
# depth_cams          = [body_depth_cam, mid_depth_cam, left_depth_cam]

base.initialize()
franka_hand.initialize()
# linkEndEffector.initialize()
body_rgb_cam.initialize()
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
robot_prim = get_prim_at_path("/World/franka")
# print(f"dof names : {robot.dof_names}")
# for name in robot.dof_names:
#     print(robot.get_dof_index(name), name)
# print(f"dof names : {robot_prim.dof_names}")    

# dof names : ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
# 0 panda_joint1
# 1 panda_joint2
# 2 panda_joint3
# 3 panda_joint4
# 4 panda_joint5
# 5 panda_joint6    
# 6 panda_joint7
# 7 panda_finger_joint1
# 8 panda_finger_joint2

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
# piper_hand_pos_world, _ = franka_hand.get_world_pose()
base_pos_world, base_rot_world = base.get_world_pose()
R_base = R.from_quat(base_rot_world)

# small reachable offset
target_pos = np.array([0.2, 0.0, 0.25])
VR_target_marker.set_world_pose(position=target_pos)
Model_target_marker.set_world_pose(position=target_pos)

# target_rot = euler_angles_to_quat(np.array([0.5 * math.pi, controller_obj[1], 0]))

# grip_action = ArticulationAction(joint_positions=robot.get_joint_positions())

# --------------------------------------
# |        Predictive Model Import     |
# --------------------------------------
# fixed parameters
save_dir = "/home/ucluser/debug_images"
input_save_dir = "/home/ucluser/input_images"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(input_save_dir, exist_ok=True)
alpha_pos       = 0.0
alpha_rot       = 0.0
state_dim       = 8 # 14
action_dim      = 7
lr_backbone     = 1e-5
backbone        = 'resnet18'
enc_layers      = 4
dec_layers      = 7
nheads          = 8
temporal_agg    = True     #args.temp
chunk_size      = 25        #args.chunk_size
max_timesteps   = 4000
policy_config   = {'lr': 1e-4,
                  'num_queries': chunk_size,
                  'kl_weight': 10,
                  'hidden_dim': 512,
                  'dim_feedforward': 2048,
                  'lr_backbone': lr_backbone,
                  'backbone': backbone,
                  'enc_layers': enc_layers,
                  'dec_layers': dec_layers,
                  'nheads': nheads,
                  'camera_names': ['cam1', 'cam2'],
                  'vq': False,

                  'action_dim': action_dim, # 16
                  'state_dim': state_dim,
                  }
root_dir = "/home/ucluser/VRWIT/RL/predictive_model"
gin.parse_config_file(f"{root_dir}/configs/base_train_config.gin", skip_unknown=True)
camera_names    = policy_config["camera_names"]
ckpt_dir        = f"{root_dir}/checkpoint_{chunk_size}_{10}"
# ckpt_dir      = "checkpoint_25_01"
# ckpt_dir      = "/media/ucluser/PortableSSD/checkpoint_x"
# ckpt_dir      = "checkpoint_x"
# ckpt_name     = f'25_best.ckpt'
# ckpt_name     = f'policy_step_15000_seed_28.ckpt'
ckpt_name       = f'policy_best.ckpt'
ckpt_path       = os.path.join(ckpt_dir, ckpt_name)
"""policy          = ACTPolicy(policy_config)
loading_status  = policy.deserialize(torch.load(ckpt_path))
print(loading_status)
policy.cuda()
policy.eval()"""
print(f'Loaded: {ckpt_path}')
stats_path      = os.path.join(ckpt_dir, f'dataset_stats.pkl')
with open(stats_path, 'rb') as f:
    stats       = pickle.load(f)
pre_process = lambda s_obs: (s_obs - stats['obs_mean']) / stats['obs_std']
post_process = lambda a: a * stats['action_std'] + stats['action_mean']
query_frequency = policy_config['num_queries']

if temporal_agg:
    end_time = 0.99 * max_timesteps
    query_frequency = 10
    num_queries = policy_config['num_queries']
    all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, action_dim]).cuda()
else:
    end_time = max_timesteps * 2
print(f"query_frequnecy: {query_frequency}")


# ---------------------------
# |         RDT-1B          |
# ---------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument('--max_publish_step', action='store', type=int, 
#                     help='Maximum number of action publishing steps', default=10000, required=False)
# parser.add_argument('--chunk_size', action='store', type=int, 
#                     help='Action chunk size',
#                     default=64, required=False)
# parser.add_argument('--arm_steps_length', action='store', type=float, 
#                     help='The maximum change allowed for each joint per timestep',
#                     default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

# parser.add_argument('--use_actions_interpolation', action='store_true',
#                     help='Whether to interpolate the actions if the difference is too large',
#                     default=False, required=False)
# parser.add_argument('--use_depth_image', action='store_true', 
#                     help='Whether to use depth images',
#                     default=False, required=False)

# parser.add_argument('--disable_puppet_arm', action='store_true',
#                     help='Whether to disable the puppet arm. This is useful for safely debugging',default=False)

# parser.add_argument('--config_path', type=str, default="/home/ucluser/RoboticsDiffusionTransformer/configs/base.yaml", 
#                     help='Path to the config file')

# parser.add_argument('--pretrained_model_name_or_path', type=str, required=False,
#                     help='Name or path to the pretrained model')

# parser.add_argument('--lang_embeddings_path', type=str, required=False, default="/home/ucluser/RoboticsDiffusionTransformer/outs/object_collection.pt",
#                     help='Path to the pre-encoded language instruction embeddings')
# parser.add_argument('--ctrl_freq', action='store', type=int, 
#                     help='The control frequency of the robot',
#                     default=25, required=False)
# args = parser.parse_args()

args = {
    'max_publish_step': 300000,
    'chunk_size':15,
    'arm_steps_length': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2, 0.2],
    'use_actions_interpolation': False,
    'use_depth_image': False,
    'disable_puppet_arm': False,
    'config_path': "/home/ucluser/RoboticsDiffusionTransformer/configs/base.yaml",
    'pretrained_model_name_or_path': "/home/ucluser/RoboticsDiffusionTransformer/checkpoints-10000",
    'lang_embeddings_path': "/home/ucluser/RoboticsDiffusionTransformer/outs/object_collection.pt",
    'ctrl_freq': 10,
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
        pretrained="/home/ucluser/RoboticsDiffusionTransformer/checkpoints-10000",
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )
    return model

def get_config(args):
    config = {
        'episode_len': 300000,
        'state_dim': 9,  # position (3) + orientation (6)
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config

# Update the observation window buffer
def update_observation_window(config, imgs, joint_state, EEF_position):
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
    
    EEF_pos = EEF_position
    EEF_pos = torch.from_numpy(EEF_pos).float().cuda()
    
    observation_window.append(
        {
            'qpos': qpos,
            'EEF_position': EEF_pos,
            'images':
                {
                    config["camera_names"][0]: imgs[0],
                    config["camera_names"][1]: imgs[1],
                    config["camera_names"][2]: imgs[2],
                },
        }
    )

def interpolate_joints(args, prev_action, cur_action):
    """
    Interpolates between two actions to smooth motion execution.
    
    Args:
        args: Arguments object containing arm_steps_length
        prev_action: Previous action (shape: [8,] for 7 joints + 1 gripper)
        cur_action: Current action (shape: [8,])
    
    Returns:
        Interpolated actions (shape: [n, 8])
    """
    # Use only the first 7 joint step limits for action dimensions 0-6
    # Dimension 7 (gripper) uses a larger step limit
    arm_steps = np.array(args.arm_steps_length)  # [7 elements]
    
    # Add gripper step limit (use 0.04 as max gripper range)
    step_limits = np.concatenate((arm_steps[:9], np.array([0.04])), axis=0)  # [8 elements]
    
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / step_limits).astype(int)
    step = np.max(step)
    
    if step <= 1:
        return cur_action[np.newaxis, :]
    
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]

def interpolate_EEF(args, prev_pos, cur_pos):
    """
    Interpolates between two EEF (End-Effector) positions to smooth motion execution.
    
    Args:
        args: Arguments object containing arm_steps_length
        prev_pos: Previous EEF position (shape: [3,] for x, y, z)
        cur_pos: Current EEF position (shape: [3,])
    
    Returns:
        Interpolated EEF positions (shape: [n, 3])
    """
    # Use a linear distance-based step size for EEF positions
    # Maximum distance allowed per step is derived from first 3 joint step limits
    max_step_size = np.mean(np.array(args.arm_steps_length[:3]))  # Average of first 3 joints
    
    # Calculate euclidean distance between positions
    distance = np.linalg.norm(cur_pos - prev_pos)
    
    # Calculate number of steps needed
    num_steps = int(np.ceil(distance / max_step_size))
    
    if num_steps <= 1:
        return cur_pos[np.newaxis, :]
    
    # Linear interpolation along the path
    new_positions = np.linspace(prev_pos, cur_pos, num_steps + 1)
    return new_positions[1:]  # Exclude the first (previous) position


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

pre_action = np.zeros(config['state_dim'])
pre_action[:9] = np.array(
    [ -70.0, -50.0, 60.0, -120.0, 50.0, 100.0, -25.0, 0.0, 0.0 ] 
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
        
"""    # body_img     = body_rgb_cam.get_rgba()
    # mid_img      = mid_rgb_cam.get_rgba()
    # body_depth   = body_depth_cam.get_depth()
    # mid_depth    = mid_depth_cam.get_depth()
    # if len(body_img) == 0 or body_depth is None or  mid_depth is None:
    #     # print("no image yet...")
    #     return None
    # print(f"#---------img size  : {len(img)}    --------#")
    # print(f"#---------img shape : {body_img.shape}   --------#")
    # print(f"#---------img shape : {body_depth.shape}   --------#")
    # color_image_1   = mid_img.copy()
    # color_image_2   = body_img.copy()
    # depth_image_1   = mid_depth.copy()
    # depth_image_2   = body_depth.copy()
    # print(f"max = {depth_image_1.max()}, min = {depth_image_1.min()}")
    
    # print("shape color_image_1 :", color_image_1.shape)
    
    depth_image_1 = np.clip(depth_image_1, 0.0, 5.0)
    depth_image_2 = np.clip(depth_image_2, 0.0, 5.0)
    
    depth_image_1 = depth_image_1[:, :, np.newaxis]  # shape: [H, W, 1]
    depth_image_2 = depth_image_2[:, :, np.newaxis]
    
    img_np_1 = color_image_1.reshape((640, 480 , 4))
    img_np_2 = color_image_2.reshape((640, 480 , 4))
    depth_image_1 = depth_image_1.reshape((640, 480 , 1))
    depth_image_2 = depth_image_2.reshape((640, 480 , 1))
    

    rgb_image_1 = img_np_1[:, :, :3]
    rgb_image_2 = img_np_2[:, :, :3]
    
    if record:
        writers[0].write(rgb_image_1)
        writers[1].write(rgb_image_2)
        depth_stacks.append(depth_image_1)
        depth_stacks.append(depth_image_2) 
    
    # cv2.imshow("color_1",rgb_image_1)
    # cv2.imshow("color_1",cv2.cvtColor(rgb_image_1, cv2.COLOR_RGB2BGR))
    # cv2.imshow("color_2",cv2.cvtColor(rgb_image_2, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)
    # depth_colormap_1 = cv2.convertScaleAbs(depth_image_1, alpha=100)
    # depth_colormap_2 = cv2.convertScaleAbs(depth_image_2, alpha=100)

    cv2.imwrite(os.path.join(input_save_dir, f"pred_cam1_{t:06d}.png"),
                depth_colormap_1,
                )
    cv2.imwrite(os.path.join(input_save_dir, f"pred_cam2_{t:06d}.png"),
                depth_colormap_2,
                )
    cv2.imwrite(
                    os.path.join(input_save_dir, f"pred_cam1_{t:06d}.png"),
                    cv2.cvtColor(color_image_1, cv2.COLOR_RGB2BGR),
                )
    cv2.imwrite(
                    os.path.join(input_save_dir, f"pred_cam2_{t:06d}.png"),
                    cv2.cvtColor(color_image_2, cv2.COLOR_RGB2BGR),
                )
    
    rgbd_image_1    = np.concatenate((rgb_image_1, depth_image_1), axis=2)
    rgbd_image_2    = np.concatenate((rgb_image_2, depth_image_2), axis=2)  
    
    
    return [rgbd_image_1, rgbd_image_2]  """

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
            ((0.0, 0.5), (-0.6, -0.3)),
            ((0.0, 0.5), (0.3, 0.6)),
            ((0.5, 1.0),(-0.5, 0.5))
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

def convert_6d_to_quaternion(rotation_6d):
    """
    Converts 6D rotation representation to quaternion.
    
    The 6D representation uses two orthonormal 3D vectors:
    - First 3 dims: first basis vector (x-axis)
    - Last 3 dims: second basis vector (y-axis)
    - z-axis is computed via cross product
    
    Args:
        rotation_6d: Array of shape [6,] or [n, 6] containing 6D rotation vectors
    
    Returns:
        Quaternion in (x, y, z, w) format. Shape [4,] or [n, 4]
    """
    is_batch = len(rotation_6d.shape) > 1
    
    if not is_batch:
        rotation_6d = rotation_6d[np.newaxis, :]
    
    batch_size = rotation_6d.shape[0]
    quaternions = []
    
    for i in range(batch_size):
        # Extract the two basis vectors
        vec1 = rotation_6d[i, :3].astype(np.float64)
        vec2 = rotation_6d[i, 3:6].astype(np.float64)
        
        # Normalize the first vector
        norm1 = np.linalg.norm(vec1)
        if norm1 < 1e-8:
            vec1 = np.array([1.0, 0.0, 0.0])
        else:
            vec1 = vec1 / norm1
        
        # Gram-Schmidt orthogonalization
        # Make vec2 orthogonal to vec1
        dot_product = np.dot(vec1, vec2)
        vec2 = vec2 - dot_product * vec1
        
        norm2 = np.linalg.norm(vec2)
        if norm2 < 1e-8:
            vec2 = np.array([0.0, 1.0, 0.0]) if abs(vec1[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        else:
            vec2 = vec2 / norm2
        
        # Compute the third basis vector via cross product
        vec3 = np.cross(vec1, vec2)
        
        # Create rotation matrix [3x3] - each vector is a column
        rotation_matrix = np.column_stack([vec1, vec2, vec3])
        
        # Convert rotation matrix to quaternion using scipy
        quat = R.from_matrix(rotation_matrix).as_quat()  # Returns (x, y, z, w)
        quaternions.append(quat)
    
    result = np.array(quaternions)
    
    if not is_batch:
        result = result[0]
    
    return result
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
pos_save    = np.array([0.15,-0.07, 3.0]) - base_pos_world
ee_pos_world, current_rotation = franka_hand.get_world_pose()
pos_marker_save = ee_pos_world - base_pos_world
pre_process = lambda s_obs: (s_obs - stats['obs_mean']) / stats['obs_std']
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
JOINT_GAIN = np.array([2.5, 2.5, 2.5, 2.5, 2.0, 2.0, 2.0])
# for cam in range(len(rgb_cams)):
#     writers.append(create_video_writer(f"rgb_{cam}", frequency, width, height))
#     depth_stacks.append([])
print(f" number of writers : {len(writers)}")
subscribe(client)
client.loop_start()
with torch.inference_mode():
    while simulation_app.is_running():
        
        sim.step(render=True)
        t0 = time.perf_counter()
        
        # --------- VR Controller Part (Start) ---------------------------------------------------------#
        """"""
        pos_controller = np.array(vr_goal_pos)
        # target_rot = euler_angles_to_quat(np.array([ math.pi, 0, 0])) # Always pointing down
        # target_rot = euler_angles_to_quat(np.array([math.pi, 0, controller_obj[2]])) # Always pointing down
        quat_controller = euler_angles_to_quat(np.array([controller_obj[2], controller_obj[0],controller_obj[1]])) #
        # target_rot = np.array(down_vector)
        # target_rot = np.array([-0.70710678, 0, 0, 0.70710678])
        
        ee_pos_world, current_rotation = franka_hand.get_world_pose()
        # ee_pos_world, current_rotation = piper_hand.get_world_pose()
        # target_pos       = pos_controller
        
        # from OFF -> ON
        if trigger_on and not prev_trigger_on:
            # print("###### CHANGE OFF -> ON ########")
            pos_start       = ee_pos_world - base_pos_world
            # pos_start       = pos_marker_save
            pos_start_ctrl  = pos_controller
            
            quat_start = R.from_quat(current_rotation)       # w,x,y,z
            # quat_start = R.from_quat(quat_save)       # w,x,y,z
            quat_start_ctrl = R.from_quat(quat_controller)

        if quat_start is None:  # trigger never pressed yet
            prev_trigger_on = trigger_on
            continue
        
        # from ON -> OFF
        if prev_trigger_on and not trigger_on:
            # print("###### CHANGE ON -> OFF ########")
            pos_save  = total_pose
            pos_marker_save = total_pose
            quat_save = total_rotation.as_quat()    

        if trigger_on:
            pos_ctrl_delta  = pos_controller - pos_start_ctrl 
            total_pose      = pos_start + pos_ctrl_delta
            target_pos      = total_pose
            # print(f"pos_start       when trigger on: {pos_start}")
            # print(f"pos_ctrl_delta  when trigger on: {pos_ctrl_delta}")
            # print(f"target_pose     when trigger on: {target_pos}")
            
            # Compute relative rotation
            quat_ctrl_delta = quat_start_ctrl.inv() * R.from_quat(quat_controller)
            # quat_difference_1 = quat_start.inv() * quat_ctrl_delta
            total_rotation = quat_start * quat_ctrl_delta   # apply delta

        else:
            # pass
            target_pos      = pos_save
            total_rotation  = R.from_quat(quat_save)
            # total_rotation  = R.from_quat(current_rotation)
            
        target_rot = total_rotation.as_quat()  # use as IK target"""
        # --------- VR Controller Part (Start) ---------------------------------------------------------#
        
        # -----  Gripper --------------------- #
        joint_efforts = robot.get_measured_joint_efforts()
        # print(f"joint efforts = {joint_efforts}")
        # current q = [ 1.269203  -0.5463999 -1.8254468 -2.8402915 -2.215929   3.2872994
        #                 2.3726237  0.04       0.04     ]

        # joint efforts = [ 4.9686432e-03 -6.8424745e+00  5.2181525e+00  1.9006359e+01
        # -6.9858432e-03 -9.1853642e-01  2.5719404e-05  2.0853344e-02
        # -2.4953645e-02]
        

        if grip_flag:
            joint_efforts[7] = - 500.0
            # joint_efforts[8] = 50.0
            
        else:
            joint_efforts[7] = 1000.0
            # joint_efforts[8] = -50.0
            # current_q[6]     = 0.04
            # current_q[7]     = -0.04
            
        # robot.apply_action(action_target)
            
        grip_action     = ArticulationAction(joint_efforts=joint_efforts)
        # grip_action_1   = ArticulationAction(joint_positions=current_q)
        robot.apply_action(grip_action)
        # robot.apply_action(grip_action_1)
        t1 = time.perf_counter()
        # print(type(action_target))
        
        # If button B is pressed but not being recorded yet, start the record
        if buttonB:
            record = True
            print("The video is being recorded!")
        # But if it is already recording and button B is press, stop the record.
        # elif buttonB and record:
        #     record = False
        # captured_img = image_capture(None, depth_stacks, rgb_cams, depth_cams, width, height, record=False)
        current_q = robot.get_joint_positions()
        # print(f"current q = {current_q}")
        
        if thumbstick == 1:
            if alpha_pos < 1.0:
                alpha_pos += 0.1
            else:
                alpha_pos = 1.0
        elif thumbstick == 3:
            if alpha_pos > 0.0:
                alpha_pos -= 0.1
            else:
                alpha_pos = 0.0
        elif thumbstick == 0:
            if alpha_rot < 1.0:
                alpha_rot += 0.1
            else:
                alpha_rot = 1.0
        elif thumbstick == 2:
            if alpha_rot > 0.0:
                alpha_rot -= 0.1
            else:
                alpha_rot = 0.0
        alpha_pos = round(alpha_pos,1)
        alpha_rot = round(alpha_rot,1)
        
        #-------- Prediction model (start) -------------------------------------------------------# 
        '''  
        # print(f"gripper: {current_q[6]}, {current_q[7]}")
        # print(f"current_q type : {type(current_q)}")
        obs_numpy = np.array(current_q)
        
        obs = pre_process(obs_numpy)
        t2  = time.perf_counter()
        # print(f"shape obs = {obs.shape}")
        if len(obs.shape) > 1:
            obs = torch.from_numpy(obs).float().cuda()
        else:
            obs = torch.from_numpy(obs).float().cuda().unsqueeze(0)
        
        if captured_img:
            # print("FOUND IMAGE")
            if t == 0:
                # warm up
                curr_image = get_image(captured_img, camera_names, stats)
                for _ in range(10):
                    # print(f"obs: {obs}, shape: {obs.shape}")
                    policy(obs, curr_image)
                print('network warm up done')
            
            if t % query_frequency == 0:
            # if t % 2 == 0:  
                curr_image = get_image(captured_img, camera_names, stats)
                # with torch.no_grad():
                # print(f"time:{t}, obs= {obs}")
                all_actions, rgb_prediction = policy(obs, curr_image)
                # show rgb reconstruciton
                """print(f"rgb output keys = {rgb_prediction.keys()}")
                output_1 = rgb_prediction[f"rgb_cam_1_1"]
                # output_2 = rgb_prediction[f"rgb_cam_2_1"]
                # print(f"shape of rgb_1 reconstruction: {output_1.shape} ")
                # print(f"shape of rgb_2 reconstruction: {output_2.shape} ")
                pred_1 = output_1[0]
                # pred_2 = output_2[0]
                img_1 = pred_1.squeeze(0)          # -> [3, 480, 640]
                # img_2 = pred_2.squeeze(0)          # -> [3, 480, 640]
                img_1 = img_1.permute(1, 2, 0)     # -> [480, 640, 3]
                # img_2 = img_2.permute(1, 2, 0)     # -> [480, 640, 3]
                img_1 = img_1.detach().cpu().numpy()
                # img_2 = img_2.detach().cpu().numpy()
                if img_1.max() <= 1.0:
                    img_1 = (img_1 * 255).astype('uint8')
                    # img_2 = (img_2 * 255).astype('uint8')
                else:
                    img_1 = img_1.astype('uint8')
                    # img_2 = img_2.astype('uint8')

                cv2.imwrite(
                    os.path.join(save_dir, f"pred_cam1_{t:06d}.png"),
                    cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR),
                )"""

                # cv2.imwrite(
                #     os.path.join(save_dir, f"pred_cam2_{t:06d}.png"),
                #     cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR),
                # )
                
                
                # all_actions, time_inf = timed_inference(policy, obs, curr_image)
                # print(f"Inference time: {time_inf*1000:.2f} ms")
                    
            # t3 = time.perf_counter()
            if all_actions is not None:
                if temporal_agg :
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    # print(f"all_time_acitons: size={all_time_actions.shape}, values={all_time_actions}")
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)     
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True).float()
                else:
                    raw_action = all_actions[:, t % query_frequency] #query_freq
            # print(f"raw_action = {raw_action}")
            # t4 = time.perf_counter()
            
                raw_action = raw_action.squeeze(0).cpu().detach().numpy()
                # print(f"raw action= {raw_action}, type={type(raw_action)}")
                action = post_process(raw_action)
                # print(f"post_processed action= {action}, type={type(action)}, shape={action.shape}")
                # target_obs = action[:action_dim]
                target_action = action[:]
                # predicted_trajectories.append(target_action)
                
                Model_target_marker.set_world_pose(position = target_action[:3] * 2 + base_pos_world
                                        ,orientation    = target_action[3:])
                '''
        #-------- Prediction model (end) ---------------------------------------------------------# 
        
        
        #-----RDT-1B (start)--------------#
        """
        t2 = time.perf_counter()
        if captured_img:
            # print("UPDATING OBSERVATION WINDOW")
                       
            update_observation_window(config=config,imgs=captured_img, joint_state=current_q, EEF_position=target_pos)
            t3 = time.perf_counter()
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
                proprio = torch.cat([observation_window[-1]['qpos'], observation_window[-1]['EEF_position']], dim=0)
                
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
                interp_actions = interpolate_joints(args, pre_action, action)
                # interp_actions = interpolate_EEF(args, pre_action[:3], action[:3])
            else:
                interp_actions = action[np.newaxis, :]
            # Execute the interpolated actions one by one
            
            # print(f"interp actions shape= {interp_actions.shape}")
            # print(f"interp actions= {interp_actions}")
            '''for act in interp_actions:
                # print(f"act = {act}")
                executed_action     = act[:8]
                executed_action[6]  = current_q[6]
                executed_action[7]  = current_q[7]
                executed_action     = joint_mins + executed_action * joint_ranges_vals
                print(f"output action: {executed_action}")'''
            for act in interp_actions:
                
                delta_q   = act[:7]
                gripper_left    = act[7]
                gripper_right   = act[8]
                target_action = act[9:12] * 50 + [0,0, 0.125]
                
                
                # print(f"gripper open : {gripper_left} : {gripper_right}")
                # print(f"target action: {target_action}")
                # print(f"type of target pos : {type(target_pos)}")
                
                
                # if args.use_robot_base:
                #     vel_action = act[14:16]
            
            pre_action = action.copy()
            # print(f"acton = {action.shape}")
            
            '''
            # delta_q = delta_q / JOINT_GAIN 
            executed_action = current_q
            # executed_action[:7] = executed_action[:7] + delta_q
            executed_action[:7] = delta_q * 20
            executed_action[7]  = gripper_left
            executed_action[8]  = gripper_left
            
            rdt_action     = ArticulationAction(joint_positions=executed_action)
            # grip_action_1   = ArticulationAction(joint_positions=current_q)
            robot.apply_action(rdt_action'''
            Model_target_marker.set_world_pose(position = target_action + base_pos_world
                                               ,orientation    = target_rot)
        """
        #-----RDT-1B (end)----------------#
        
        
        
        # t5 = time.perf_counter()
        # t3 = time.perf_counter()
            
        # if thumbstick is not None:
        print(f"alpha_pos = {alpha_pos}, alpha_rot = {alpha_rot}", end='\r', flush=True)
        # print(SAVE + "\033[1A" + CLEAR + f"alpha = {alpha_pos:.3f}, beta = {alpha_rot:.3f}, save the video : {record}" + RESTORE , end="", flush=True)
        
        # target_pos = np.array([0.5,0.5,0.5])
        VR_target_marker.set_world_pose(position    = target_pos + base_pos_world
                                    ,orientation    = target_rot)
        # Model_target_marker.set_world_pose(position = ee_pos_world )
        if target_action is not None:
            target_pos = target_pos * (1 - alpha_pos) + target_action[:3] * alpha_pos
            # interpolated_quat = slerp(target_rot, target_quat, alpha_rot)
            # target_rot = interpolated_quat / torch.norm(interpolated_quat)

        # print("total_rotation:", target_rot)
        # target_rot = torch.from_numpy(target_rot).float()
        
        
        # print("shape of goal_rot:", goal_rot.shape)
        # print("type  of goal_rot:", type(goal_rot))
        
        prev_trigger_on = trigger_on 
        
        
        
        action_target, success = kin_solver.compute_inverse_kinematics(
                target_position     = target_pos,
                target_orientation  = target_rot
            )

        # t4 = time.perf_counter()
        if success:
            print(f"target_pose = {target_pos + base_pos_world}")
            # print(action_target)
            # print(f"base_pos_world = {base_pos_world}")
            # print(f"piper_hand_pos = {ee_pos_world}")
            # print(f"position difference = {target_pos * 0.5 + base_pos_world - ee_pos_world }")
            #alpha = 0.05  # smoothing
            #q = (1 - alpha) * current_q + alpha * action_target
            robot.apply_action(action_target)
            
        t+=1
        # t5 = time.perf_counter()
        if t1 and t2 and t3 and t4 and t5:
            print(f"t1-t0={(t1-t0)*1000:.2f}, t2-t1={(t2-t1)*1000:.2f}, t3-t2={(t3-t2)*1000:.2f}, t4-t3={(t4-t3)*1000:.2f}, t5-t4={(t5-t4)*1000:.2f}")
        # Reset the position of the arm back to starting position
        if buttonA:
            # print("RESETTING THE ENVIRONMENT")
            t = 0
            # Release all VideoWriters before moving/deleting files
            # for cam in range(len(rgb_cams)):
            #     writers[cam].release()
            sim.stop()
            sim.reset()
            set_new_poses()
            base.initialize()
            robot.initialize()
            franka_hand.initialize()
            # linkEndEffector.initialize()
            # target_pos = np.array([0.14,-0.03,2.5]) - base_pos_world
            
            if record:
                print(f"Saving successful Ep {ep_tracker}")
                for cam in range(len(rgb_cams)):
                    shutil.move(f"rgb_{cam}.avi", f"videos/cam_{cam}/rgb_ep_{ep_tracker}.avi")
                    # writers[cam]= create_video_writer(f"rgb_{cam}", frequency, width, height)
                ep_tracker += 1
            else:
                for cam in range(len(rgb_cams)):
                    try:
                        os.remove(f"rgb_{cam}.avi")
                        # writers[cam]= create_video_writer(f"rgb_{cam}", frequency, width, height)
                    except FileNotFoundError:
                        pass
            
            sim.play()
            sim.step(render=True)
            record = False
            continue
        
    
simulation_app.close()
client.loop_stop()