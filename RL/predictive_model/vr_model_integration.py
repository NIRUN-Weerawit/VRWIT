"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

piper Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of piper robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

import collections
import json
import os
import pickle
import shutil
import cv2
import csv
import h5py
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from torch.utils.tensorboard import SummaryWriter
from paho.mqtt import client as mqtt_client 
from threading import Lock
import math
import numpy as np
import torch
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import time
import copy
from policy import ACTPolicy
from einops import rearrange
from torchvision import transforms
import matplotlib.pyplot as plt

torch.set_printoptions(precision=4, sci_mode=False)


# Add custom arguments
custom_parameters = [
    {"name": "--controller",    "type": str,    "default": "ik",     "help": "Controller to use for piper. Options are {ik, osc}"},
    {"name": "--num_envs",      "type": int,    "default": 1,        "help": "Number of environments to create"},
    {"name": "--num_box",       "type": int,    "default": 1,        "help": "Number of boxes in the environment"},
    {"name": "--record",        "type": bool,   "default": False,    "help": "Whether to record or not"},
    {"name": "--seed",          "type": int,    "default": 10,       "help": "Set random seed for the simulation"},
    {"name": "--headless",      "type": bool,   "default": False,    "help": "Set random seed for the simulation"},
    {"name": "--alpha",         "type": float,  "default": 0.5,      "help": "Set random seed for the simulation"},
    {"name": "--temp",          "type": bool,   "default": False,    "help": "Controller to use for piper. Options are {ik, osc}"},
    {"name": "--topic",         "type": str,    "default": "",       "help": "Controller to use for piper. Options are {ik, osc}"},
    {"name": "--i",             "type": int,    "default": 0,        "help": "Number of boxes in the environment"},
    {"name": "--chunk_size",    "type": int,    "default": 25,       "help": "Number of boxes in the environment"},
    {"name": "--ver",           "type": str,    "default": "01",     "help": "Number of boxes in the environment"},
    {"name": "--delay",         "type": int,    "default": 0,        "help": "Controller to use for piper. Options are {ik, osc}"},
    {"name": "--name",           "type": str,   "default": "wee",    "help": "Number of boxes in the environment"},
    {"name": "--mode",           "type": str,   "default": "A",      "help": "Number of boxes in the environment"},
]   
args = gymutil.parse_arguments(
    description="Piper Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)
# ANSI shorthands
SAVE = "\033[s"
RESTORE = "\033[u"
CLEAR = "\033[K"
writer = SummaryWriter("logs_exp")
broker = "sora2.uclab.jp"
port = 1883
client_id = 'PiPER-control-wee'
DELAY = args.delay > 1

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
recv_messages = collections.deque(maxlen=args.delay)
old_time = time.time()
avg = 0.0
latency = 0
acc_latency = 0
avg_latency = 0
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
            # print(f"size of the recv_messages :{len(recv_messages)}" )
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
            vr_goal_pos     = [data_pos['x'], data_pos['y'], data_pos['z']]
            controller_obj  = [data_controller['_z'], data_controller['_x'], data_controller['_y']]
            vr_goal_rot     = data_rot       
        # print("gripper", gripper)
        # print("type", type(vr_joints))
        # print("vr_joint", vr_joints)
        # print("vr_goal_pos", vr_goal_pos)
        # print("vr_goal_rot", vr_goal_rot)
        # print('len(vr):', len(vr_joints))
        # print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

    client.subscribe(topic)
    client.on_message = on_message
    
# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments
topic = f"control/{args.topic}"

# fixed parameters
state_dim   = 8 # 14
action_dim  = 7
lr_backbone = 1e-5
backbone    = 'resnet18'
enc_layers  = 4
dec_layers  = 7
nheads      = 8
temporal_agg    = args.temp
chunk_size  = args.chunk_size
max_timesteps = 4000
policy_config = {'lr': 1e-4,
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
camera_names   = policy_config["camera_names"]
ckpt_dir = f"checkpoint_{chunk_size}_{args.ver}"
# ckpt_dir = "checkpoint_25_01"
# ckpt_dir = "/media/ucluser/PortableSSD/checkpoint_x"
# ckpt_dir = "checkpoint_x"
# ckpt_name = f'25_best.ckpt'
# ckpt_name = f'policy_step_15000_seed_28.ckpt'
ckpt_name = f'policy_best.ckpt'
# storage_dir = f"/media/ucluser/PortableSSD/raw_datasets/success_seed_{args.seed}"
storage_dir = f"raw_datasets/success_seed_{args.seed}"
# set random seed
np.random.seed(args.seed)

ckpt_path = os.path.join(ckpt_dir, ckpt_name)
policy = ACTPolicy(policy_config)
loading_status = policy.deserialize(torch.load(ckpt_path))
print(loading_status)
policy.cuda()
policy.eval()
print(f'Loaded: {ckpt_path}')
stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)
pre_process = lambda s_obs: (s_obs - stats['obs_mean']) / stats['obs_std']
post_process = lambda a: a * stats['action_std'] + stats['action_mean']
query_frequency = policy_config['num_queries']

if temporal_agg:
    end_time = 0.99 * max_timesteps
    query_frequency = 1
    num_queries = policy_config['num_queries']
    all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, action_dim]).cuda()
else:
    end_time = max_timesteps * 2
print(f"query_frequnecy: {query_frequency}")

# Grab controller
controller  = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
device      = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params                  = gymapi.SimParams()
sim_params.up_axis          = gymapi.UP_AXIS_Z
sim_params.gravity          = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt               = 1.0 / 60.0
sim_params.substeps         = 4
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type                    = 1
    sim_params.physx.num_position_iterations        = 8
    sim_params.physx.num_velocity_iterations        = 1
    sim_params.physx.rest_offset                    = 0.0
    sim_params.physx.contact_offset                 = 0.001
    sim_params.physx.friction_offset_threshold      = 0.001
    sim_params.physx.friction_correlation_distance  = 0.0005
    sim_params.physx.num_threads                    = args.num_threads
    sim_params.physx.use_gpu                        = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

"""elif args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8"""




# Set controller parameters
# IK params
damping = 0.05

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
# key_intensity  = gymapi.Vec3(1.2, 1.2, 1.2)   # RGB intensity (1≈100 %)
# ambient        = gymapi.Vec3(0.3, 0.3, 0.3)   # soft fill so shadows are visible
# direction      = gymapi.Vec3(-0.5, -1.0, -2.0)  # x,y,z → points down toward the ground

if sim is None:
    raise Exception("Failed to create sim")

# create viewer
if not args.headless:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")

asset_root = "../../../../assets"

# create table asset
table_dims                          = gymapi.Vec3(0.6, 1.0, 0.01)
table_asset_options                 = gymapi.AssetOptions()
table_asset_options.fix_base_link   = True
table_asset_options.armature        = 0.01
table_asset                         = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)
table_pose                          = gymapi.Transform()
table_pose.p                        = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)
print("Creating asset '%s' " % ("table"))

# create box asset
box_size                    = 0.04
box_dim                     = [box_size] * 3
box_asset_options           = gymapi.AssetOptions()
box_asset_options.density   = 1500
box_asset_options.armature  = 0.01
box_asset                   = gym.create_box(sim, box_size, box_size, box_size, box_asset_options)
box_pose                    = gymapi.Transform()
box_pose_np                 = np.array((0,0,0))
print("Creating asset '%s' " % ("box1"))

# create tray asset
tray_dim                            = [0.15, 0.15, 0.005] #small
# tray_dim = [0.2, 0.2, 0.01] #original
tray_color                          = gymapi.Vec3(0.24, 0.35, 0.8)
tray_asset_file                     = "urdf/tray/traybox_smaller.urdf"
tray_asset_options                  = gymapi.AssetOptions()
tray_asset_options.armature         = 0.01
tray_asset_options.density          = 8000
tray_asset_options.override_inertia = True
tray_asset                          = gym.load_asset(sim, asset_root, tray_asset_file, tray_asset_options)
corner                              = table_pose.p - table_dims * 0.5
x           = 0.3 #corner.x  # + table_dims.x * 0.2
y           = 0.1 #corner.y + table_dims.y * 0.8
z           = tray_dim[2]
tray_pose   = gymapi.Transform()
tray_pose.p = gymapi.Vec3(x, y, z)

print("Loading asset '%s' from '%s'" % (tray_asset_file, asset_root))


# Load piper asset
piper_asset_file                        = "urdf/piper_description/urdf/piper_description.urdf"
cube_asset_file                         = "urdf/cube.urdf"
asset_options                           = gymapi.AssetOptions()
asset_options.fix_base_link             = True
asset_options.flip_visual_attachments   = True
asset_options.armature                  = 0.01
# print("Loading asset '%s' from '%s'" % (piper_asset_file, asset_root))
piper_asset = gym.load_asset(sim, asset_root, piper_asset_file, asset_options)

# get joint limits and ranges for piper
piper_dof_props     = gym.get_asset_dof_properties(piper_asset)
piper_lower_limits  = piper_dof_props['lower']
piper_upper_limits  = piper_dof_props['upper']
piper_ranges        = piper_upper_limits - piper_lower_limits
piper_mids          = 0.5 * (piper_upper_limits + piper_lower_limits)
piper_num_dofs      = len(piper_dof_props)

    
piper_dof_props["driveMode"][:6] = gymapi.DOF_MODE_POS #DOF_MODE_POS
piper_dof_props["stiffness"][:6].fill(100000.0)
piper_dof_props["damping"][:6].fill(40.0)
# grippers
piper_dof_props["driveMode"][6:8] = gymapi.DOF_MODE_VEL        # gymapi.DOF_MODE_EFFORT #DOF_MODE_POS   
piper_dof_props["stiffness"][6:].fill(100000.0)
piper_dof_props["damping"][6:].fill(40.0)

# default dof states and position targets
piper_num_dofs          = gym.get_asset_dof_count(piper_asset)
default_dof_pos         = np.zeros(piper_num_dofs, dtype=np.float32)
default_dof_pos[:6]     = piper_mids[:6]
# grippers open
default_dof_pos[6:]     = piper_upper_limits[6:]

default_dof_state           = np.zeros(piper_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"]    = default_dof_pos

# send to torch
default_dof_pos_tensor  = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
piper_link_dict         = gym.get_asset_rigid_body_dict(piper_asset)
piper_hand_index        = piper_link_dict["piper_hand"]
print("piper hand index", piper_hand_index)

# set rigid bodies' properties
# piper_rigid_props       = gym.get_asset_rigid_shape_properties(piper_asset)
# print(f"PIPER PROPS: {piper_rigid_props}")
box_rigid_props         = gym.get_asset_rigid_shape_properties(box_asset)
p                       = box_rigid_props[0]
p.friction         = 10.0    # static/dynamic friction (try 0.5→2.0)
p.rolling_friction = 5.0    # prevents it from rolling out of grip
p.restitution      = 0.0    # no bounce
p.compliance       = 0.0    # fully stiff contacts
p.contact_offset   = 0.001
p.rest_offset      = 0.0
gym.set_asset_rigid_shape_properties(box_asset, box_rigid_props)
# configure env grid
num_envs    = args.num_envs
num_box     = args.num_box
num_per_row = int(math.sqrt(num_envs))
spacing     = 1.0
env_lower   = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper   = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

piper_pose     = gymapi.Transform()
piper_pose.p   = gymapi.Vec3(0, 0, 0)

table_pose      = gymapi.Transform()
table_pose.p    = gymapi.Vec3(0.35, 0.0, 0.5 * table_dims.z )

box_pose        = gymapi.Transform()

camera_handles          = []
camera_props            = gymapi.CameraProperties()
camera_props.width      = 640
camera_props.height     = 480
camera_1_position       = gymapi.Vec3(0.75, 0.0, 0.46)
camera_1_target         = gymapi.Vec3(0, 0, 0.03)
# camera_2_position       = gymapi.Vec3(0.4, - 0.5, 0.6)
# camera_2_target         = gymapi.Vec3(0, 0, 0)

# Attractor setup
attractor_handles = []
attractor_properties = gymapi.AttractorProperties()
attractor_properties.stiffness = 5e5
attractor_properties.damping = 5e3
# Make attractor in all axes
attractor_properties.axes = gymapi.AXIS_ALL
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
# Create helper geometry used for visualization
# Create a vr-wireframe axis
axes_geom = gymutil.AxesGeometry(0.1)
# Create a vr-wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 0, 0))

pose_model = gymapi.Transform()
pose_model.p = gymapi.Vec3(0, 0.0, 0.0)
pose_model.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Create a model-wireframe axis
model_axes_geom = gymutil.AxesGeometry(0.1)
# Create a model-wireframe sphere
model_sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
model_sphere_pose = gymapi.Transform(r=model_sphere_rot)
model_sphere_geom = gymutil.WireframeSphereGeometry(0.02, 15, 15, model_sphere_pose, color=(0, 1, 0))

# create a shadow box wireframe 
shadow_rot  = gymapi.Quat.from_euler_zyx(math.pi * 0.5, 0,0 )
shadow_pose = gymapi.Transform(r=shadow_rot)
shadow_geom = gymutil.WireframeBoxGeometry(0.05, 0.05, 0.01, model_sphere_pose, color=(0.1,0.1,0.1))

contact_alert_goem_1 = gymutil.WireframeSphereGeometry(0.009, 40, 40, model_sphere_pose, color=(1, 0, 0))
contact_alert_goem_2 = gymutil.WireframeSphereGeometry(0.009, 40, 40, model_sphere_pose, color=(1, 0, 0))

local_transform         = gymapi.Transform()
local_transform.p       = gymapi.Vec3(0.0,0.059,0.035)
# local_transform.r       = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.1,-0.85,-0.1), np.radians(90.0))
# local_transform.r       = gymapi.Quat.from_euler_zyx(-1.5, -1.6,0)
r1 = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0),  np.radians( 20))   # yaw
r2 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0),  np.radians( -90))   # pitch
r3 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1),  np.radians( -90))   # pitch
rot = r1 *r3 * r2         # NOTE: Gym quaternions multiply right‑to‑left            
# rot *= r3          
local_transform.r       = rot

envs            = []
tray_handles    = []
piper_handles   = []
box_handles     = []
tray_idxs       = []
box_idxs        = []
hand_idxs       = []
claws_idxs      = []
init_pos_list   = []
init_rot_list   = []

# add ground plane
plane_params        = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)
unfinished_color = gymapi.Vec3(0.14, 0.72, 0.83)
finished_color = gymapi.Vec3(0.4, 1, 0.4)


def random_box_pose():
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.1)
    box_pose.p.y = table_pose.p.y + np.random.uniform(-0.2, 0.2)
    box_pose.p.z = table_dims.z + 0.5 * box_size
    # box_pose_np     = np.array([box_pose.p.x,box_pose.p.y,box_pose.p.z])
    # init_box_pose = np.zeros(1, dtype=gymapi.RigidBodyState.dtype)
    # init_box_pose['pose']['p'][0] =(box_pose.p.x, box_pose.p.y, box_pose.p.z)
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    return box_pose

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    # add table
    # table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add tray
    tray_handle     = gym.create_actor(env, tray_asset, tray_pose, "tray", i, 0)
    tray_handles.append(tray_handle)
    gym.set_rigid_body_color(env, tray_handles[i], 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
    # get global index of tray in rigid body state tensor
    tray_idx        = gym.get_actor_rigid_body_index(env, tray_handle, 0, gymapi.DOMAIN_SIM)
    tray_idxs.append(tray_idx)
    
    # add box
    box_handles.append([])
    box_idxs.append([])
    
    for n in range(num_box):
        box_handle          = gym.create_actor(env, box_asset, random_box_pose(), "box_" + str(n), i, 0)
        unfinished_color    = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, unfinished_color) #color
        box_handles[i].append(box_handle)
        
        # get global index of box in rigid body state tensor
        box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
        box_idxs[i].append(box_idx)
    
    

    # add piper
    piper_handle = gym.create_actor(env, piper_asset, piper_pose, "piper", i, 2)
    piper_handles.append(piper_handle)
    
    # set dof properties
    gym.set_actor_dof_properties(env, piper_handle, piper_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, piper_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, piper_handle, default_dof_pos)

    # get box props
    box_props = gym.get_actor_rigid_shape_properties(env, box_handle)
    print(f"BOX HANDLE PROPS: ", box_props[0].friction, box_props[0].rolling_friction)
    
    # get inital hand pose
    hand_handle     = gym.find_actor_rigid_body_handle(env, piper_handle, "piper_hand")  #"") LinkEndEffector
    hand_pose       = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([0.1, 0.1, 0.3])
    # init_pos_list.append([tray_pose.p.x, tray_pose.p.y, tray_pose.p.z])
    init_rot_list.append([-0.95, -0.25, 0.0, 0.0])
    # init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    # init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
    
    # get global index of hand in rigid body state tensor
    hand_idx        = gym.find_actor_rigid_body_index(env, piper_handle, "piper_hand", gymapi.DOMAIN_SIM) #    LinkEndEffector
    hand_idxs.append(hand_idx)
    
    claw_left_idx   = gym.find_actor_rigid_body_index(env, piper_handle, "link7", gymapi.DOMAIN_SIM) 
    claw_right_idx  = gym.find_actor_rigid_body_index(env, piper_handle, "link8", gymapi.DOMAIN_SIM) 
    claws_idxs.append([claw_left_idx, claw_right_idx])
    
    
    # add camera
    cam_1 = gym.create_camera_sensor(env, camera_props)
    cam_2 = gym.create_camera_sensor(env, camera_props)
    #set the location of camera sensor
    gym.set_camera_location(cam_1, env, camera_1_position, camera_1_target)
    # gym.set_camera_location(cam_2, env, camera_2_position, camera_2_target)
    cam_pos     = gym.find_actor_rigid_body_handle(env, piper_handle, "piper_hand")  #"piper_hand")
    gym.attach_camera_to_body(cam_2, env, cam_pos   , local_transform, gymapi.FOLLOW_TRANSFORM)
    # gym.attach_camera_to_body(camera_handle, env, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
    
    camera_handles.append([])
    camera_handles[i].append(cam_1)
    camera_handles[i].append(cam_2)
    
    props = gym.get_actor_rigid_body_states(env, piper_handle, gymapi.STATE_POS)
    body_dict = gym.get_actor_rigid_body_dict(env, piper_handle)
    # Initialize the attractor
    attractor_properties.target = props['pose'][:][body_dict["piper_hand"]]
    attractor_properties.target.p.y = 0.0 #-= 0.05
    attractor_properties.target.p.z = 0.0 #0.03
    attractor_properties.target.p.x = 0.0 #0.1 
    attractor_properties.rigid_handle = hand_handle

    # Draw axes and sphere at attractor location
    gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

    gymutil.draw_lines(model_axes_geom, gym, viewer, env, attractor_properties.target)
    gymutil.draw_lines(model_sphere_geom, gym, viewer, env, attractor_properties.target)
    shadow_pos =attractor_properties.target
    shadow_pos.p.z = 0.0
    gymutil.draw_lines(shadow_geom, gym, viewer, env, shadow_pose)
    
    piper_handles.append(piper_handle)
    # attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)
    # attractor_handles.append(attractor_handle)
    

unfinished_box_idxs = copy.deepcopy(box_idxs)
finished_box_idxs   = []

cam_pos         = gymapi.Vec3(1.0, 0.0, 0.6)
cam_target      = gymapi.Vec3(0.0, 0.0, 0.0)
middle_env      = envs[num_envs // 2 + num_per_row // 2]
if not args.headless: gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# box corner coords, used to determine grasping yaw
box_half_size   = 0.5 * box_size
corner_coord    = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners         = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir        = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base piper, tensor has shape (num envs, 10, 6, 9)
_jacobian       = gym.acquire_jacobian_tensor(sim, "piper")
jacobian        = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to piper hand
j_eef           = jacobian[:, piper_hand_index - 1, :, :6]

# get mass matrix tensor
_massmatrix     = gym.acquire_mass_matrix_tensor(sim, "piper")
mm              = gymtorch.wrap_tensor(_massmatrix)
mm              = mm[:, :6, :6]          # only need elements corresponding to the piper arm

# get contact tensor
_net_cf = gym.acquire_net_contact_force_tensor(sim)
net_cf = gymtorch.wrap_tensor(_net_cf)


def find_closest_box_idx(unfinished_box_idxs):
    # unfinished_box_idxs: list of lists, each sublist contains box indices for that env
    boxes_pose = []
    for idxs in unfinished_box_idxs:
        # idxs is a list of box indices for this env
        if len(idxs) == 0:
            # If no unfinished boxes, append a dummy row (will be ignored)
            boxes_pose.append(torch.full((1, 3), float('inf'), device=rb_states.device))
        else:
            boxes_pose.append(rb_states[idxs, :3])
    # boxes_pose: list of [num_unfinished, 3] tensors
    # Pad to max length for stacking
    max_boxes       = max(x.shape[0] for x in boxes_pose)
    padded_boxes    = []
    for x in boxes_pose:
        if x.shape[0] < max_boxes:
            pad     = torch.full((max_boxes - x.shape[0], 3), float('inf'), device=rb_states.device)
            x       = torch.cat([x, pad], dim=0)
        padded_boxes.append(x)
    boxes_pose      = torch.stack(padded_boxes)  # [num_envs, max_boxes, 3]
    norms           = torch.norm(boxes_pose, dim=2)
    min_indices     = torch.argmin(norms, dim=1)
    closest_box_idx = []
    for env_i, idx in enumerate(min_indices):
        if len(unfinished_box_idxs[env_i]) == 0:
            closest_box_idx.append(-1)  # or some invalid index
        else:
            closest_box_idx.append(unfinished_box_idxs[env_i][idx.item()])
    closest_box_idx = torch.tensor(closest_box_idx, device=rb_states.device)
    return closest_box_idx  #, min_indices

# get rigid body state tensor
_rb_states      = gym.acquire_rigid_body_state_tensor(sim)
rb_states       = gymtorch.wrap_tensor(_rb_states)
# print("rb_state shape", rb_states.shape)
# print("rb_state shape", rb_states.shape[0])
# print("rb_state shape", rb_states.shape[1])

rb_states_clone = rb_states.clone()
# print("device: ", rb_states_clone.device)
# print("rb_states_clone[tray_idxs, :] =", rb_states_clone[tray_idxs])
init_tray_rot   = rb_states_clone[tray_idxs, 3:7]
# print("box_ids:", box_idxs)
# print("box_ids:", len(box_idxs))
# print("init_cube_rot", init_cube_rot.shape)
# closest_box_idxs    = find_closest_box_idx(box_idxs)
# print("closest box_ids:", closest_box_idxs)
# print("closest box_ids:", len(closest_box_idxs))
find_new_idx        = torch.full((num_envs,), True, dtype=torch.bool, device=device)
closest_box_idxs    = find_closest_box_idx(unfinished_box_idxs)


# get dof state tensor
_dof_states     = gym.acquire_dof_state_tensor(sim)
dof_states      = gymtorch.wrap_tensor(_dof_states)  #torch.Size([num_envs * 8, 2]) pos & vel
dof_pos         = dof_states[:, 0].view(num_envs, 8, 1) #shape = [num_envs, 8, 1]
# dof_pos_buffer = torch.zeros_like(dof_pos, device=device).squeeze(2).unsqueeze(1)  # want shape = [num_envs, 1, 8]
# dof_pos_buffer = [] * num_envs # a list of tensors of size (num_envs)
dof_pos_buffer  = [torch.empty((0, 8), device=device) for _ in range(num_envs)] # a list of tensors of size (num_envs)

# print("dof_pos", dof_pos.shape)
# print("dof_pos buffer", dof_pos_buffer)
# print("dof_pos buffer", dof_pos_buffer.shape)
dof_vel         = dof_states[:, 1].view(num_envs, 8, 1)
init_dof_states = dof_states.clone().view(num_envs, 8, 2)

# get dof force tensor
# _dof_forces     = gym.acquire_dof_force_tensor(sim)
# dof_forces      = gymtorch.wrap_tensor(_dof_forces)

# Create a tensor noting whether the hand should return to the initial position
hand_restart    = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action      = torch.zeros_like(dof_pos).squeeze(-1)

hand_pos_buffer = [torch.empty((0, 7), device=device) for _ in range(num_envs)]
effort_action   = torch.zeros_like(pos_action)
t = 0
return_pos      = [0.1, 0.1, 0.3]
return_rot      = [-0.95, -0.25, 0.0, 0.0]
# prev_hand_vel = torch.zeros_like(rb_states[hand_idxs, 7:]).to(device)
prev_hand_pos   = torch.zeros_like(rb_states[hand_idxs, :3]).to(device)
# Create a tensor noting whether the env should restart
envs_restart    = torch.full([num_envs], False, dtype=torch.bool).to(device)
froze_count     = torch.zeros(num_envs).to(device)
frozen_threshold        = 0.1
# Convert to tensors
tray_dim_tensor = torch.tensor(tray_dim, device=device)
box_dim_tensor = torch.tensor(box_dim, device=device)
r_min = 0.2
r_max = 0.4
theta = 70
# Add these before the simulation loop
success_counter = 0
success_count_each_env = torch.zeros(num_envs, device=device)
# success_count_each_env =  [0.]* num_envs
time_envs = torch.full((num_envs,), 0, dtype=int, device=device)

def random_pos_tensor():
    """
        Return: torch.cat((tray_pose, boxes_pose), dim=1)
    """
    radius_tensor   = torch.FloatTensor(num_envs, num_box + 1).uniform_(r_min, r_max).to(device)       # num_envs x num_objs (3)
    theta_tensor    = torch.FloatTensor(num_envs, num_box + 1).uniform_(- theta/180 * np.pi, theta/180 * np.pi).to(device)   
    x_tensor   = radius_tensor * torch.cos(theta_tensor)           # num_envs x 1
    y_tensor   = radius_tensor * torch.sin(theta_tensor)           # num_envs x 1
    z_tensor_tray   = torch.full([num_envs], table_dims.z).to(device)                   # num_envs x 1
    z_tensor_cube   = torch.full([num_envs], table_dims.z + 0.5 * box_size).to(device)   # num_envs x 1
    
    # random_pose     = torch.stack((torch.stack([x_tensor[:, 0], y_tensor[:, 0], z_tensor_tray], dim=1), 
    #                                torch.stack([x_tensor[:, 1], y_tensor[:, 1], z_tensor_cube], dim=1)), dim=1).to(device)  
    tray_pose    = torch.stack([x_tensor[:, 0], y_tensor[:, 0], z_tensor_tray], dim=1).unsqueeze(1)
    boxes_pose  = torch.stack([torch.stack([x_tensor[:, j+1], y_tensor[:, j+1], z_tensor_cube], dim=1) for j in range(num_box)], dim=1)
    

    random_pose     = torch.cat((tray_pose, boxes_pose), dim=1)

    
    return random_pose
    
def reset(restart_envs, dof_state):
    
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    # print("Some environments will be restarted:", torch.where(restart_envs)[0])
    restart_indices = torch.where(restart_envs)[0].to(rb_states.device)
    states_buffer                   = dof_state.clone().view(num_envs, 8, 2)
    states_buffer[restart_indices]  = init_dof_states[restart_indices]
    dof_state                       = states_buffer.view(num_envs * 8,2)
    random_pose                     = random_pos_tensor()

    tray_reset_idxs                 = torch.tensor(tray_idxs, device=rb_states.device)[restart_indices]
    box_reset_idxs                  = torch.tensor(box_idxs, device=rb_states.device)[restart_indices]

    rb_states[tray_reset_idxs, :3]  = random_pose[restart_indices, 0, :3]
    rb_states[tray_reset_idxs, 3:7] = torch.tensor([0., 0., 0., 1.0], device=device)
    # for n in range(num_box):
    # rb_states[box_reset_idxs, :3]   = random_pose[restart_indices, 1:num_box + 1, :3]
    rb_states[box_reset_idxs, :3]   = random_pose[restart_indices, 1:num_box + 1, :3] #.reshape(-1, 3)
    

    gym.set_rigid_body_state_tensor(sim, gymtorch.unwrap_tensor(rb_states))
    
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
    


def store_joints_states(env: int, ep: int, observation: object, action: object, success: bool):
    """file = h5py.File("dataset/joint_angles.h5py", 'w')
    group = file.create_group(f"env_{env}_ep_{ep}")
    dataset = file.store_joints_states()"""
    
    with open(f"{storage_dir}/env_{env}/observations/env_{env}_ep_{ep}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(observation.numpy())
    with open(f"{storage_dir}/env_{env}/actions/env_{env}_ep_{ep}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(action.numpy())    


def create_video_writer(env, old_writers): 
    """
        create a new pair of video writers in an envronment
        Return: color_writer_1, color_writer_2 
    """
    for writer in old_writers:
        writer.release()
    
    color_writer_1  = cv2.VideoWriter(f'color_1_env_{env}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    color_writer_2  = cv2.VideoWriter(f'color_2_env_{env}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    depth_writer_1  = cv2.VideoWriter(f'depth_1_env_{env}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    depth_writer_2  = cv2.VideoWriter(f'depth_2_env_{env}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    return color_writer_1, color_writer_2, depth_writer_1, depth_writer_2

def store_depth_data(env, ep, depth_1, depth_2):
    with h5py.File(f"{storage_dir}/env_{env}/depths/episode_{ep}.hdf5", 'w') as f:
        """f.create_dataset('action', data=action_data) # if action is joint angles, then dim=[k x 6], else if action is goal pose, then dim = [k x 7] (3+4)
        
        obs_grp     = f.create_group('observations')
        
        obs_grp.create_dataset('obs', data=obs_data) # [k x 6]
        obs_grp.create_dataset('qvel', data=qvel_data) # [k x 6]
        
        img_grp     = f.create_group('images')
        img_grp.create_dataset('cam1', data=rgb_1_data) # [k x H x W x C] C=3 
        img_grp.create_dataset('cam2', data=rgb_2_data) # [k x H x W x C] C=3"""
        if "depths" not in f:
            f.create_group('depths')
        
        f["depths"].create_dataset('cam1', data=depth_1, compression="gzip", compression_opts=1) # [k x H x W x C] C=1
        f["depths"].create_dataset('cam2', data=depth_2, compression="gzip", compression_opts=1) # [k x H x W x C] C=1

depth_stack_1 = [[] for _ in range(num_envs)]
depth_stack_2 = [[] for _ in range(num_envs)]

def image_processing(writers):
    for i in range(num_envs):
        color_image_1   = gym.get_camera_image(sim, envs[i], camera_handles[i][0], gymapi.IMAGE_COLOR)
        color_image_2   = gym.get_camera_image(sim, envs[i], camera_handles[i][1], gymapi.IMAGE_COLOR)
        depth_image_1   = gym.get_camera_image(sim, envs[i], camera_handles[i][0], gymapi.IMAGE_DEPTH)
        depth_image_2   = gym.get_camera_image(sim, envs[i], camera_handles[i][1], gymapi.IMAGE_DEPTH)
        # print("depth_image1_shape",depth_image_1[50:60, 50:55])
        # print("depth_image2_shape",depth_image_2[50:60, 50:55])
        
        
        
        depth_image_1 = np.clip(depth_image_1, -3.0, -0.0)
        depth_image_2 = np.clip(depth_image_2, -3.0, -0.0)
        
        img_np_1 = color_image_1.reshape((camera_props.height, camera_props.width, 4))
        img_np_2 = color_image_2.reshape((camera_props.height, camera_props.width, 4))
        # depth_colormap_1 = cv2.convertScaleAbs(depth_image_1, alpha=1)
        
        
        # depth_norm_1 = (depth_image_1 - np.min(depth_image_1)) / (np.max(depth_image_1) - np.min(depth_image_1))
        # depth_norm_2 = (depth_image_2 - np.min(depth_image_2)) / (np.max(depth_image_2) - np.min(depth_image_2))
        # print(f"cam1:min = {np.min(depth_image_1)}, max= {np.max(depth_image_1)}")
        # print(f"cam2:min = {np.min(depth_image_2)}, max= {np.max(depth_image_2)}")
        
        # print("depth_image_shape",depth_colormap_1)
        # print("distance",depth_colormap_1[100][100])
        # depth_colormap_1 = cv2.convertScaleAbs(depth_image_1, alpha=30)
        # depth_colormap_2 = cv2.convertScaleAbs(depth_image_2, alpha=100)
        
        # writers[i][2].write(depth_image_1)
        # writers[i][3].write(depth_image_2)
        rgb_image_1 = img_np_1[:, :, :3]
        rgb_image_2 = img_np_2[:, :, :3]
        
        # rgb_image_1 = rgb_image_1.astype(np.uint8)    # Ensure type
        # rgb_image_2 = rgb_image_2.astype(np.uint8)    # Ensure type
        
        # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # print("color image", bgr)
        # print("color image shape", img_np.shape)
        cv2.imshow("Color1", rgb_image_1)
        cv2.imshow("Color2", rgb_image_2)
        # cv2.imshow("Depth_1",depth_colormap_1)
        # cv2.imshow("Depth_2",depth_colormap_2)
        cv2.waitKey(1)
        writers[i][0].write(rgb_image_1)
        writers[i][1].write(rgb_image_2)
        depth_stack_1[i].append(depth_image_1)
        depth_stack_2[i].append(depth_image_2)

def image_capture():
    for i in range(num_envs):
        
        
        color_image_1   = gym.get_camera_image(sim, envs[i], camera_handles[i][0], gymapi.IMAGE_COLOR)
        color_image_2   = gym.get_camera_image(sim, envs[i], camera_handles[i][1], gymapi.IMAGE_COLOR)
        depth_image_1   = gym.get_camera_image(sim, envs[i], camera_handles[i][0], gymapi.IMAGE_DEPTH)
        depth_image_2   = gym.get_camera_image(sim, envs[i], camera_handles[i][1], gymapi.IMAGE_DEPTH)
        
        
        
        
        depth_image_1 = np.clip(depth_image_1, -3.0, -0.0)
        depth_image_2 = np.clip(depth_image_2, -3.0, -0.0)
        
        """depth_colormap_1 = cv2.convertScaleAbs(depth_image_1, alpha=120)
        depth_colormap_2 = cv2.convertScaleAbs(depth_image_2, alpha=100)
        cv2.imshow("Depth_1",depth_colormap_1)
        cv2.imshow("Depth_2",depth_colormap_2)
        cv2.waitKey(1)"""
        
        depth_image_1 = depth_image_1[:, :, np.newaxis]  # shape: [H, W, 1]
        depth_image_2 = depth_image_2[:, :, np.newaxis]
        
        img_np_1 = color_image_1.reshape((camera_props.height, camera_props.width, 4))
        img_np_2 = color_image_2.reshape((camera_props.height, camera_props.width, 4))

        rgb_image_1 = img_np_1[:, :, :3]
        rgb_image_2 = img_np_2[:, :, :3]
        
        # cv2.imshow("color_1",rgb_image_1)
        cv2.imshow("color_2",cv2.cvtColor(rgb_image_2, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        rgbd_image_1    = np.concatenate((rgb_image_1, depth_image_1), axis=2)
        rgbd_image_2    = np.concatenate((rgb_image_2, depth_image_2), axis=2)
        
        
        
        return [rgbd_image_1, rgbd_image_2]  
      
writers = []

try:
    for env in range(num_envs):
        os.makedirs(f"{storage_dir}/env_{env}/observations" ,  exist_ok=True)
        os.makedirs(f"{storage_dir}/env_{env}/actions" , exist_ok=True)
        os.makedirs(f"{storage_dir}/env_{env}/depths" ,  exist_ok=True)
        os.makedirs(f"{storage_dir}/env_{env}/colors" ,  exist_ok=True)       
except FileExistsError:
    pass
if args.record:
    for j in range(num_envs):
        color_writer_1, color_writer_2, depth_writer_1, depth_writer_2 = create_video_writer(j)
        writers.append([color_writer_1, color_writer_2, depth_writer_1, depth_writer_2])

Eps_success     = [args.i] * num_envs
Eps_failure     = [0] * num_envs


image_list = [] # for visualization
obs_list = []
target_obs_list = []
rewards = []
alpha = args.alpha
alpha_pos = 0.0
alpha_rot = 0.0

def get_observations(dof_states):
    joint_positions     = dof_states[:, 0].view(num_envs, 8)
    # print(f"joint_positions = {joint_positions}")
    return joint_positions.tolist()

def get_image(ts, camera_names, stats):
    # print(f"stats = {stats}")
    curr_images = []
    depth_images = []
    for cam in range(len(camera_names)):
        curr_image  = rearrange(ts[cam], 'h w c -> c h w')
        # print(f" img shape {curr_image.shape}")
        curr_images.append(curr_image[:3])
        depth_images.append(curr_image[3])
    
    curr_image  = np.stack(curr_images,     axis=0)
    depth_image = np.stack(depth_images,    axis=0)
    # print(f"curr img shape {curr_image.shape}")
    for cam in range(len(camera_names)):
        depth_image[cam] = (depth_image[cam] - stats[f"depth_mean_cam{cam+1}"]) / stats[f"depth_std_cam{cam+1}"]
    
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)    
    depth_image = torch.from_numpy(depth_image).float().cuda().unsqueeze(1).unsqueeze(0)
    # print(f"depth img shape {depth_image.shape}")
    
    
    curr_image = torch.concatenate([curr_image, depth_image], axis = 2)
    # print(f"final img shape {curr_image.shape}")

    return curr_image

def control_gripper_velocity(piper_velocity_target):
    if grip_flag:
        piper_velocity_target[6]  = - 0.8
        piper_velocity_target[7]  =   0.8
    else:
        piper_velocity_target[6]  =   0.8
        piper_velocity_target[7]  = - 0.8
    velocity_tensor = torch.as_tensor(piper_velocity_target)
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(velocity_tensor))

def control_gripper_force(force_target):
    if grip_flag:
        force_target[6]  =  -20
        force_target[7]  =  20 
    else:
        force_target[6]  =  20
        force_target[7]  =  -20
    force_tensor = torch.as_tensor(force_target)
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(force_tensor))

import math
from typing import Tuple

def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
    return u

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def slerp(q0, q1, t):
    """
    Performs Spherical Linear Interpolation (SLERP) between two quaternions.

    Args:
        q0 (torch.Tensor): The starting quaternion (shape [4]).
        q1 (torch.Tensor): The ending quaternion (shape [4]).
        t (float): The interpolation factor, between 0.0 and 1.0.

    Returns:
        torch.Tensor: The interpolated quaternion.
    """
    q1 = torch.from_numpy(q1)
    # Calculate the dot product between the two quaternions
    dot = torch.dot(q0, q1)

    # If the dot product is negative, the quaternions are more than 90 degrees
    # apart. To take the shorter path, we need to flip one of them.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Set a threshold for when to fall back to linear interpolation (LERP)
    # to avoid division by zero and numerical instability near dot == 1.
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If the quaternions are very close, linearly interpolate and normalize.
        result = q0 + t * (q1 - q0)
        return result / torch.linalg.norm(result)

    # Standard SLERP formula
    theta_0 = torch.acos(dot)        # Angle between quaternions
    sin_theta_0 = torch.sin(theta_0) # Sine of the angle

    theta = theta_0 * t              # Angle for the interpolated quaternion
    sin_theta = torch.sin(theta)     # Sine of the new angle

    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q0) + (s1 * q1)

subscribe(client)
client.loop_start()
Q_VR2GYM    = gymapi.Quat(-0.70710678, 0.0, 0.0, 0.70710678)   # −90° about X (Y-up → Z-up)
# Q_GYM2VR    = Q_VR2GYM.inverse()
IDENTITY    = gymapi.Quat(0, 0.0, 0.0, 1.0) 
quat_save_gym = None
quat_start_gym = None
Q_HAND_OFFSET = gymapi.Quat.from_euler_zyx(math.pi,  math.pi, -0.665454952)
q_prev  =  None
vr_rot_old = np.zeros(3)
quat_start = None
quat_save = gymapi.Quat(-0.70710678, 0.0, 0.0, 0.70710678)
pos_save = gymapi.Vec3(0,0,0)
t = 0
goal_rot = torch.tensor([ 0.6416,  0.3907, -0.3102,  0.5826]).unsqueeze(0)
goal_pos = torch.tensor([ 0, 0,0 ]).unsqueeze(0)
prev_pos = gymapi.Vec3(0,0,0)
predicted_trajectories = []
actual_trajectores      = []
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()
fq = 0.01
success_tracker = 0
total_ep    = 0

while not gym.query_viewer_has_closed(viewer):
    tic = time.time()
    # read the oldest message from mqtt
    if DELAY:
        if len(recv_messages) > 0 and tic - old_time >= 0.00:
            # if avg / fq  > 3:
                # _ = recv_messages.popleft()
            mqtt_msg, mqtt_time= recv_messages.popleft()
            latency = time.time() - mqtt_time
            grip_flag   = json.loads(mqtt_msg)['grip']
            data_pos = json.loads(mqtt_msg)['goal_pos']
            data_controller = json.loads(mqtt_msg)['controller_object']
            trigger_on      = json.loads(mqtt_msg)['sending']
            vr_goal_pos     = [data_pos['x'], data_pos['y'], data_pos['z']]
            controller_obj  = [data_controller['_z'], data_controller['_x'], data_controller['_y']]
            
            # print(f"grip_flag:{grip_flag}, vr_goal_pos:{vr_goal_pos}, controller_obj:{controller_obj}")
            old_time = tic
            # print(f"trigger_on:{trigger_on}")

    
    
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_net_contact_force_tensor(sim)
    # gym.refresh_dof_force_tensor(sim)
    
    gym.clear_lines(viewer)
    
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)
     
    if torch.any(find_new_idx):
        new_box_idxs = find_closest_box_idx(unfinished_box_idxs)
        closest_box_idxs = torch.where(find_new_idx, new_box_idxs, closest_box_idxs)
        
        find_new_idx[:] = False
        
        
    # # Prediction part
    obs = get_observations(dof_states)
    
    obs_numpy = np.array(obs)
    
    obs = pre_process(obs_numpy)
    # print(f"shape obs = {obs.shape}")
    if len(obs.shape) > 1:
        obs = torch.from_numpy(obs).float().cuda()
    else:
        obs = torch.from_numpy(obs).float().cuda().unsqueeze(0)
        
    if t == 0:
        # warm up
        curr_image = get_image(image_capture(), camera_names, stats)
        for _ in range(10):
            # print(f"obs: {obs}, shape: {obs.shape}")
            policy(obs, curr_image)
        print('network warm up done')
        
    
    if t % query_frequency == 0:
    # if t % 2 == 0:  
        curr_image = get_image(image_capture(), camera_names, stats)
        with torch.no_grad():
            # print(f"time:{t}, obs= {obs}")
            all_actions = policy(obs, curr_image)
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            # trajectory = all_actions.squeeze(0).cpu().detach().numpy()
            # predicted_trajectories.append(post_process(trajectory))
            
            # print(f"all_actions size= {all_actions.shape}, query fre. = {query_frequency}")
            # print(f"time:{t}, all_actions = {all_actions}")
    if temporal_agg:
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
    raw_action = raw_action.squeeze(0).cpu().detach().numpy()
    # print(f"raw action= {raw_action}, type={type(raw_action)}")
    action = post_process(raw_action)
    # print(f"post_processed action= {action}, type={type(action)}, shape={action.shape}")
    # target_obs = action[:action_dim]
    target_action = action[:]
    predicted_trajectories.append(target_action)
        
    boxes_states = rb_states[closest_box_idxs, :]
    

    box_pos     =   boxes_states[:, :3]
    box_rot     =   boxes_states[:, 3:7]
    box_vel     =   boxes_states[:, 7:]

    tray_pos   = rb_states[tray_idxs, :3]
    tray_rot   = rb_states[tray_idxs, 3:7]
    # print("tray_pos:", tray_pos)
    # print("tray_pos:", tray_pos.shape)
    hand_pos = rb_states[hand_idxs, :3]
    # print(f"hand pos: {hand_pos}")
    hand_rot = rb_states[hand_idxs, 3:7]
    # print("hand_rot:", hand_pos)
    hand_vel = rb_states[hand_idxs, 7:]

    is_bounded_by_tray = (torch.abs(box_pos[:, :2] - tray_pos[:, :2]) < (tray_dim_tensor[:2]  -  box_dim_tensor[:2]) / 2)
 
    
    # Deploy actions
    # with vr_joints_lock:
    #     vr_goal_pos_copy    = vr_goal_pos.copy()
        
    #     # vr_rot          = np.array(vr_goal_rot.copy(), dtype=float)
    #     # vr_rot_new      = vr_goal_rot.copy()
    #     # quat_controller  = gymapi.Quat(*vr_goal_rot)
    #     # print(f"quat_controller = {quat_controller}")
    #     # vr_rot              = np.array([vr_goal_rot_copy['_x'], vr_goal_rot_copy['_y'], vr_goal_rot_copy['_z']], dtype=float)
    #     grip_flag_copy      = grip_flag
    for i in range(num_envs):
        # Update attractor target from current piper state
        pose = gymapi.Transform()
        
        props = gym.get_actor_rigid_body_states(envs[i], piper_handles[i], gymapi.STATE_POS)
        current_pos = props['pose'][:][body_dict["piper_hand"]][0]
        current_rot = props['pose'][:][body_dict["piper_hand"]][1]
        
        current_pos = gymapi.Vec3(*current_pos)
        
        actual_trajectores.append(np.array([current_pos.x, current_pos.y, current_pos.z]))
        current_rot = gymapi.Quat(*current_rot)
        # print(f"current_rot = {current_rot}")
        # print(f"current_pos = {current_pos}")
        # pose.p = gymapi.Vec3(-vr_goal_pos_copy[0], vr_goal_pos_copy[2], vr_goal_pos_copy[1])
        
        # pose.p = gymapi.Vec3(0.2, 0.2, 0.2)
        
        # piper_dof_states        = gym.get_actor_dof_states(envs[i], piper_handles[i], gymapi.STATE_POS)
        # end_effector            = piper_dof_states['pos']
        # print(f"row: {roll},    pitch:{pitch},  yaw:{yaw}")

            
        
        # vr_rot_np   = np.array([vr_goal_rot_copy[0], vr_goal_rot_copy[1], vr_goal_rot_copy[2], vr_goal_rot_copy[3]])
        
        # vr_goal_rot_copy = np.array([vr_goal_rot_copy['x'],
        #                          vr_goal_rot_copy['y'],
        #                          vr_goal_rot_copy['z'],
        #                          vr_goal_rot_copy['w']], dtype=np.float32)
        
        
        # current_rot = np.array([current_rot['x'],
        #                          current_rot['y'],
        #                          current_rot['z'],
        #                          current_rot['w']], dtype=np.float32)
        
        # vr_roll, vr_pitch, vr_yaw = quat_to_euler(*vr_rot)
        # ee_roll, ee_pitch, ee_yaw = quat_to_euler(*current_rot)
        # quat_controller  = gymapi.Quat(*vr_goal_rot)
        # print(f"controller_object = {controller_object}")
        
        # if t < 10000:
        #     controller_obj[0] = 0
        #     controller_obj[2] = 0
        # elif t < 2000:
        #     controller_obj[0] = 0
        #     controller_obj[2] = 0
        # elif t < 3000:
        #     controller_obj[1] = 0
        #     controller_obj[2] = 0
        # else:
            # t = 0
        # t += 1
        # print(t)
        quat_controller = gymapi.Quat.from_euler_zyx(*controller_obj)
        # pos_controller  = np.array([-vr_goal_pos_copy[0], vr_goal_pos_copy[2], vr_goal_pos_copy[1]])
        pos_controller  = gymapi.Vec3(vr_goal_pos[2], vr_goal_pos[0], vr_goal_pos[1])
        # quat_controller = gymapi.Quat(*vr_goal_rot)
        # print(f"pos_controller = {pos_controller}, quat_controller = {quat_controller}")
        
        if trigger_on and not prev_trigger_on:
            # props_1 = gym.get_actor_rigid_body_states(envs[i], piper_handles[i], gymapi.STATE_POS)
            # quat_start = props_1['pose'][:][body_dict["piper_hand"]][1]
            quat_start  = quat_save #gymapi.Quat(*quat_start)
            pos_start   = current_pos
            pos_start_ctrl  = pos_controller
            quat_start_ctrl = quat_controller  
            # print(f"#----------------capture quat_start! {quat_start} quat_start_ctrl:{quat_start_ctrl} ----------------#")

        
        if quat_start is None:     # trigger never pressed yet
            prev_trigger_on = trigger_on
            continue
        
        if prev_trigger_on and not trigger_on:
            # props = gym.get_actor_rigid_body_states(envs[i], piper_handles[i], gymapi.STATE_POS)
            # quat_save = props['pose'][:][body_dict["piper_hand"]][1]
            quat_save = total_rotation # gymapi.Quat(*quat_save)
            pos_save  = total_pose
            # print(f"#----------------save quat_save! {quat_save} pos_save:{pos_save} ----------------#")

        if trigger_on:
            # calculate the position difference
            pos_ctrl_delta  = pos_controller - pos_start_ctrl 
            # print(f"pos_start = {pos_start}, pos_ctrl_delta= {pos_ctrl_delta}")
            total_pose      = pos_start + pos_ctrl_delta
            pose.p          = total_pose    #gymapi.Vec3(*total_pose)
            goal_pos        = torch.tensor([total_pose.x, total_pose.y, total_pose.z]).unsqueeze(0)
            # print(f"total_pose = {total_pose}")
            # calculate the rotation difference
            quat_ctrl_delta = quat_start_ctrl.inverse() * quat_controller
            
            quat_difference_1 = quat_start.inverse() * quat_ctrl_delta
            """if quat_save is None:
                quat_difference_2 = IDENTITY
            else:
                quat_difference_2 = quat_start.inverse() * quat_save """       
            
            total_rotation = (quat_ctrl_delta * quat_start) #* quat_difference_2
            pose.r      = total_rotation 
            # print(f"quat_ctrl_delta= {quat_ctrl_delta}, total_rotation = {total_rotation}, quat_start = {quat_start},quat_start_ctrl={quat_start_ctrl}")
            goal_rot    = torch.tensor([total_rotation.x, total_rotation.y ,total_rotation.z, total_rotation.w]).unsqueeze(0)
            # alpha = args.alpha
            # 
            # print(f"goal_rot = {goal_rot}")
            # print(f"quat_start = {quat_start}, quat_start_ctrl = {quat_start_ctrl}, quat_controller = {quat_controller} quat_ctrl_delta= {quat_ctrl_delta}, total_rotation  = {total_rotation}")
        else:
            # if quat_save is None:
            #     pose.p      = current_pos
            #     pose.r      = current_rot
            #     goal_pos    = torch.tensor([current_pos.x, current_pos.y, current_pos.z]).unsqueeze(0) 
            #     goal_rot    = torch.tensor([current_rot.x, current_rot.y ,current_rot.z, current_rot.w]).unsqueeze(0) 
                
            # else:
            # print(f"CURRENT_POS = {current_pos}")
            pose.p      = current_pos         # gymapi.Vec3(*total_pose)
            # print(f"pose.p = {pose.p}")
            pose.r      = current_rot  #total_rotation 
            goal_pos    = torch.tensor([pos_save.x, pos_save.y, pos_save.z]).unsqueeze(0) 
            # goal_pos    = torch.tensor([current_pos.x, current_pos.y, current_pos.z]).unsqueeze(0) 
            goal_rot    = torch.tensor([total_rotation.x, total_rotation.y, total_rotation.z, total_rotation.w]).unsqueeze(0) 
            
            # alpha = 1.0
                # print(f"total_rotation  = {total_rotation}")
                
        
            # quat_save = pose.r
     
 
        
        # pose.r = gymapi.Quat.from_euler_zyx(vr_goal_rot_copy[2], vr_goal_rot_copy[1], vr_goal_rot_copy[0])
        # pose.r = gymapi.Quat(0,0,0.707107, -0.707107)
        # pose.r = gymapi.Quat(0.0,0.0, -0.70710678, 0.70710678)
        # pose.r  = gymapi.Quat(*vr_goal_rot_copy)
        # print("pose: ", pose.p)
        # print("rot: ", pose.r)
        # yaw = 15 / 180 * math.pi # radians
        # yaw_q = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), yaw)
        
        # x = gymapi.Quat.from_euler_zyx(0, 0, 0)
        # y = gymapi.Quat.from_euler_zyx(0, -math.pi, 0)
        # print(f"y = {y}, type = {type(y)}")
        
        # pose.r = yaw_q * y
        # down = gymapi.Quat(-0.70710678, 0, 0, 0.70710678)
        
        # ------- first-time trigger: remember the start pose -------------

  
        # ------- convert the delta into Gym's coordinate frame ----------
        # delta_gym = (Q_VR2GYM * delta_vr) * Q_VR2GYM.inverse()

        
        # gym.set_attractor_target(envs[i], attractor_handles[i], pose)
        
        
        # goal_pos = torch.tensor([-vr_goal_pos_copy[0], vr_goal_pos_copy[2], vr_goal_pos_copy[1]]).unsqueeze(0)
        # goal_rot = torch.tensor([0,0,0,-1]).unsqueeze(0)
        
        writer.add_scalars("x",{"control": goal_pos[0, 0],
                                "predicted": target_action[0],
                                "actual": prev_pos.x}, t)
        
        writer.add_scalars("y",{"control": goal_pos[0,1],
                                "predicted": target_action[1],
                                "actual": prev_pos.y}, t)
        
        writer.add_scalars("z",{"control": goal_pos[0,2],
                                "predicted": target_action[2],
                                "actual": prev_pos.z}, t)
        
        """writer.add_scalars("postion",{"x_actual":       goal_pos[0, 0],
                                      "x_predicted":    target_action[0],
                                      "y_actual":       goal_pos[0, 1],
                                      "y_predicted":    target_action[1],
                                      "z_actual":       goal_pos[0, 2],
                                      "z_predicted":    target_action[2]
                                      }, t)"""
        
        
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
        # if thumbstick is not None:
        # print(f"alpha_pos = {alpha_pos}, alpha_rot = {alpha_rot}", end='\r', flush=True)
        print(SAVE + "\033[1A" + CLEAR + f"alpha = {alpha_pos:.3f}, beta = {alpha_rot:.3f}" + RESTORE , end="", flush=True)
        
       
        
        
        """if buttonB:
            
            if alpha >= args.alpha:
                alpha = 0
            else:
                alpha += 0.1
            alpha = round(alpha,1)
            print(f"alpha = {alpha}")"""
            
        goal_pos = goal_pos * (1 - alpha_pos) + target_action[:3] * alpha_pos 
        
        # goal_rot = goal_rot * (1 - alpha) + target_action[3:] * alpha 
        interpolated_quat = slerp(goal_rot.squeeze(0), target_action[3:], alpha_rot)
        goal_rot = interpolated_quat.unsqueeze(0) / torch.norm(interpolated_quat)
        # print(f"norm goal rot = {torch.normz(goal_rot)}")
        # print(f"goal_rot: {goal_rot}")
        pose_model = gymapi.Transform()
        pose_model.p = gymapi.Vec3(*target_action[:3])
        pose_model.r = gymapi.Quat(*target_action[3:])
        
        hand_pos_buffer     = [torch.cat((hand_pos_buffer[k], torch.cat([goal_pos, goal_rot], dim=1)[k].unsqueeze(0)),dim=0) for k in range(num_envs)]
        dof_pos_buffer      = [torch.cat((dof_pos_buffer[k],    dof_pos[k].squeeze(1).unsqueeze(0)),    dim=0) for k in range(num_envs)]
        
        pos_err = goal_pos - hand_pos
        orn_err = orientation_error(goal_rot, hand_rot)
        
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1).float()
        pos_action[:, :6] = dof_pos.squeeze(-1)[:, :6] + control_ik(dpose)
        
        # Deploy actions
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
        
        
        # pose.r = gymapi.Quat(*vr_goal_rot_copy)
        # pose.r = gymapi.Quat.from_euler_zyx(roll, pitch, pitch)
        """gymutil.draw_lines(axes_geom,   gym, viewer, envs[i], pose)
        gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)"""
        # piper_dof_states        = gym.get_actor_dof_states(envs[i], piper_handles[i], gymapi.STATE_POS)
        # gym_piper_states        =  piper_dof_states['pos']

        piper_velocity_target   = gym.get_actor_dof_velocity_targets(envs[i], piper_handles[i])
        control_gripper_velocity(piper_velocity_target)
        # print(f"forces = {dof_forces}")
        # control_gripper_force(dof_forces)
        
        
    prev_trigger_on = trigger_on 
    prev_pos = current_pos  
        
    prev_hand_pos = hand_pos
    
    is_inside_tray = (torch.abs(box_pos[:, 2] - tray_pos[:, 2])   < 1.5 * box_size).unsqueeze(1)

    is_in_tray = torch.cat([is_bounded_by_tray, is_inside_tray], dim=1).all(dim=1)
    is_in_tray = (is_in_tray & (torch.norm(box_vel[:, :3]) < 0.05))
    
    num_success = is_in_tray.sum().item()
    
        
    # env_success = is_in_tray.all(dim=1)
    
    """if buttonB:
        success_count_each_env[0] += 1 """
        
    if torch.any(is_in_tray): 
        success_idx = torch.where(is_in_tray)[0]
        # print("success_idx", success_idx)

        success_count_each_env[is_in_tray] += 1 
    
        to_be_removed = closest_box_idxs[success_idx] #remove the current closest box from successful environments
        
        
        # unfinished_box_idxs[is_in_tray].remove(to_be_removed)
        for env_i, box_idx in zip(success_idx.tolist(), to_be_removed.tolist()):
            # Remove the box index from the unfinished list for this environment
            if box_idx in unfinished_box_idxs[env_i]:
                unfinished_box_idxs[env_i].remove(box_idx)
                gym.set_rigid_body_color(envs[env_i], box_handles[env_i][box_idxs[env_i].index(box_idx)], 0, gymapi.MESH_VISUAL_AND_COLLISION, finished_color) #color
                # print(f"The box id {box_idx} in the environment {env_i} is put in the box")
                # print(f"There is/are {len(unfinished_box_idxs[env_i])} left in the environment {env_i}")
        
        find_new_idx[success_idx] = True #to reset the closest box idxs

    finished_envs = (success_count_each_env >= 1) 
    # finished_envs = (success_count_each_env >= num_box) 
    # print("finished_envs", finished_envs)

    restart_envs = finished_envs  | (time_envs >=  end_time)
    
    # update viewer
    """gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)"""
    
    if args.record:
        image_processing(writers)
    # reset the environment if it stops moving for some time.
    if buttonA: 
        restart_envs = torch.full_like(restart_envs, True)
        # print("RESET ENVIRONMENT")
        # print(f"goal_pos = {goal_pos}")
        # goal_pos = hand_pos.clone()
        # reset(envs, dof_states)
    
    if torch.any(restart_envs):
        total_ep += 1
        if torch.any(finished_envs) & args.record:
            for i, finished in enumerate(finished_envs): 
                # print(f"There is/are {len(unfinished_box_idxs[i])} left in the environment {i}")
                # print(f"hand pos shape : {hand_pos_buffer[i].shape} , {hand_pos_buffer[i].shape[0]}")
                # print(f"dof pos shape : {dof_pos_buffer[i].shape} , {dof_pos_buffer[i].shape[0]}")
                print(f"Saving successful Ep {Eps_success[i]} of environment {i}, success tracker = {success_count_each_env}")
                shutil.move(f"color_1_env_{i}.avi", f"{storage_dir}/env_{i}/colors/color_1_env_{i}_ep_{Eps_success[i]}.avi")
                shutil.move(f"color_2_env_{i}.avi", f"{storage_dir}/env_{i}/colors/color_2_env_{i}_ep_{Eps_success[i]}.avi")
                store_joints_states(i, Eps_success[i], dof_pos_buffer[i], hand_pos_buffer[i], True)
                store_depth_data(i, Eps_success[i], depth_stack_1[i], depth_stack_2[i])
                Eps_success[i] += 1

                dof_pos_buffer[i]       = torch.empty((0, 8))
                hand_pos_buffer[i]      = torch.empty((0, 7))
                depth_stack_1[i]        = []
                depth_stack_2[i]        = []
                
                writers[i] = create_video_writer(i, writers[i])
                
        elif torch.any(finished_envs):
            for i, finished in enumerate(finished_envs): 
                # print(f"There is/are {len(unfinished_box_idxs[i])} left in the environment {i}")
                # print(f"hand pos shape : {hand_pos_buffer[i].shape} , {hand_pos_buffer[i].shape[0]}")
                # print(f"dof pos shape : {dof_pos_buffer[i].shape} , {dof_pos_buffer[i].shape[0]}")
                success_tracker += 1
                print(f"SUCCESS!: {args.name}: Mode {args.mode} success tracker = {success_tracker}, time_envs={time_envs[i]} ts, total= {total_ep}")
                print(SAVE + "\033[7A" + CLEAR + f"SUCCESS!: {args.name}: Mode {args.mode}  success tracker = {success_tracker}, time_envs={time_envs[i]} ts, total= {total_ep}"+ RESTORE, end="", flush=True)
                
                Eps_success[i] += 1

                dof_pos_buffer[i]       = torch.empty((0, 8))
                hand_pos_buffer[i]      = torch.empty((0, 7))
                depth_stack_1[i]        = []
                depth_stack_2[i]        = []
                
        else:
            for i, restart in enumerate(restart_envs): 
                if restart:    
                    # print(SAVE + "\033[7A" + CLEAR + f"Discarding an episode, time_envs={time_envs[i]} ts, total ={total_ep}"+ RESTORE, end="", flush=True)
                    if args.record:
                        os.remove(f"color_1_env_{i}.avi")
                        os.remove(f"color_2_env_{i}.avi")
                    # os.remove(f"depth_1_env_{i}.avi")
                    # os.remove(f"depth_2_env_{i}.avi")
                    dof_pos_buffer[i]       = torch.empty((0, 8))
                    hand_pos_buffer[i]      = torch.empty((0, 7))
                    depth_stack_1[i]        = []
                    depth_stack_2[i]        = []
                    if args.record:
                        writers[i] = create_video_writer(i, writers[i]) 
        
        # success_count_each_env[restart_envs] = 0 # normal testing experiment
                
        reset(restart_envs, dof_states)
        pos_save = gymapi.Vec3(0.055,0.0,0.22)
        quat_save = gymapi.Quat(-0.478,-0.478,-0.521,-0.521)
        pose.p = pos_save
        pose.r = quat_save
        predicted_trajectories = np.array(predicted_trajectories)
        actual_trajectores     = np.array(actual_trajectores)
        # print(f"shape of predicted_trajectories: {len(predicted_trajectories)}x [{predicted_trajectories[0].shape}]")
        # print(f"shape of predicted_trajectories: {predicted_trajectories.shape}")
        # print(f"shape of actual_trajectories: {actual_trajectores.shape}")
        ax.clear()
        """for i in range(predicted_trajectories.shape[0]):
            ax.plot(predicted_trajectories[i, :,0], predicted_trajectories[i, :,1], predicted_trajectories[i, :,2], color='green')
            ax.scatter(predicted_trajectories[i,0,0], predicted_trajectories[i,0,1], predicted_trajectories[i,0,2], color='green') 
            ax.scatter(predicted_trajectories[i,-1,0], predicted_trajectories[i,-1,1], predicted_trajectories[i,-1,2], color='red')"""
        ax.plot(predicted_trajectories[:,0],predicted_trajectories[:,1],predicted_trajectories[:,2],  color='green')
        ax.plot(actual_trajectores[:,0],actual_trajectores[:,1],actual_trajectores[:,2],  label='Actual_path', color='grey')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_xlim([-0.1, 0.5])
        # ax.set_ylim([-0.4, 0.4])
        # ax.set_zlim([0,0.5 ])
        ax.legend()
        ax.view_init(elev = 30, azim= 0)
        # plt.show() #blocking plot
        
        # plt.draw()
        # plt.pause(0.1)
        
        # plt.clf() 
        predicted_trajectories  = []
        actual_trajectores      = []
        
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, action_dim]).cuda()
        t = 0
        # froze_count[restart_envs] = 0  
        # change_sign = torch.logical_not(frozen_envs)
        # frozen_envs = torch.full_like(frozen_envs, False).to(device)    
        success_count_each_env[restart_envs] = 0 

        for i, do_reset in enumerate(restart_envs):
            if do_reset:
                time_envs[i] = 0
                unfinished_box_idxs[i] = copy.deepcopy(box_idxs[i])
                # color
                
                for n in range(num_box):
                    unfinished_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                    gym.set_rigid_body_color(envs[i], box_handles[i][n], 0, gymapi.MESH_VISUAL_AND_COLLISION, unfinished_color)
   
    # gymutil.draw_lines(axes_geom,   gym, viewer, envs[0], pose)
    gymutil.draw_lines(sphere_geom, gym, viewer, envs[0], pose)
    # gymutil.draw_lines(model_axes_geom,   gym, viewer, envs[0], pose_model)
    # gymutil.draw_lines(model_sphere_geom, gym, viewer, envs[0], pose_model)
    shadow_pose = gymapi.Transform()
    shadow_pose.p = current_pos
    shadow_pose.p.z = 0.01
    shadow_pose.r = shadow_rot
    gymutil.draw_lines(shadow_geom, gym, viewer, envs[0], shadow_pose)
    
    # signal when the claws touch anything
    if (net_cf[claws_idxs[0][0]][2] > 0):
        alert_pos = gymapi.Transform()
        alert_pos.p = current_pos
        alert_pos.p.y -= 0.05
        alert_pos.p.z += 0.2
        gymutil.draw_lines(contact_alert_goem_1, gym, viewer, envs[0], alert_pos)
        # gym.set_rigid_body_color(envs[0], piper_handles[0], 6, gymapi.MESH_VISUAL_AND_COLLISION, unfinished_color)
    if (net_cf[claws_idxs[0][1]][2] > 0):
        alert_pos = gymapi.Transform()
        alert_pos.p = current_pos
        alert_pos.p.y += 0.05
        alert_pos.p.z += 0.2
        gymutil.draw_lines(contact_alert_goem_2, gym, viewer, envs[0], alert_pos)
    
    
    if not args.headless:
        gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    gym.end_access_image_tensors(sim)
    time_envs[:] += 1
    t+=1
    """if t % 500 == 0:
        print(f"Total frozen resets: {frozen_counter}")
        print(f"Total successes: {success_counter}")"""
    toc = time.time()
    fq = toc - tic
    acc_latency += latency
    if time_envs[0] % 20 == 0:
        avg_latency = acc_latency / 20
        acc_latency = 0
    print(SAVE + "\033[2A" + CLEAR + f"frequency: {1/fq:.1f} Hz, latency={avg_latency* 1000:.2f}"+ RESTORE, end="", flush=True)
    tic = toc

# print(f"Total frozen resets: {frozen_counter}")

# cleanup

if not args.headless:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
cv2.destroyAllWindows()


client.loop_stop()