import math
import pickle
import shutil
from policy import ACTPolicy
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

import matplotlib.pyplot as plt 
import numpy as np
import cv2
import gin

from scripts.utils import slerp, get_observations, get_image, create_video_writer

from pxr import UsdPhysics
"""import inspect
from omni.isaac.motion_generation.lula import LulaKinematicsSolver

print(inspect.signature(LulaKinematicsSolver))"""

import collections
import json
import os

from paho.mqtt import client as mqtt_client 
from threading import Lock

import numpy as np
import torch
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*has been deprecated.*",
)

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
            controller_obj  = [data_controller['_x'], data_controller['_y'], data_controller['_z']]
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



# ---------------------------
# |       Simulation        |
# ---------------------------
open_stage("/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description/piper_env_warehouse_3.usd")
sim = SimulationContext()
dt  = sim.get_physics_dt()
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
robot = SingleArticulation("/World/piper_description")
robot.initialize()

base                = SingleRigidPrim("/World/piper_description/link1")
piper_hand          = SingleRigidPrim("/World/piper_description/piper_hand")
linkEndEffector     = SingleXFormPrim("/World/piper_description/piper_hand/linkEndEffector")
cube                = SingleXFormPrim("/World/Xform")
bin                 = SingleXFormPrim("/World/Xform_bin")
teddy_bear          = SingleXFormPrim("/World/Xform_teddy_bear")
dex_cube            = SingleXFormPrim("/World/Xform_dex_cube")
rubik               = SingleXFormPrim("/World/Xform_rubik")
nvidia_cube         = SingleXFormPrim("/World/Xform_nvidia_cube")
mug                 = SingleXFormPrim("/World/Xform_mug")

body_rgb_cam        = Camera(
                    prim_path="/World/piper_description/piper_hand/Xform_camera/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
                    frequency=frequency,
                    resolution=(width, height),)
mid_rgb_cam        = Camera(
                    prim_path="/World/Realsense_mid/RSD455/Camera_OmniVision_OV9782_Color",
                    frequency=frequency,
                    resolution=(width, height),)
body_depth_cam      = Camera(
                    prim_path="/World/piper_description/piper_hand/Xform_camera/Realsense/RSD455/Camera_Pseudo_Depth",
                    frequency=frequency,
                    resolution=(width, height),)
mid_depth_cam        = Camera(
                    prim_path="/World/Realsense_mid/RSD455/Camera_Pseudo_Depth",
                    frequency=frequency,
                    resolution=(width, height),)

rgb_cams            = [body_rgb_cam, mid_rgb_cam]
depth_cams          = [body_depth_cam, mid_depth_cam]

base.initialize()
piper_hand.initialize()
linkEndEffector.initialize()
body_rgb_cam.initialize()
mid_rgb_cam.initialize()
body_depth_cam.initialize()
mid_depth_cam.initialize()
body_depth_cam.add_distance_to_image_plane_to_frame()
mid_depth_cam.add_distance_to_image_plane_to_frame()
# body_depth_cam.get_annotator("distance_to_image_plane")


# body_rgb_cam.add_motion_vectors_to_frame()
robot_prim = get_prim_at_path("/World/piper_description")
stage = get_prim_at_path("/World/piper_description").GetStage()

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
    robot_description_path="/home/ucluser/isaacgym/assets/urdf/piper_description/config/piper_robot.yaml",
    urdf_path="/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description.urdf"
)
ik_solver.set_default_position_tolerance(0.09)
ik_solver.set_default_orientation_tolerance(0.09)

print(f"#-------------- get_default_position_tolerance() = {ik_solver.get_default_position_tolerance()}")
print(f"#-------------- get_default_orientation_tolerance() = {ik_solver.get_default_orientation_tolerance()}")

#-------------- get_default_position_tolerance() = 0.001
#-------------- get_default_orientation_tolerance() = 0.010000041667134873

kin_solver = ArticulationKinematicsSolver(
    robot_articulation=robot,
    kinematics_solver=ik_solver,
    end_effector_frame_name="linkEndEffector" #piper_hand 
)
piper_hand_pos_world, _ = piper_hand.get_world_pose()
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

def image_capture(writers: list | None, depth_stacks: list, rgb_cams: list, depth_cams: list, width: int, heigth: int, record: bool) -> list | None:
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
        heigth (int): Height of the camera images (note: typo in parameter name).
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
    assert len(rgb_cams) == len(depth_cams)
    for i in range(len(rgb_cams)):
        
        img     = rgb_cams[i].get_rgba()
        depth   = depth_cams[i].get_depth()
        
        if len(img) == 0 or depth is None:
            return None
        color_image = img.copy()
        # Reshape to (height, width, 4) - note: height comes first!
        color_image = color_image.reshape((heigth, width, 4))
        rgb_image   = color_image[:, :, :3]
        bgr_image   = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        depth_image = depth.copy()
        depth_image = np.clip(depth_image, 0.0, 5.0)
        depth_image = depth_image[:, :, np.newaxis] 
        depth_image = depth_image.reshape((heigth, width, 1))
        
        
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

    # cv2.imwrite(os.path.join(input_save_dir, f"pred_cam1_{t:06d}.png"),
    #             depth_colormap_1,
    #             )
    # cv2.imwrite(os.path.join(input_save_dir, f"pred_cam2_{t:06d}.png"),
    #             depth_colormap_2,
    #             )
    # cv2.imwrite(
    #                 os.path.join(input_save_dir, f"pred_cam1_{t:06d}.png"),
    #                 cv2.cvtColor(color_image_1, cv2.COLOR_RGB2BGR),
    #             )
    # cv2.imwrite(
    #                 os.path.join(input_save_dir, f"pred_cam2_{t:06d}.png"),
    #                 cv2.cvtColor(color_image_2, cv2.COLOR_RGB2BGR),
    #             )
    
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
    pos_1, q_1 = random_pose(is_tray=True)
    pos_2, q_2 = random_pose(is_tray=False)
    pos_3, q_3 = random_pose(is_tray=False)
    pos_4, q_4 = random_pose(is_tray=False)
    pos_5, q_5 = random_pose(is_tray=False)
    pos_6, q_6 = random_pose(is_tray=False)
    pos_7, q_7 = random_pose(is_tray=False)
    
    bin.set_world_pose(
        position=pos_1,
        orientation=q_1,
    )
    teddy_bear.set_world_pose(
        position=pos_2,
        orientation=q_2,
    )
    dex_cube.set_world_pose(
        position=pos_3,
        orientation=q_3,
    )
    rubik.set_world_pose(
        position=pos_4,
        orientation=q_4,
    )
    nvidia_cube.set_world_pose(
        position=pos_5,
        orientation=q_5,
    )
    mug.set_world_pose(
        position=pos_6,
        orientation=q_6,
    )
    cube.set_world_pose(
        position=pos_7,
        orientation=q_7,
    )

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
ee_pos_world, current_rotation = linkEndEffector.get_world_pose()
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
# for cam in range(len(rgb_cams)):
#     writers.append(create_video_writer(f"rgb_{cam}", frequency, width, height))
#     depth_stacks.append([])
print(f" number of writers : {len(writers)}")
subscribe(client)
client.loop_start()
while simulation_app.is_running():
    sim.step(render=True)
    t0 = time.perf_counter()
    # --------- VR Controller Part ---------------------------------------------------------#
    pos_controller = np.array(vr_goal_pos)
    # target_rot = euler_angles_to_quat(np.array([0.5 * math.pi, 0, 0])) # Always pointing down
    # target_rot = euler_angles_to_quat(np.array([0.5 * math.pi, 0, controller_obj[2]])) # Always pointing down
    quat_controller = euler_angles_to_quat(np.array([controller_obj[2], controller_obj[0],controller_obj[1]])) #
    # target_rot = np.array(down_vector)
    # target_rot = np.array([-0.70710678, 0, 0, 0.70710678])
    
    ee_pos_world, current_rotation = linkEndEffector.get_world_pose()
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
        
    target_rot = total_rotation.as_quat()  # use as IK target
    
    # -----  Gripper --------------------- #
    joint_efforts = robot.get_measured_joint_efforts()
    if grip_flag:
        joint_efforts[6] = - 40.0
        joint_efforts[7] = 40.0
        
    else:
        joint_efforts[6] = 40.0
        joint_efforts[7] = -40.0
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
    captured_img = image_capture(None, depth_stacks, rgb_cams, depth_cams, width, height, record=False)
    
    #-------- Prediction model (start) -------------------------------------------------------# 
    '''current_q = robot.get_joint_positions()  
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
            with torch.no_grad():
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
    #-------- Prediction model (end) -------------------------------------------------------# 
    # t5 = time.perf_counter()
    t3 = time.perf_counter()
        
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
    # print(SAVE + "\033[1A" + CLEAR + f"alpha = {alpha_pos:.3f}, beta = {alpha_rot:.3f}, save the video : {record}" + RESTORE , end="", flush=True)
    
    # target_pos = np.array([0.2,0.2,0.2])
    VR_target_marker.set_world_pose(position    = target_pos + base_pos_world
                                ,orientation    = target_rot)
    
    if target_action is not None:
        target_pos = target_pos * (1 - alpha_pos) + target_action[:3] * alpha_pos * 2
        interpolated_quat = slerp(target_rot, target_action[3:], alpha_rot)
        target_rot = interpolated_quat / torch.norm(interpolated_quat)

    # print("total_rotation:", target_rot)
    # target_rot = torch.from_numpy(target_rot).float()
    
    
    # print("shape of goal_rot:", goal_rot.shape)
    # print("type  of goal_rot:", type(goal_rot))
    
    prev_trigger_on = trigger_on 
    
    
    # print("target_pos", target_pos)
    
    # R_target_world = R.from_quat(target_rot)
    # R_target_base = R_base.inv() * R_target_world
    # target_rot_base = R_target_base.as_quat()
        
    action_target, success = kin_solver.compute_inverse_kinematics(
            target_position     = target_pos * 0.5,
            target_orientation  = target_rot * 0.5
        )
    t4 = time.perf_counter()

    if success:
        print(f"target_pose = {target_pos + base_pos_world}")
        print(action_target)
        # print(f"base_pos_world = {base_pos_world}")
        # print(f"piper_hand_pos = {ee_pos_world}")
        # print(f"position difference = {target_pos * 0.5 + base_pos_world - ee_pos_world }")
        #alpha = 0.05  # smoothing
        #q = (1 - alpha) * current_q + alpha * action_target
        robot.apply_action(action_target)
        
    t+=1
    t5 = time.perf_counter()
    if t1 and t2 and t3 and t4 and t5:
        print(f"t1-t0={(t1-t0)*1000:.2f}, t3-t1={(t2-t1)*1000:.2f}, t4-t3={(t4-t3)*1000:.2f}, t5-t4={(t5-t4)*1000:.2f}")
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
        piper_hand.initialize()
        linkEndEffector.initialize()
        target_pos = np.array([0.14,-0.03,2.5]) - base_pos_world
        
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