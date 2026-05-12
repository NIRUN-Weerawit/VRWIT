"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

import os
import shutil
import cv2
import csv
import h5py
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time
import copy

from torch.utils.tensorboard import SummaryWriter



class Gym_env():
    def __init__(self):
        
        self.num_envs           = 1

        self.seed               = 22
        self.sim_device         = "cpu"
        self.env                = None
        self.gym                = None     #self.gym = gymapi.acquire_gym()
        self.sim                = None     #self.sim = self.gym.create_sim(args['compute_device_id'], args['graphics_device_id'], args['physics_engine'], sim_params)
        
        self.viewer             = None
        self.envs               = []
        
        self.piper_dof_states   = []
        self.piper_body_states  = []
        self.piper_handles      = []
        self.piper_velocity_target = []
        self.piper_hand         = "link6"
        self.saved_dof_states   = None
        
        self.cube_states        = []
        self.cube_handles       = []
        
        self.states_bf          = []
        
        self.cube_pose          = [0.3, 0.2, 0.0]        # Target position
        self.goal_rot           = [0.0, 0.0, 0.0, -1.0]   # Target orientation (quaternion)

        # self.sphere_geom        = None

        self.asset_root  =  "/home/ucluser/isaacgym/assets"
        # self.asset_root  =  "/home/wee_ucl/workspace/Piper_RL/assets/"
        self.success_dataset_dir= None
        self.failure_dataset_dir= None
        self.piper_lower_limits = []
        self.piper_upper_limits = []
        self.piper_mids         = []
        self.piper_num_dofs     = None
        
        self.envs               = []
        self.tray_handles       = []
        self.piper_handles      = []
        self.box_handles        = []
        self.camera_handles     = []
        self.attractor_handles  = []
        self.tray_idxs          = []
        self.box_idxs           = []
        self.unfinished_box_idxs= []
        self.hand_idxs          = []
        self.init_pos_list      = []
        self.init_rot_list      = []
        self.writers            = []
        self.dof_states         = None
        self.init_dof_states    = None
        self.rb_states          = None
        self.camera_props       = gymapi.CameraProperties()
        
        self.writer             = SummaryWriter("logs_replay")
        
        
        self.time_counter       = 0
        self.time_ep            = 0
        self.goal_dist_initial  = 0
        
        self.table_dims         = gymapi.Vec3(0.6, 1.0, 0.01)
        self.table_pose         = gymapi.Transform()
        self.cube_size          = 0.04
        self.tray_pose          = gymapi.Transform()
        self.tray_color         = None
        self.num_box            = 2
        self.t                  = 0
        
        torch.set_printoptions(precision=4, sci_mode=False)

        # GPU configuration
        if self.sim_device == 'cuda':
            print("CUDA IS AVAILABLE")
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"
        
        for j in range(self.num_envs):
            color_writer_1, color_writer_2 = self.create_video_writer(j)
            self.writers.append([color_writer_1, color_writer_2])
    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    
    def parse_device_str(self, device_str):
        # defaults
        device = 'cpu'
        device_id = 0

        if device_str == 'cpu' or device_str == 'cuda':
            device = device_str
            device_id = 0
        else:
            device_args = device_str.split(':')
            assert len(device_args) == 2 and device_args[0] == 'cuda', f'Invalid device string "{device_str}"'
            device, device_id_s = device_args
            try:
                device_id = int(device_id_s)
            except ValueError:
                raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
        return device, device_id
        
    def init_gym(self):
        """
            Create a parametrized empty gym environment 
        """
            
        # Initialize gym
        self.gym = gymapi.acquire_gym()

        args = dict(
            compute_device_id=0, 
            flex=False, 
            graphics_device_id=0, 
            num_envs=1, 
            num_threads=0, 
            physics_engine= gymapi.SIM_PHYSX, 
            physx=False, 
            pipeline='gpu', 
            sim_device='cpu', 
            sim_device_type='cuda', 
            slices=0, 
            subscenes=0, 
            use_gpu=False, 
            use_gpu_pipeline=False
        )
        
        # configure sim
        sim_params                  = gymapi.SimParams()
        sim_params.up_axis          = gymapi.UP_AXIS_Z
        sim_params.gravity          = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt               = 1.0 / 60.0
        sim_params.substeps         = 2
        sim_params.use_gpu_pipeline = args['use_gpu_pipeline']
        if args['physics_engine'] == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type                    = 1
            sim_params.physx.num_position_iterations        = 8
            sim_params.physx.num_velocity_iterations        = 1
            sim_params.physx.rest_offset                    = 0.0
            sim_params.physx.contact_offset                 = 0.001
            sim_params.physx.friction_offset_threshold      = 0.001
            sim_params.physx.friction_correlation_distance  = 0.0005
            sim_params.physx.num_threads                    = args['num_threads']
            sim_params.physx.use_gpu                        = args['use_gpu']
        else:
            raise Exception("This example can only be used with PhysX")


        #TODO: Consider where sim should be initialized
        self.sim = self.gym.create_sim(args['compute_device_id'], args['graphics_device_id'], args['physics_engine'], sim_params)

        if self.sim is None:
            raise Exception("Failed to create sim")


        # Create viewer

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        # Add ground plane
        plane_params        = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        
        # return gym, sim, viewer

    def load_piper(self):
        """
            Create a piper asset
        """
        piper_asset_file = "urdf/piper_description/urdf/piper_description.urdf"
        # piper_description_chain = Chain.from_urdf_file(self.asset_root + "/" + piper_asset_file)
        asset_options                           = gymapi.AssetOptions()
        asset_options.fix_base_link             = True
        asset_options.flip_visual_attachments   = True
        asset_options.armature                  = 0.01
        piper_asset                             = self.gym.load_asset(self.sim, self.asset_root, piper_asset_file, asset_options)
        print("Loading asset '%s' from '%s'" % (piper_asset_file, self.asset_root))
        
        return piper_asset  #, piper_description_chain

    def load_table(self):
        """
            Create a table asset
        """
        table_asset_options                 = gymapi.AssetOptions()
        table_asset_options.fix_base_link   = True
        table_asset_options.armature        = 0.01
        table_asset                         = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, table_asset_options)
        self.table_pose.p                   = gymapi.Vec3(0.5, 0.0, 0.5 * self.table_dims.z)
        print("Creating asset '%s' " % ("table"))
        
        return table_asset
    
    def load_cube(self):
        """
            Create a cube asset 
        """
        cube_size                    = self.cube_size
        cube_dim                     = [cube_size] * 3
        cube_asset_options           = gymapi.AssetOptions()
        cube_asset_options.density   = 1000
        cube_asset_options.armature  = 0.01
        cube_asset                   = self.gym.create_box(self.sim, cube_size, cube_size, cube_size, cube_asset_options)
        cube_pose                    = gymapi.Transform()
        cube_pose_np                 = np.array((0,0,0))
        print("Creating asset '%s' " % ("cube1"))
        
        return cube_asset

    def load_tray(self):
        """
            Create a tray asset
        """
        tray_dim                            = [0.15, 0.15, 0.005] #small
        # tray_dim = [0.2, 0.2, 0.01] #original
        self.tray_color                     = gymapi.Vec3(0.24, 0.35, 0.8)
        tray_asset_file                     = "urdf/tray/traybox_smaller.urdf"
        tray_asset_options                  = gymapi.AssetOptions()
        tray_asset_options.armature         = 0.01
        tray_asset_options.density          = 8000
        tray_asset_options.override_inertia = True
        tray_asset                          = self.gym.load_asset(self.sim, self.asset_root, tray_asset_file, tray_asset_options)
        x               = 0.3 #corner.x  # + table_dims.x * 0.2
        y               = 0.1 #corner.y + table_dims.y * 0.8
        z               = tray_dim[2]
        
        self.tray_pose.p     = gymapi.Vec3(x, y, z)
        print("Loading asset '%s' from '%s'" % (tray_asset_file, self.asset_root))
        
        return tray_asset
    
    def create_piper_env(self):
        """Create a simulation environment for PiPER 

        Parameters
        ----------
        args : dict
            simulation-related parameters, consisting of the following keys:
            
            -- sim_device            Physics Device in PyTorch-like syntax
            -- pipeline              Tensor API pipeline (cpu/gpu)
            - graphics_device_id    Graphics Device ID
            - physics_engine        Use FleX or PhysX for physics
            - num_threads           Number of cores used by PhysX
            --subscenes             Number of PhysX subscenes to simulate in parallel
            --slices                Number of client threads that process env slices
            - num_envs              Number of environments to create
            - total_time            Time to reach the target pose
            - use_gpu
            - use_gpu_pipeline
            - compute_device_id
            - 

        Returns
        -------
        envs
            a list of all simulated environments
        sim
            a simulation instance with Piper robots with objects
        viewer
            viewer of the gym environment
        """
        
        # Time to wait in seconds before moving robot
        next_piper_update_time = 1.0
        # Set up the env grid
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        piper_pose = gymapi.Transform()
        piper_pose.p = gymapi.Vec3(0, 0.0, 0.0)
        piper_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        cube_pose = gymapi.Transform()
        cube_pose.p = gymapi.Vec3(self.cube_pose[0], self.cube_pose[1], self.cube_pose[2])
        cube_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        
        goal_pose = gymapi.Transform()
        goal_pose.p = gymapi.Vec3(self.cube_pose[0], self.cube_pose[1], self.cube_pose[2])
        goal_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        
        # Create an wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))
        axes_geom = gymutil.AxesGeometry(0.1)
        
        attractor_properties = gymapi.AttractorProperties()
        attractor_properties.stiffness = 5e5
        attractor_properties.damping = 5e3
        # Make attractor in all axes
        attractor_properties.axes = gymapi.AXIS_ALL
        

        #create empty gym environment  
        self.set_seed(self.seed)
        self.init_gym()
        piper_asset = self.load_piper()
        cube_asset  = self.load_cube()
        tray_asset  = self.load_tray()
        table_asset = self.load_table()

        piper_pose     = gymapi.Transform()
        piper_pose.p   = gymapi.Vec3(0, 0, 0)
        
        # get joint limits and ranges for piper
        piper_dof_props             = self.gym.get_asset_dof_properties(piper_asset)
        piper_lower_limits          = piper_dof_props['lower']
        piper_upper_limits          = piper_dof_props['upper']
        piper_ranges                = piper_upper_limits - piper_lower_limits
        piper_mids                  = 0.5 * (piper_upper_limits + piper_lower_limits)
        
        # default dof states and position targets
        self.piper_num_dofs         = self.gym.get_asset_dof_count(piper_asset)
        default_dof_pos             = np.zeros(self.piper_num_dofs, dtype=np.float32)
        default_dof_pos[:6]         = piper_mids[:6]
        
        # grippers open
        default_dof_pos[6:]         = piper_upper_limits[6:]
        default_dof_state           = np.zeros(self.piper_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"]    = default_dof_pos
        
        

        piper_dof_props["driveMode"][:6].fill(gymapi.DOF_MODE_POS)
        piper_dof_props["stiffness"][:6].fill(400.0)
        piper_dof_props["damping"][:6].fill(40.0)
        
        piper_dof_props["driveMode"][6:].fill(gymapi.DOF_MODE_POS)
        piper_dof_props["stiffness"][6:].fill(800.0)
        piper_dof_props["damping"][6:].fill(40.0)
        
        table_pose      = gymapi.Transform()
        table_pose.p    = gymapi.Vec3(0.35, 0.0, 0.5 * self.table_dims.z )

        box_pose        = gymapi.Transform()

        
        
        self.camera_props.width      = 640
        self.camera_props.height     = 480
        camera_1_position       = gymapi.Vec3(0.4, 0.5, 0.5)
        camera_1_target         = gymapi.Vec3(0, 0, 0)
        camera_2_position       = gymapi.Vec3(0.4, - 0.5, 0.6)
        camera_2_target         = gymapi.Vec3(0, 0, 0)
        
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


        # unfinished_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        # finished_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        
        print("Creating %d environments" % self.num_envs)
        num_per_row = int(math.sqrt(self.num_envs))

        def random_box_pose():
            box_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.1)
            box_pose.p.y = table_pose.p.y + np.random.uniform(-0.2, 0.2)
            box_pose.p.z = self.table_dims.z + 0.5 * self.cube_size
            # box_pose_np     = np.array([box_pose.p.x,box_pose.p.y,box_pose.p.z])
            # init_box_pose = np.zeros(1, dtype=gymapi.RigidBodyState.dtype)
            # init_box_pose['pose']['p'][0] =(box_pose.p.x, box_pose.p.y, box_pose.p.z)
            box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            return box_pose

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # add table
            # table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

            # add tray
            tray_handle     = self.gym.create_actor(env, tray_asset, self.tray_pose, "tray", i, 0)
            self.tray_handles.append(tray_handle)
            self.gym.set_rigid_body_color(env, self.tray_handles[i], 0, gymapi.MESH_VISUAL_AND_COLLISION, self.tray_color)
            # get global index of tray in rigid body state tensor
            tray_idx        = self.gym.get_actor_rigid_body_index(env, tray_handle, 0, gymapi.DOMAIN_SIM)
            self.tray_idxs.append(tray_idx)
            
            # add box
            self.box_handles.append([])
            self.box_idxs.append([])
            
            for n in range(self.num_box):
                box_handle          = self.gym.create_actor(env, cube_asset, random_box_pose(), "box_" + str(n), i, 0)
                unfinished_color    = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, unfinished_color) #color
                self.box_handles[i].append(box_handle)
                
                # get global index of box in rigid body state tensor
                box_idx = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
                self.box_idxs[i].append(box_idx)
            
            

            # add piper
            piper_handle = self.gym.create_actor(env, piper_asset, piper_pose, "piper", i, 2)
            self.piper_handles.append(piper_handle)
            
            # set dof properties
            self.gym.set_actor_dof_properties(env, piper_handle, piper_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, piper_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, piper_handle, default_dof_pos)

            # get inital hand pose
            hand_handle     = self.gym.find_actor_rigid_body_handle(env, piper_handle, "piper_hand")
            hand_pose       = self.gym.get_rigid_transform(env, hand_handle)
            self.init_pos_list.append([0.1, 0.1, 0.3])
            # init_pos_list.append([tray_pose.p.x, tray_pose.p.y, tray_pose.p.z])
            self.init_rot_list.append([-0.95, -0.25, 0.0, 0.0])
            # init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            # init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
            
            # get global index of hand in rigid body state tensor
            hand_idx        = self.gym.find_actor_rigid_body_index(env, piper_handle, "piper_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)
            
            # add camera
            cam_1 = self.gym.create_camera_sensor(env, self.camera_props)
            cam_2 = self.gym.create_camera_sensor(env, self.camera_props)
            #set the location of camera sensor
            self.gym.set_camera_location(cam_1, env, camera_1_position, camera_1_target)
            self.gym.attach_camera_to_body(cam_2, env, hand_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles.append([])
            self.camera_handles[i].append(cam_1)
            self.camera_handles[i].append(cam_2)
            
            
            
            # gym.attach_camera_to_body(camera_handle, env, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            
            # Initialize the attractor
            props       = self.gym.get_actor_rigid_body_states(env, piper_handle, gymapi.STATE_POS)
            body_dict   = self.gym.get_actor_rigid_body_dict(env, piper_handle)
            
            attractor_properties.target = props['pose'][:][body_dict["piper_hand"]]
            attractor_properties.target.p.y = 0.0 #-= 0.05
            attractor_properties.target.p.z = 0.0 #0.03
            attractor_properties.target.p.x = 0.0 #0.1 
            attractor_properties.rigid_handle = hand_handle

            # Draw axes and sphere at attractor location
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, env, attractor_properties.target)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, env, attractor_properties.target)

            attractor_handle = self.gym.create_rigid_body_attractor(env, attractor_properties)
            self.attractor_handles.append(attractor_handle)

        self.unfinished_box_idxs = copy.deepcopy(self.box_idxs)
        
        # Point camera at environments
        cam_pos = gymapi.Vec3(4, 3, 3)
        cam_target = gymapi.Vec3(-4, -3, 0)

        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)
        
        # initial hand position and orientation tensors
        init_pos = torch.Tensor(self.init_pos_list).view(self.num_envs, 3).to(self.device)
        init_rot = torch.Tensor(self.init_rot_list).view(self.num_envs, 4).to(self.device)
        
        # get rigid body state tensor
        _rb_states      = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states       = gymtorch.wrap_tensor(_rb_states)
        # print("rb_state shape", rb_states.shape)
        # print("rb_state shape", rb_states.shape[0])
        # print("rb_state shape", rb_states.shape[1])

        rb_states_clone = self.rb_states.clone()
        # print("device: ", rb_states_clone.device)
        # print("rb_states_clone[tray_idxs, :] =", rb_states_clone[tray_idxs])
        init_tray_rot   = rb_states_clone[self.tray_idxs, 3:7]
        init_cube_rot   = rb_states_clone[self.box_idxs, 3:7]
        print("box_ids:", self.box_idxs)

        # get dof state tensor
        _dof_states             = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states         = gymtorch.wrap_tensor(_dof_states)  #torch.Size([24, 2])
        self.init_dof_states    = self.dof_states.clone().view(self.num_envs, 8, 2)
                
        print("Issacgym piper simulation sytem is completed")

    
    def step(self, predicted_actions):
        # self.time_envs[:] += 1
        self.t      +=1
        # self.step_physics()
        # print("step physics")
        self.refresh_tensors()
        # print("refresh_tensors")
        # self.apply_rl_actions(predicted_actions)
        self.apply_attractor_target(predicted_actions)
        # print(predicted_actions)
        self.wait()
        self.wait()
        self.step_physics()
        props       = self.gym.get_actor_rigid_body_states(self.envs[0], self.piper_handles[0], gymapi.STATE_POS)
        body_dict   = self.gym.get_actor_rigid_body_dict(self.envs[0], self.piper_handles[0])
        hand_pose   = props['pose'][:][body_dict["piper_hand"]]
        self.writer.add_scalars("x", {'predicted': predicted_actions[0],
                                      'actual': hand_pose[0][0]}, self.t)
        self.writer.add_scalars("y", {'predicted': predicted_actions[1],
                                      'actual': hand_pose[0][1]}, self.t)
        self.writer.add_scalars("z", {'predicted': predicted_actions[2],
                                      'actual': hand_pose[0][2]}, self.t)
        
        # self.evaluate(predicted_actions, real_actions)
        self.render() 
        #get the images
        self.finish_ep()
        
        # return 
    
    def wait(self):
        self.step_physics()
        self.render()
    
    def get_observations(self):
        
        joint_positions     = self.dof_states[:, 0].view(self.num_envs, 8)
        # print(f"joint_positions = {joint_positions}")
        return joint_positions.tolist()
    
    def step_physics(self):
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
    def refresh_tensors(self):
        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

    def render(self):
        # Step rendering
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        
        # self.image_processing(writers)
        
    def finish_ep(self):
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)
        self.gym.end_access_image_tensors(self.sim)

    def reset(self):
        print('#-----------------RESETTING ENV-----------------#')
        
        # print("restart_indices", restart_indices)
        # print("restart_indices", restart_indices.shape)
        # states_buffer                   = self.dof_states.clone().view(self.num_envs, 8, 2)
        # states_buffer                   = self.init_dof_states
        # dof_state                       = states_buffer.view(self.num_envs * 8,2)
        random_pose                     = self.random_pos_tensor()
        
        # tray_reset_idxs                 = torch.tensor(self.tray_idxs, device=self.device)
        # box_reset_idxs                  = torch.tensor(self.box_idxs, device=self.device)
        
        # print("rb_states[box_reset_idxs, :3] ", rb_states[box_reset_idxs, :3])
        # print("rb_states[box_reset_idxs, :3] shape ", rb_states[box_reset_idxs, :3].shape)
        # print("random_pose[restart_indices, 1:num_box + 1, :3]", random_pose[restart_indices, 1:num_box + 1, :3])
        # print("random_pose[restart_indices, 1:num_box + 1, :3] shape", random_pose[restart_indices, 1:num_box + 1, :3].reshape(-1, 3).shape)
        self.rb_states[self.tray_idxs, :3]  = random_pose[:, 0, :3]
        self.rb_states[self.tray_idxs, 3:7] = torch.tensor([0., 0., 0., 1.0], device=self.device)
        # for n in range(num_box):
        # rb_states[box_reset_idxs, :3]   = random_pose[restart_indices, 1:num_box + 1, :3]
        self.rb_states[self.box_idxs, :3]   = random_pose[:, 1:self.num_box + 1, :3] #.reshape(-1, 3)
        
        self.gym.set_rigid_body_state_tensor(self.sim, gymtorch.unwrap_tensor(self.rb_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.init_dof_states))

    def random_pos_tensor(self):
        """
            Return: torch.cat((tray_pose, boxes_pose), dim=1)
        """
        r_min = 0.2
        r_max = 0.4
        theta = 70
        radius_tensor   = torch.FloatTensor(self.num_envs, self.num_box + 1).uniform_(r_min, r_max).to(self.device)       # self.num_envs x num_objs (3)
        theta_tensor    = torch.FloatTensor(self.num_envs, self.num_box + 1).uniform_(- theta/180 * np.pi, theta/180 * np.pi).to(self.device)   
        x_tensor   = radius_tensor * torch.cos(theta_tensor)           # self.num_envs x 1
        y_tensor   = radius_tensor * torch.sin(theta_tensor)           # self.num_envs x 1
        z_tensor_tray   = torch.full([self.num_envs], self.table_dims.z).to(self.device)                   # self.num_envs x 1
        z_tensor_cube   = torch.full([self.num_envs], self.table_dims.z + 0.5 * self.cube_size).to(self.device)   # self.num_envs x 1
        
        # random_pose     = torch.stack((torch.stack([x_tensor[:, 0], y_tensor[:, 0], z_tensor_tray], dim=1), 
        #                                torch.stack([x_tensor[:, 1], y_tensor[:, 1], z_tensor_cube], dim=1)), dim=1).to(self.device)  
        tray_pose    = torch.stack([x_tensor[:, 0], y_tensor[:, 0], z_tensor_tray], dim=1).unsqueeze(1)
        boxes_pose  = torch.stack([torch.stack([x_tensor[:, j+1], y_tensor[:, j+1], z_tensor_cube], dim=1) for j in range(self.num_box)], dim=1)
        
        # print("tray_pose dim", tray_pose.shape)
        # print("boxes_pose dim", boxes_pose.shape)
        # print("tray_pose dim", tray_pose)
        # print("boxes_pose dim", boxes_pose)
        random_pose     = torch.cat((tray_pose, boxes_pose), dim=1)
        
        # print("random pose: ", random_pose)
        # print("random pose shape: ", random_pose.shape)
        
        return random_pose
    
    def apply_attractor_target(self, predicted_actions=None):
        """Specify the actions to be performed by the rl agent(s).

        If no actions are provided at any given step, the rl agents default to
        performing actions specified by SUMO.

        Parameters
        ----------
        predicted_actions : array_like
            list of actions provided by the RL algorithm
        """
        if predicted_actions is None:
            return
        
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*predicted_actions[:3])
        
        # print(vr_goal_pos_copy)
        # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(*predicted_actions[3:])
        # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        # print("pose: ", pose.p)
        # props       = self.gym.get_actor_rigid_body_states(self.envs[0], self.piper_handles[0], gymapi.STATE_POS)
        # body_dict   = self.gym.get_actor_rigid_body_dict(self.envs[0], self.piper_handles[0])
        # hand_pose   = props['pose'][:][body_dict["piper_hand"]]
        # print(f"hand pos {hand_pose[0]}")
        # diff        = [(hand_pose[0][0] - pose.p.x)/ pose.p.x, (hand_pose[0][1] - pose.p.y) /pose.p.y, (hand_pose[0][1] - pose.p.z) / pose.p.z]
        
        # print(f"piper hand pos diff  {diff}")
        self.gym.set_attractor_target(self.envs[0], self.attractor_handles[0], pose)
            
    def evaluate(self, predicted_actions, real_actions):
        """
            actions: np.array(7) or list : 3-position, 4-orientation
            states:  3-position, 4-orientation
        """
        actions = np.array(predicted_actions)
        # hand_pos = self.rb_states[self.hand_idxs, :]
        # hand_position = self.rb_states[self.hand_idxs, :3]
        # hand_rot = self.rb_states[self.hand_idxs, 3:7]
        squared_error = (hand_pos - actions) ** 2
        mse_pos = np.sum(squared_error[:3]) / len(actions)
        mse_rot = np.sum(squared_error[3:7]) / len(actions)
        
        return mse_pos, mse_rot
            
    def create_dataset_dirs(self):
        try:
            os.makedirs("dataset/success_seed_" + str(self.seed), exist_ok=True)
            os.makedirs("dataset/failure_seed_" + str(self.seed), exist_ok=True)
            os.makedirs("videos/success_seed_" + str(self.seed), exist_ok=True)
            os.makedirs("videos/failure_seed_" + str(self.seed), exist_ok=True)
        except FileExistsError:
            pass
                
    def create_video_writer(self, env): 
        """
            create a new pair of video writers in an envronment
            Return: color_writer_1, color_writer_2 
        """
        color_writer_1  = cv2.VideoWriter(f'color_1_env_{env}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
        color_writer_2  = cv2.VideoWriter(f'color_2_env_{env}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
        return color_writer_1, color_writer_2 
    # depth_writer_1    = cv2.VideoWriter('depth_1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    # depth_writer_2    = cv2.VideoWriter('depth_2.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    
    def image_capture(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        # self.render()
        color_image_1   = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][0], gymapi.IMAGE_COLOR)
        color_image_2   = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][1], gymapi.IMAGE_COLOR)
        depth_image_1   = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][0], gymapi.IMAGE_DEPTH)
        depth_image_2   = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][1], gymapi.IMAGE_DEPTH)
        # print("depth_image_type",type(depth_image_1))
        
        
        
        img_np_1 = color_image_1.reshape((self.camera_props.height, self.camera_props.width, 4))
        img_np_2 = color_image_2.reshape((self.camera_props.height, self.camera_props.width, 4))
        depth_image_1 = np.clip(depth_image_1, -3.0, -0.0)
        depth_image_2 = np.clip(depth_image_2, -3.0, -0.0)
        depth_image_1 = depth_image_1[:, :, np.newaxis]  # shape: [H, W, 1]
        depth_image_2 = depth_image_2[:, :, np.newaxis]
        
        
        # print("distance",depth_image_1[100][100])
        # depth_colormap_1 = cv2.convertScaleAbs(depth_image_1, alpha=1)
        # Normalize to [0, 1] or [-1, 1] if needed
        
        # depth_norm_1 = (depth_image_1 - np.min(depth_image_1)) / (np.max(depth_image_1) - np.min(depth_image_1))
        # depth_norm_2 = (depth_image_2 - np.min(depth_image_2)) / (np.max(depth_image_2) - np.min(depth_image_2))
        
        
        # print("depth_image_shape",depth_colormap_1)
        # print("distance",depth_colormap_1[100][100])
        # depth_colormap_2 = cv2.convertScaleAbs(depth_image_2, alpha=1)
        # print("capture!")
        rgb_image_1     = img_np_1[:, :, :3]
        rgb_image_2     = img_np_2[:, :, :3]
        
        rgbd_image_1    = np.concatenate((rgb_image_1, depth_image_1), axis=2)
        rgbd_image_2    = np.concatenate((rgb_image_2, depth_image_2), axis=2)
        
        # print("size of color", rgb_image_1.shape)
        # cv2.imshow("cam1", np.asanyarray(rgb_image_1))
        # cv2.imshow("cam2", np.asanyarray(rgb_image_2))
        # cv2.waitKey(1)
        # self.finish_ep()
        # depth_stack_1[i].append(depth_image_1)
        # depth_stack_2[i].append(depth_image_2)
        self.gym.end_access_image_tensors(self.sim)
        
        return [rgbd_image_1, rgbd_image_2]   
     
    def store_joints_states(self, env: int, ep: int, joints_data: object, success: bool):
        """
        file = h5py.File("dataset/joint_angles.h5py", 'w')
        group = file.create_group(f"env_{env}_ep_{ep}")
        dataset = file.store_joints_states()
        """
        if success:
            with open(f"dataset/success_seed_{self.seed}/env_{env}_ep_{ep}.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(joints_data.numpy())
        else:
            with open(f"dataset/failure_seed_{self.seed}/env_{env}_ep_{ep}.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(joints_data.numpy())

    def stop_simulation(self):
        print("Done")

        # print(f"Total frozen resets: {frozen_counter}")
        # print(f"Total successes: {success_counter}")
        # cleanup
        # color_writer_1.release()
        # color_writer_2.release()
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        cv2.destroyAllWindows()

class Replay_env():
    def __init__(self, dataset_dir):
        self.num_envs       = 1
        self.dataset_dir    = dataset_dir
        """self.joint_states   =
        self.end_effector_pos =
        self.num_queries    =
        self.query_frequency    ="""
        self.step_count     = 0
        self.sample_count   = 0 
        self.actions, self.obs, self.color_1, self.color_2, self.depths_1, self.depths_2   = self.read_state_data()
        self.ep_len         = len(self.actions)
        self.stats          = dict()
        self.stats['total_pos_diff']    = 0
        self.stats['total_rot_diff']    = 0
        self.stats['mean_pos_diff']     = 0
        self.stats['mean_rot_diff']     = 0
        self.stats['total_pos_RMSEP']   = 0
        self.stats['total_rot_RMSEP']   = 0
        self.RMSEP_pos                  = 0
        self.RMSEP_rot                  = 0 
        self.writer         = SummaryWriter("logs_replay")
        
        
    def step(self, predicted_actions=None):
        # print(f"step_count ={self.step_count}, prediction={predicted_actions}")
        if predicted_actions is not None:
            self.evaluate(predicted_actions) 
        else:
            self.record()
        self.step_count += 1
        
    def reset(self):
        self.step_count = 0 
        self.sample_count = 0
        print(f"Total pos diff = {self.stats['total_pos_diff']}, Total rot diff = {self.stats['total_rot_diff']}")
        print(f"Mean pos diff = {self.stats['total_pos_diff'] / self.ep_len}, Mean rot diff = {self.stats['total_rot_diff'] / self.ep_len}")
        
        self.stats['total_pos_diff']    = 0
        self.stats['total_rot_diff']    = 0
        self.stats['mean_pos_diff']     = 0
        self.stats['mean_rot_diff']     = 0
          
    def read_state_data(self):
        with h5py.File(self.dataset_dir, 'r') as root:
            actions = root['/actions'][()]
            obs = root['/observations'][()]
            color_1_frames  = root['/images/colors/cam1'][()]   
            color_2_frames  = root['/images/colors/cam2'][()]   
            depth_1_frames  = root['/images/depths/cam1'][()]   
            depth_2_frames  = root['/images/depths/cam2'][()]  
            
        print(f"action shape:{actions.shape},   obs shape:{obs.shape}")
        
        return actions, obs, color_1_frames, color_2_frames, depth_1_frames, depth_2_frames
            
    def image_capture(self):
        """
            return one image per camera of shape [H x W x C], where C = 4
        """
        depth_1 = self.depths_1[self.step_count]
        depth_2 = self.depths_2[self.step_count]
        rgb_1   = self.color_1[self.step_count]
        rgb_2   = self.color_2[self.step_count]

        
        depth_1 = depth_1[:, :, np.newaxis] 
        depth_2 = depth_2[:, :, np.newaxis]
        
        rgbd_image_1   = np.concatenate((rgb_1, depth_1), axis=2)
        rgbd_image_2   = np.concatenate((rgb_2, depth_2), axis=2)
        
        return [rgbd_image_1, rgbd_image_2]   
    
    def get_observations(self):
        '''
            return observations
        '''       
        # print(f"step_count = {self.step_count}")
        return self.obs[self.step_count]

    def evaluate(self, predicted_actions):
        self.sample_count += 1
        predicted_pos   = predicted_actions[:3]
        predicted_rot   = predicted_actions[3:]

        current_actions = self.actions[self.step_count]
        actual_pos      = np.array(current_actions[:3])
        actual_rot      = np.array(current_actions[3:])
        
        pos_diff        = (actual_pos - predicted_pos)
        rot_diff        = (actual_rot - predicted_rot)
        
        pos_diff_sq     = pos_diff ** 2
        rot_diff_sq     = rot_diff ** 2
        
        pos_diff_sqrt        = np.sqrt(pos_diff_sq)
        rot_diff_sqrt        = np.sqrt(rot_diff_sq)
        
        self.stats['total_pos_diff']    += pos_diff_sqrt
        self.stats['total_rot_diff']    += rot_diff_sqrt
        
        self.stats['total_pos_RMSEP']    += pos_diff_sq
        self.stats['total_rot_RMSEP']    += rot_diff_sq
        
        RMSEP_pos   = np.mean(np.sqrt(self.stats['total_pos_RMSEP'] / self.sample_count) * 100)
        RMSEP_rot   = np.mean(np.sqrt(self.stats['total_rot_RMSEP'] / self.sample_count) * 100)
        self.RMSEP_pos = RMSEP_pos 
        self.RMSEP_rot = RMSEP_rot
        
        # print(f"t= {self.step_count}, RMSEP= {RMSEP_pos}%, {RMSEP_rot}% ")
        # print(f"Position difference = {pos_diff}, Rotation diff = {rot_diff}")
        self.writer.add_scalars("x",{"actual": actual_pos[0],
                                     "predicted": predicted_pos[0]}, self.step_count)
        self.writer.add_scalars("y",{"actual": actual_pos[1],
                                     "predicted": predicted_pos[1]}, self.step_count)
        self.writer.add_scalars("z",{"actual": actual_pos[2],
                                     "predicted": predicted_pos[2]}, self.step_count)
        
        self.writer.add_scalars("RMSEP",{"pos": RMSEP_pos,
                                     "rot": RMSEP_rot}, self.step_count)
    
    def record(self):
        current_actions = self.actions[self.step_count]
        actual_pos      = np.array(current_actions[:3])
        actual_rot      = np.array(current_actions[3:])
        
        self.writer.add_scalars("x",{"recorded": actual_pos[0]}, self.step_count)
        self.writer.add_scalars("y",{"recorded": actual_pos[1]}, self.step_count)
        self.writer.add_scalars("z",{"recorded": actual_pos[2]}, self.step_count)
        
        
    def stop_simulation(self):
        pass

class Replay_env_old():
    def __init__(self, replay_1_dir, replay_2_dir, dataset_dir):
        self.num_envs       = 1
        self.replay_1_dir   = replay_1_dir
        self.replay_2_dir   = replay_2_dir
        self.dataset_dir    = dataset_dir
        """self.joint_states   =
        self.end_effector_pos =
        self.num_queries    =
        self.query_frequency    ="""
        self.step_count           = 0
        self.frames_1        = self.video_to_frames(self.replay_1_dir)
        self.frames_2        = self.video_to_frames(self.replay_2_dir)
        self.ep_len         = len(self.frames_1)
        self.actions, self.qpos, self.depths_1, self.depths_2         = self.read_state_data()
        self.stats          = dict()
        self.stats['total_pos_diff']    = 0
        self.stats['total_rot_diff']    = 0
        self.stats['mean_pos_diff']     = 0
        self.stats['mean_rot_diff']     = 0
        self.writer         = SummaryWriter("logs_replay")
        
        
    def step(self, predicted_actions=None):
        if predicted_actions is not None:
            self.evaluate(predicted_actions) 
        else:
            self.record()
        self.step_count += 1
        
    def reset(self):
        self.step_count = 0 
        print(f"Total pos diff = {self.stats['total_pos_diff']}, Total rot diff = {self.stats['total_rot_diff']}")
        print(f"Mean pos diff = {self.stats['total_pos_diff'] / self.ep_len}, Mean rot diff = {self.stats['total_rot_diff'] / self.ep_len}")
        
        self.stats['total_pos_diff']    = 0
        self.stats['total_rot_diff']    = 0
        self.stats['mean_pos_diff']     = 0
        self.stats['mean_rot_diff']     = 0
        
    def video_to_frames(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            # Save frame as image
            # frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
            
            frames.append(frame)
            
            # cv2.imwrite(frame_filename, frame)
            
            # print(f"Saved {frame_filename}")
            # frame_count += 1
        cap.release()
        # print(f"Done. Total frames saved: {frame_count}")
        return frames
    
    def read_state_data(self):
        with h5py.File(self.dataset_dir, 'r') as root:
            actions = root['/action'][()]
            qpos = root['/observations/qpos'][()]
            depth_1_frames  = root['/depths/cam1'][()]   
            depth_2_frames  = root['/depths/cam2'][()]  
        
        print(f"action shape:{actions.shape},   qpos shape:{qpos.shape}")
        
        
        return actions, qpos, depth_1_frames, depth_2_frames
            
    def image_capture(self):
        """
            return one image per camera of shape [H x W x C], where C = 4
        """
        depth_1 = self.depths_1[self.step_count]
        depth_2 = self.depths_2[self.step_count]
        rgb_1   = self.frames_1[self.step_count]
        rgb_2   = self.frames_2[self.step_count]
        
        depth_1 = depth_1[:, :, np.newaxis] 
        depth_2 = depth_2[:, :, np.newaxis]
        
        rgbd_image_1   = np.concatenate((rgb_1, depth_1), axis=2)
        rgbd_image_2   = np.concatenate((rgb_2, depth_2), axis=2)
        
        
        
        return [rgbd_image_1, rgbd_image_2]   
    
    def get_observations(self):
        '''
            return qpos
        '''       
        return self.qpos[self.step_count]

    def evaluate(self, predicted_actions):
        
        predicted_pos   = predicted_actions[:3]
        predicted_rot   = predicted_actions[3:]

        current_actions = self.actions[self.step_count]
        actual_pos      = np.array(current_actions[:3])
        actual_rot      = np.array(current_actions[3:])
        
        pos_diff        = np.sqrt((actual_pos - predicted_pos) ** 2)
        rot_diff        = np.sqrt((actual_rot - predicted_rot) ** 2)
        self.stats['total_pos_diff']    += pos_diff
        self.stats['total_rot_diff']    += rot_diff
        print(f"t= {self.step_count}, predicted pos= {predicted_pos}  Actual pos = {actual_pos}")
        # print(f"Position difference = {pos_diff}, Rotation diff = {rot_diff}")
        self.writer.add_scalars("x",{"actual": actual_pos[0],
                                     "predicted": predicted_pos[0]}, self.step_count)
        self.writer.add_scalars("y",{"actual": actual_pos[1],
                                     "predicted": predicted_pos[1]}, self.step_count)
        self.writer.add_scalars("z",{"actual": actual_pos[2],
                                     "predicted": predicted_pos[2]}, self.step_count)
    
    def record(self):
        current_actions = self.actions[self.step_count]
        actual_pos      = np.array(current_actions[:3])
        actual_rot      = np.array(current_actions[3:])
        
        self.writer.add_scalars("x",{"recorded": actual_pos[0]}, self.step_count)
        self.writer.add_scalars("y",{"recorded": actual_pos[1]}, self.step_count)
        self.writer.add_scalars("z",{"recorded": actual_pos[2]}, self.step_count)
        
        
    def stop_simulation(self):
        pass
    
    
    