
import math
import numpy as np
import torch
import random
import time
import copy
import os
import shutil
import cv2
import csv
import h5py
import argparse
# def store_joints_states(env: int, ep: int, joints_data: object, success: bool):
#     file = h5py.File("dataset/joint_angles.h5py", 'w')
#     group = file.create_group(f"env_{env}_ep_{ep}")
#     dataset = file.store_joints_states()
#     if success:
#         with open(f"dataset/success_seed_{args.seed}/env_{env}_ep_{ep}.csv", 'w') as f:
#             writer = csv.writer(f)
#             writer.writerows(joints_data.numpy())
#     else:
#         with open(f"dataset/failure_seed_{args.seed}/env_{env}_ep_{ep}.csv", 'w') as f:
#             writer = csv.writer(f)
#             writer.writerows(joints_data.numpy())

# seed = args.seed
# joints_obs_dir = "dataset"
# success_joints_obs_dir     = os.path.join(joints_obs_dir, f"success_seed_{seed}")
# failure_joints_obs_dir     = os.path.join(joints_obs_dir, f"failure_seed_{seed}")
# videos_dir      = "video"
# success_videos_dir     = os.path.join(videos_dir, f"success_seed_{seed}")
# failure_videos_dir     = os.path.join(videos_dir, f"failure_seed_{seed}")

# TODO: make a function to generate images from a video
def video_to_frames(video_path):
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
    


def retrieve_data(ep, obs_dir, actions_dir, colors_1_dir, colors_2_dir, depths_dir):
   
    actions_data = []
    
    observations_data    = []
    joint_angles    = []
    EE_pose         = []
    
    
    # Read the observations: joint angles (8) + End-effector pose (3 + 4), total = 15
    with open(obs_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  
                observations_data.append([float(val) for val in row])
    
    # Read the actions: controller's position (3) + quarternion rotation (4)                
    with open(actions_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  
                actions_data.append([float(val) for val in row])
                
    # assert len(qpos_data) == len(action_data), f"Data mismatch! qpos={len(qpos_data)}, actions={len(action_data)}"
    print(f"Read {len(observations_data)} joint positions and {len(actions_data)} actions data!")

    # Convert video to frames
    rgb_1_frames    = video_to_frames(colors_1_dir)  # [T, H, W, C]
    rgb_2_frames    = video_to_frames(colors_2_dir)
    
    with h5py.File(depths_dir, 'r') as root:
        depth_1_frames  = root['/depths/cam1'][()]   
        depth_2_frames  = root['/depths/cam2'][()]   
    
    # size correction checks
    assert len(rgb_1_frames) == len(rgb_2_frames) == len(depth_1_frames) == len(depth_2_frames) == len(actions_data) == len(observations_data)  , \
        f"Data mismatch! qpos={len(observations_data)}, actions={len(actions_data)}, rgb cam1={len(rgb_1_frames)}, rgb cam2={len(rgb_2_frames)}, depth cam1={len(depth_1_frames)}, depth cam2={len(depth_2_frames)}"

    # Package outputs
    # qpos_data = action_data
    # qvel_data = []         # Placeholder if not available
    sim       = True       # Metadata
    compress  = False      # Metadata

    return [ep, actions_data, observations_data, rgb_1_frames, rgb_2_frames, depth_1_frames, depth_2_frames , sim, compress]


def create_dataset(storage_dir, ep, action_data , observations_data, rgb_1_data, rgb_2_data, depth_1_data, depth_2_data, sim=True, compress=False):
    with h5py.File(os.path.join(storage_dir, f"episode_{ep}.hdf5"), 'w') as f:
        f.create_dataset('actions', data=action_data, compression="gzip", compression_opts=7) # if action is joint angles, then dim=[k x 6], else if action is goal pose, then dim = [k x 7] (3+4)
        f.create_dataset('observations', data=observations_data, compression="gzip", compression_opts=7) # [k x 6]
        # obs_grp.create_dataset('qvel', data=qvel_data, compression="gzip", compression_opts=7) # [k x 6]
        
        img_grp     = f.create_group('images')
        
        color_grp   = img_grp.create_group('colors')
        color_grp.create_dataset('cam1', data=rgb_1_data, compression="gzip", compression_opts=4) # [k x H x W x C] C=3 
        color_grp.create_dataset('cam2', data=rgb_2_data, compression="gzip", compression_opts=4) # [k x H x W x C] C=3
        
        depth_grp   = img_grp.create_group('depths')
        depth_grp.create_dataset('cam1', data=depth_1_data, compression="gzip", compression_opts=4) # [k x H x W x C] C=1
        depth_grp.create_dataset('cam2', data=depth_2_data, compression="gzip", compression_opts=4) # [k x H x W x C] C=1
        
        # Optional attributes
        f.attrs['sim'] = sim
        f.attrs['compress'] = compress
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',          action='store', type=int, default=0,                help='save_every',required=True)
    parser.add_argument('--seed',       action='store', type=int, default=12,               help='save_every',required=True)
    parser.add_argument('--num_envs',   action='store', type=int, default=1,                help='save_every',required=True)
    parser.add_argument('--src_dir',    action='store', type=str, default="raw_datasets",   help='save_every',required=True)
    args = parser.parse_args()
    # main(vars(parser.parse_args()))
    
    print(f"args={args}")
    joint_obs_dir   = 'dataset'
    i               = args.i
    seed            = args.seed
    env             = 0
    num_envs        = args.num_envs
    success         = True
    # target_storage_dir = f"datasets/seed_{seed}"
    # target_storage_dir = f"/media/ucluser/Extreme SSD/datasets/seed_{seed}"
    target_storage_dir = f"/media/ucluser/PortableSSD/model_datasets/seed_{seed}"
    # target_storage_dir   = f"/home/wee_ucl/workspace/datasets/seed_{seed}"
    os.makedirs(target_storage_dir, exist_ok=True)
    
    # joints_obs_dir  = "dataset"
    # videos_dir      = "videos"
    # joints_obs_dir  = "/media/ucluser/Extreme SSD/dataset"
    # src_dataset_dir  = "/media/ucluser/PortableSSD/raw_datasets"
    src_dataset_dir  = "raw_datasets"
    # videos_dir      = "/media/ucluser/Extreme SSD/videos"
    suffix = f"success_seed_{seed}" if success else f"failure_seed_{seed}"
    src_dataset_dir = os.path.join(src_dataset_dir, suffix)
    # videos_dir     = os.path.join(videos_dir, suffix)
    assert src_dataset_dir
    # assert videos_dir
    
    
    
    # videos_obs_sessions = [d for d in os.listdir(videos_dir)  ]
    # assert len(joints_obs_sessions) == len(videos_obs_sessions), \
        # f"Number of episodes mismatch: joints_data={len(joints_obs_sessions)}, videos_data={len(videos_obs_sessions)}"
        
    for env in range(num_envs):
        # obs_dir = os.path.join(joints_obs_dir, f"env_{env}_ep_{ep}.csv")
        src_dataset_dirr = os.path.join(src_dataset_dir, f"env_{env}")
        # videos_dirr     = os.path.join(joints_obs_dir, f"env_{env}/videos")
        observations_dir      = os.path.join(src_dataset_dir, f"env_{env}/observations")
        joints_obs_sessions = [d for d in os.listdir(observations_dir)]
        print(f"size of env_{env}: {len(joints_obs_sessions)}")
        for j in range(len(joints_obs_sessions)):  #- 1):
            j += args.i
            # Construct file paths
            print(f"working on ep {j}")
            obs_dir      = os.path.join(src_dataset_dirr, f"observations/env_{env}_ep_{j}.csv")
            actions_dir     = os.path.join(src_dataset_dirr, f"actions/env_{env}_ep_{j}.csv")
            depths_dir      = os.path.join(src_dataset_dirr, f"depths/episode_{j}.hdf5")
            colors_1_dir       = os.path.join(src_dataset_dirr, f"colors/color_1_env_{env}_ep_{j}.avi")
            colors_2_dir       = os.path.join(src_dataset_dirr, f"colors/color_2_env_{env}_ep_{j}.avi")
            
            # obs_dir = os.path.join(joints_obs_dir, joints_obs_sessions[j])
            # episode_cam1_dir   = os.path.join(videos_dir, )
            # episode_cam2_dir   = os.path.join(videos_dir,)
            if not os.path.exists(obs_dir):
                continue
            # Ensure files exist
            assert os.path.exists(obs_dir),  f"Missing joint file: {obs_dir}"
            assert os.path.exists(actions_dir), f"Missing joint file: {actions_dir}"
            assert os.path.exists(colors_1_dir),   f"Missing camera 1 rgb file: {colors_1_dir}"
            assert os.path.exists(colors_2_dir),   f"Missing camera 2 rgb file: {colors_2_dir}"
            assert os.path.exists(depths_dir),  f"Missing camera 1 depth file: {depths_dir}"
            
            create_dataset(target_storage_dir, *retrieve_data(i, obs_dir, actions_dir, colors_1_dir, colors_2_dir, depths_dir))
            i += 1
    
    print(f"Done! Total episodes: {i}")