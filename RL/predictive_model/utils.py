import random
import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf

import IPython
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def loss_reducing(loss: torch.Tensor):
    total_loss = sum([x for x in loss.values()])
    return total_loss
    
def pack_sequence_dim(x):
    ''' Pack the batch and seqence_length dimension.'''
    if isinstance(x, torch.Tensor):
        b, s = x.shape[:2]
        return x.view(b * s, *x.shape[2:])

    if isinstance(x, list):
        return [pack_sequence_dim(elt) for elt in x]

    output = {}
    for key, value in x.items():
        output[key] = pack_sequence_dim(value)
    return output


def unpack_sequence_dim(x, b, s):
    ''' Unpack the batch and seqence_length dimension.'''
    if isinstance(x, torch.Tensor):
        return x.view(b, s, *x.shape[1:])

    if isinstance(x, list):
        return [unpack_sequence_dim(elt, b, s) for elt in x]

    output = {}
    for key, value in x.items():
        output[key] = unpack_sequence_dim(value, b, s)
    return output


def stack_list_of_dict_tensor(output, dim=1):
    ''' Stack list of dict of tensors'''
    new_output = {}
    for outter_key, outter_value in output.items():
        if len(outter_value) > 0:
            new_output[outter_key] = {}
            for inner_key in outter_value[0].keys():
                new_output[outter_key][inner_key] = torch.stack(
                    [x[inner_key] for x in outter_value], dim=dim)
    return new_output

def _interpolate_resize(x, size, mode=tvf.InterpolationMode.NEAREST):
    '''Resize the tensor with interpolation
    
    Args:
        x: Tensor of shape [N, C, H, W] or [C, H, W]
        size: Tuple (height, width) for output size
        mode: Interpolation mode
    '''
    import torch.nn.functional as F
    
    original_shape = x.shape
    
    # Ensure 4D tensor [N, C, H, W]
    if x.ndim == 3:
        # [C, H, W] -> [1, C, H, W]
        x = x.unsqueeze(0)
        needs_squeeze = True
    elif x.ndim == 4:
        needs_squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {x.ndim}D with shape {original_shape}")
    
    # Use torch.nn.functional.interpolate instead of tvf.resize
    # This is more reliable and gives us better control
    mode_map = {
        tvf.InterpolationMode.NEAREST: 'nearest',
        tvf.InterpolationMode.BILINEAR: 'bilinear',
        tvf.InterpolationMode.BICUBIC: 'bicubic',
    }
    
    interp_mode = mode_map.get(mode, 'bilinear')
    align_corners = None if interp_mode == 'nearest' else False
    
    x = F.interpolate(
        x, 
        size=size,  # (H, W)
        mode=interp_mode,
        align_corners=align_corners,
        antialias=True if interp_mode != 'nearest' else False
    )
    
    # Remove batch dimension if we added it
    if needs_squeeze:
        x = x.squeeze(0)  # [1, C, H, W] -> [C, H, W]
    
    return x

def interpolate_resize(x, size, mode=tvf.InterpolationMode.NEAREST):
    '''Resize the tensor with interpolation
    '''
    # x = tvf.resize(x, size, interpolation=mode, antialias=True)
    # x.unsqueeze(1)
    was_5d = x.ndim == 5
    if was_5d:
        b, s, c, h, w = x.shape
        x = x.reshape(b * s, c, h, w)
    x = tvf.resize(x, size, interpolation=mode, antialias=True)
    if was_5d:
        x = x.reshape(b, s, c, *x.shape[-2:])
    return x


def compose_rgb_labels(batch):
    '''Compose RGB labels at different downsampling factors
    
    Args:
        batch: Either a dict with 'rgb_label_1' or a tensor
    
    Returns:
        dict: Dictionary with rgb labels at different scales
    '''
    # 1. Get the image tensor from batch
    if isinstance(batch, dict):
        if 'rgb_label_1' in batch:
            img = batch['rgb_label_1']
        elif 'rgb' in batch:
            img = batch['rgb']
        else:
            img = next(iter(batch.values()))
    else:
        img = batch

    # 2. If there is a camera dimension, select ONE camera
    #    img shape: [B, num_cams, C, H, W]
    # if img.ndim == 5:
    #     # pick camera cam_idx
    #     img = img[:, 0]      # -> [B, C, H, W]
    img = img[:, :, :, :3]  # drop depth   [B, cam_num, frames, C, H, W] = [B, 2, 5, 3, H, W]
    # img = img.unsqueeze(1)
    # 3. Build pyramid of downsampled labels
    output = {}
    # print("img shape: ", img.shape)
    # img shape = [batch, cam_name, frames, c, h, w]
    batch, cam_name, frames, c, h, w = img.shape
    for cam in range(cam_name):
        for frame in range(frames):
            cam_img = img[:, cam, frame]
            cam_img = cam_img.unsqueeze(1)
            output[f'rgb_cam_{cam+1}_label_1_{frame}'] = cam_img
            assert cam_img.ndim == 5 , f"shape of cam_{cam+1}_img is {cam_img.shape}"
    
    # output['rgb__label_1'] = img
    h, w = img.shape[-2:]
    
    # Create downsampled versions
    for downsample_factor in [2, 4]:
        for cam in range(cam_name): 
            for frame in range(frames):
                size = (h // downsample_factor, w // downsample_factor)
                previous_label_factor = downsample_factor // 2
                
                output[f'rgb_cam_{cam+1}_label_{downsample_factor}_{frame}'] = interpolate_resize(
                    output[f'rgb_cam_{cam+1}_label_{previous_label_factor}_{frame}'],
                    size,
                    mode=tvf.InterpolationMode.BILINEAR,
                )
    # print("*****************batch rgb_label_1 shape: ", output['rgb_label_1'].shape)
    # print("*****************batch rgb_label_2 shape: ", output['rgb_label_2'].shape)
    # print("*****************batch rgb_label_4 shape: ", output['rgb_label_4'].shape)
    return output

def _compose_rgb_labels(batch):
    '''Compose RGB labels at different downsampling factors
    
    Args:
        batch: Either a dict with 'rgb_label_1' or a tensor
    
    Returns:
        dict: Dictionary with rgb labels at different scales
    '''
    # Handle case where batch is just a tensor
    if isinstance(batch, torch.Tensor):
        output = {}
        output['rgb_label_1'] = batch
        h, w = batch.shape[-2:]
    else:
        output = batch.copy() if isinstance(batch, dict) else {'rgb_label_1': batch}
        h, w = output['rgb_label_1'].shape[-2:]
    
    # Create downsampled versions
    for downsample_factor in [2, 4]:
        size = (h // downsample_factor, w // downsample_factor)
        previous_label_factor = downsample_factor // 2
        
        output[f'rgb_label_{downsample_factor}'] = interpolate_resize(
            output[f'rgb_label_{previous_label_factor}'],
            size,
            mode=tvf.InterpolationMode.BILINEAR,
        )
    
    return output

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class):
        super(EpisodicDataset).__init__()
        self.episode_ids        = episode_ids
        self.dataset_path_list  = dataset_path_list
        self.camera_names       = camera_names
        self.norm_stats         = norm_stats
        self.episode_len        = episode_len
        self.chunk_size         = chunk_size
        self.cumulative_len     = np.cumsum(self.episode_len)
        self.max_episode_len    = max(episode_len)
        self.policy_class       = policy_class
        if self.policy_class    == 'Diffusion':
            self.augment_images = True
        else:
            self.augment_images = False
        self.transformations    = None
        self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

    # def __len__(self):
    #     return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index   = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts        = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id      = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts    = self._locate_transition(index)
        dataset_path            = self.dataset_path_list[episode_id]
        try:
            # print(dataset_path)
            with h5py.File(dataset_path, 'r') as root:
                try: # some legacy data does not have this attribute
                    is_sim = root.attrs['sim']
                except:
                    is_sim = False
                compressed = root.attrs.get('compress', False)
                action = root['/actions'][()]
                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                # get observation at start_ts only
                obs = root['/observations'][start_ts]
                # qvel = root['/observations/qvel'][start_ts]
                image_dict  = dict()
                depth_dict  = dict()
                
                for cam_name in self.camera_names:
                    # print("cam:", cam_name)
                    image_dict[cam_name] = root[f'/images/colors/{cam_name}'][start_ts:start_ts+3]
                    depth_dict[cam_name] = root[f'/images/depths/{cam_name}'][start_ts:start_ts+3]
                if compressed:
                    for cam_name in image_dict.keys():
                        decompressed_image      = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name]    = np.array(decompressed_image)
                        decompressed_depth      = cv2.imdecode(depth_dict[cam_name], 1)
                        depth_dict[cam_name]    = np.array(decompressed_depth)

                # Pad images and depths to ensure they have exactly 3 frames
                for cam_name in self.camera_names:
                    img = image_dict[cam_name]
                    dep = depth_dict[cam_name]
                    # print(f"shape of img {img.shape}")
                    # print(f"shape of dep {dep.shape}")
                    # Ensure img and dep are 4D (frames, H, W, C)
                    if img.ndim == 3:
                        # If decompressed, each frame is a single image (H, W, C)
                        # This shouldn't happen as cv2.imdecode on a sequence should give multiple frames
                        img = np.expand_dims(img, axis=0)
                    if dep.ndim == 3:
                        dep = np.expand_dims(dep, axis=0)
                    
                    # Pad to 3 frames if fewer frames are available
                    if img.shape[0] < 3:
                        pad_frames = 3 - img.shape[0]
                        # Repeat the last frame to pad
                        last_img = img[-1:].copy()  # shape: (1, H, W, C)
                        last_dep = dep[:, -1:].copy()  # shape: (1, H, W, C)
                        image_dict[cam_name] = np.concatenate([img] + [last_img] * pad_frames, axis=0)
                        depth_dict[cam_name] = np.concatenate([dep] + [last_dep] * pad_frames, axis=1)
                    else:
                        image_dict[cam_name] = img
                        depth_dict[cam_name] = dep
                
                # get all actions after and including start_ts
                if is_sim:
                    action = action[start_ts:]
                    action_len = episode_len - start_ts
                else:
                    action = action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                    action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

            # self.is_sim = is_sim
            padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(self.max_episode_len)
            is_pad[action_len:] = 1

            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[:self.chunk_size]

            # new axis for different cameras
            all_cam_images  = []
            all_cam_depths  = []
            # depth_means     = []
            # depth_stds      = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
                depth_dict[cam_name] = (np.array(depth_dict[cam_name])  - self.norm_stats[f"depth_mean_{cam_name}"]) / self.norm_stats[f"depth_std_{cam_name}"]
                all_cam_depths.append(depth_dict[cam_name])
                # depth_means.append(np.array(depth_dict[cam_name]).flatten().mean(dim=[0]).float())
                # depth_stds.append(torch.clip(np.array(depth_dict[cam_name]).flatten().std(dim=[0]).float(), 1e-2, np.inf))
                        
            
            all_cam_images  = np.stack(all_cam_images,  axis=0)
            all_cam_depths  = np.stack(all_cam_depths,  axis=0)
            # depth_mean      = np.array(all_cam_depths).flatten().mean()
            # depth_std       = np.clip(np.array(all_cam_depths).flatten().std(), 1e-2, np.inf)

                
            # depth_image = torch.from_numpy(depth_image).float().cuda().unsqueeze(1).unsqueeze(0)
            # depth_means     = np.stack(depth_means,     axis=0).reshape(-1,1,1,1)
            # depth_stds      = np.stack(depth_stds,      axis=0).reshape(-1,1,1,1)
            # all_cam_depths  = (all_cam_depths - depth_means) / depth_stds   #normalize the depth information each camera differently
            # all_cam_depths  = (all_cam_depths - depth_mean) / depth_std     #normalize the depth information each camera the same way
            
            # construct observations
            image_data  = torch.from_numpy(all_cam_images.copy())
            depth_data  = torch.from_numpy(all_cam_depths.copy())
            obs_data   = torch.from_numpy(obs).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad      = torch.from_numpy(is_pad).bool()
            # print("shape of image data:", {image_data.shape})
            # print("shape of depth data:", {depth_data.shape})
            # channel last  # [cam_names frames height width channels]
            image_data = torch.einsum('k f h w c -> k f c h w', image_data)
            depth_data = torch.einsum('k c f h w -> k f c h w', depth_data)

            # augmentation
            if self.transformations is None:
                print('Initializing transformations')
                original_size = image_data.shape[3:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    transforms.Resize(original_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
                ]

            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)

            # normalize image and change dtype to float
            image_data = image_data / 255.0
           
            if self.policy_class == 'Diffusion':
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            
            visual_data = torch.concatenate([image_data, depth_data], axis = 2)
            obs_data = (obs_data - self.norm_stats["obs_mean"]) / self.norm_stats["obs_std"]

        except Exception as e:
            print(f"Error loading {dataset_path} in __getitem__")
            import traceback
            traceback.print_exc()
            raise e  # re-raise to stop execution
        """except:
            print(f'Error loading {dataset_path} in __getitem__')
            quit()"""

        # print(image_data.dtype, obs_data.dtype, action_data.dtype, is_pad.dtype)
        return visual_data, obs_data, action_data, is_pad


def get_norm_stats(dataset_path_list, short: bool ):
    # all_qpos_data   = []
    all_obs_data = []
    all_action_data = []
    all_episode_len = []
    all_depths_data = dict()
    depth_running_sum     = dict()
    depth_running_sq_sum  = dict()
    depth_count           = dict()
    if short: 
        for dataset_path in dataset_path_list:
            try:
                with h5py.File(dataset_path, 'r') as root:
                    obs    = root['/observations'][()]
            except Exception as e:
                print(f'Error loading observations from {dataset_path} in get_norm_stats')
                print(e)
                quit()
            all_episode_len.append(len(obs))
            stats = None
        
    else:
        for dataset_path in dataset_path_list:
            try:
                with h5py.File(dataset_path, 'r') as root:
                    obs    = root['/observations'][()]
                    actions  = root['/actions'][()]    
                    
                    for cam in root['/images/depths'].keys():
                        # if cam not in all_depths_data:
                            # all_depths_data[cam] = []
                        # all_depths_data[cam].append(root[f'/depths/{cam}'][()])
                        depth = root[f'/images/depths/{cam}'][()]
                        if cam not in depth_running_sum:
                            depth_running_sum[cam]      = 0
                            depth_running_sq_sum[cam]   = 0
                            depth_count[cam]            = 0
                        depth_running_sum[cam]    += depth.sum()
                        depth_running_sq_sum[cam] += (depth ** 2).sum()
                        depth_count[cam]          += depth.size 
                    # print(f"depth: type={type(depth)}, shape={depth.shape}, size={depth.size}")
                    
                        
                        
            except Exception as e:
                print(f'Error loading action and/or depth info from {dataset_path} in get_norm_stats')
                print(e)
                quit()
            # all_qpos_data.append(torch.from_numpy(qpos))
            all_obs_data.append(torch.from_numpy(obs))
            all_action_data.append(torch.from_numpy(actions))
            all_episode_len.append(len(obs))
            
        # all_qpos_data   = torch.cat(all_qpos_data,  dim=0)
        all_obs_data   = torch.cat(all_obs_data,  dim=0)
        all_action_data = torch.cat(all_action_data,dim=0)
        
        """depth_means     = {}
        depth_stds      = {}
        for cam, depth_list in all_depths_data.items():
            depth_combined      = np.concatenate(depth_list, axis = 0)
            depth_tensor        = torch.from_numpy(depth_combined)
            depth_means[cam]    = depth_tensor.mean().float()
            stds                = depth_tensor.std().float()
            depth_stds[cam]     = torch.clip(stds, 1e-2, np.inf)"""
        
        # normalize action data
        action_mean     = all_action_data.mean(dim=[0]).float()
        action_std      = all_action_data.std(dim=[0]).float()
        action_std      = torch.clip(action_std, 1e-2, np.inf) # clipping

        # normalize observations data
        obs_mean       = all_obs_data.mean(dim=[0]).float()
        obs_std        = all_obs_data.std(dim=[0]).float()
        obs_std        = torch.clip(obs_std, 1e-2, np.inf) # clipping

        action_min      = all_action_data.min(dim=0).values.float()
        action_max      = all_action_data.max(dim=0).values.float()

        eps = 0.0001
        stats = {"action_mean": action_mean.numpy(), 
                "action_std": action_std.numpy(),
                "action_min": action_min.numpy() - eps,
                "action_max": action_max.numpy() + eps,
                "obs_mean": obs_mean.numpy(),
                "obs_std": obs_std.numpy()
                }
        
        # normalize depth data        
        for cam in depth_running_sum:
            mean    = depth_running_sum[cam] / depth_count[cam]
            var     = depth_running_sq_sum[cam] / depth_count[cam] - mean ** 2
            std     = np.clip(np.sqrt(var), 1e-2, np.inf)
            stats[f"depth_mean_{cam}"]  = mean 
            stats[f"depth_std_{cam}"]   = std 
            
        print("#------------------------STATS--------------------------#")
        print(stats)
        print("#-------------------------------------------------------#")
    return stats, all_episode_len


def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99):
    dataset_dirr = dataset_dir_l
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    # print(f"dataset len {len(dataset_dir_l)}")
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')

    # obtain normalization stats for qpos and action
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     print('Loaded pretrain dataset stats')
    _, all_episode_len = get_norm_stats(dataset_path_list, short=True)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    
    # obtain normalization stats for qpos and action
    if load_pretrain:
        with open(os.path.join(dataset_dirr, 'dataset_stats.pkl'), 'rb') as f:
            norm_stats = pickle.load(f)
        print('Loaded pretrain dataset stats')
    else:
        norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]), short= False)
        print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class)
    train_num_workers = 5
    val_num_workers = 5
    print(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=2)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim



### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])    

    return peg_pose, socket_pose

### helper functions

def _compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def compute_dict_mean(epoch_dicts):
    if not epoch_dicts:
        return {}
    keys = epoch_dicts[0].keys()
    result = {}
    for k in keys:
        vals = [d[k] for d in epoch_dicts]
        if isinstance(vals[0], torch.Tensor):
            stacked = torch.stack(vals, dim=0)
            result[k] = stacked.mean(dim=0)
        else:
            result[k] = sum(vals) / len(vals)
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

