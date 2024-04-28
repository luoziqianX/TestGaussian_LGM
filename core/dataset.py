import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter
import pytorch_lightning as pl
from kiui.cam import orbit_camera

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ThumanDataset(Dataset):
    def __init__(self,
                 opt: Options,
                 dataset_dir,
                 num_frames,
                 control_mode: str = 'smplx',
                 i_start=0,
                 iters=2445,
                 training: bool = False,
                 device: str = 'cuda:0',
                 use_half=False):
        assert 21 % num_frames == 0
        super().__init__()
        self.dataset_dir = dataset_dir
        self.num_frames = num_frames
        self.control_mode = control_mode
        self.istart = i_start
        self.iters = iters
        self.training = training
        self.opt = opt
        self.gap = 21 // num_frames

        items = ['{:04}'.format(i) for i in range(self.istart, self.iters)]
        if training:
            items = items[:int((iters - i_start) * 0.8)]
        else:
            items = items[int((iters - i_start) * 0.8):]
        self.items = items

        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1
        self.device = device
        self.use_half = use_half

    def __len__(self):
        return len(self.items * 21)

    def get_campos(self, frame_id):
        elevation = 10
        azimuth = np.linspace(0, 360, 21 + 1)[:-1] % 360
        azi = azimuth[frame_id]
        cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=self.opt.cam_radius, opengl=True)).unsqueeze(0)

        cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2)  # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix  # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3]
        return cam_pos, cam_poses, cam_view, cam_view_proj

    def __getitem__(self, idx):
        data_idx = idx // 21
        start_frame = idx % 21
        frame_list_dir = ['{:04}.png'.format((start_frame + self.gap * i) % 21) for i in range(self.num_frames)]
        frame_list = [os.path.join(self.dataset_dir, self.items[data_idx], 'images', frame) for frame in frame_list_dir]
        hint_list = [os.path.join(self.dataset_dir, self.items[data_idx], self.control_mode, frame) for frame in
                     frame_list_dir]
        frames_id = [(start_frame + self.gap * i) % 21 for i in range(self.num_frames)]
        images = []
        hints = []
        cam_poses = []
        cam_views = []
        camposes_list = []
        cam_views_proj = []
        for frame_id, frame_path, hint_path in zip(frames_id, frame_list, hint_list):
            image = np.array(cv2.imread(frame_path)).astype(np.uint8)
            b = image[:, :, 0]
            g = image[:, :, 1]
            r = image[:, :, 2]
            image = np.stack([r, g, b], axis=-1)
            image = torch.from_numpy(image.astype(np.float32) / 255)
            hint = np.array(cv2.imread(hint_path)).astype(np.uint8)
            b = hint[:, :, 0]
            g = hint[:, :, 1]
            r = hint[:, :, 2]
            hint = np.stack([r, g, b], axis=-1)
            hint = torch.from_numpy(hint.astype(np.float32) / 255)
            campos, camposes, cam_view, cam_view_proj = self.get_campos(frame_id)
            images.append(image)
            hints.append(hint)
            cam_poses.append(campos)
            cam_views.append(cam_view)
            cam_views_proj.append(cam_view_proj)
            camposes_list.append(camposes)

        images = torch.stack(images, dim=0)
        images = images.permute(0, 3, 1, 2).contiguous()
        hints = torch.stack(hints, dim=0)
        hints = hints.permute(0, 3, 1, 2).contiguous()
        cam_poses = torch.stack(cam_poses, dim=0)
        cam_views = torch.stack(cam_views, dim=0)
        cam_views_proj = torch.stack(cam_views_proj, dim=0)
        camposes_list = torch.stack(camposes_list, dim=0)

        imaages_input = TF.normalize(images, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        imaages_input = F.interpolate(imaages_input, size=(self.opt.input_size, self.opt.input_size),
                                      mode='bilinear', align_corners=False)
        images_output = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size),
                                      mode='bilinear', align_corners=False)
        hints = F.interpolate(hints, size=(self.opt.input_size, self.opt.input_size),
                              mode='bilinear', align_corners=False)
        # build rays for input views
        rays_embeddings = []
        for i in range(self.num_frames):
            rays_o, rays_d = get_rays(camposes_list[i, 0], self.opt.input_size, self.opt.input_size,
                                      self.opt.fovy)  # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)  # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()  # [V, 6, h, w]
        final_input = torch.cat([imaages_input, rays_embeddings], dim=1)  # [V=4, 9, H, W]
        input = final_input

        result = dict(
            input=input,
            images_output=images_output,
            hint=hints,
            cam_pos=cam_poses[:, 0],
            cam_view=cam_views[:, 0],
            cam_view_proj=cam_views_proj[:, 0], )

        for key in result.keys():
            if self.use_half:
                result[key] = result[key].half()
            result[key] = result[key].to(self.device)

        return result
