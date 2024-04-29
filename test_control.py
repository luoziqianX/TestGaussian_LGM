# LGM import
import os
import tyro
import imageio
import cv2
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import gradio as gr
import argparse

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options, config_defaults
from core.models import LGM, ControlLGM
from mvdream.pipeline_mvdream import MVDreamPipeline

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class defaut_name:
    output_ply_name = 'pointclouds_control.ply'
    output_video_name = 'video_control.mp4'


def load_input(args):
    input_dir = args.input_dir
    control_type = args.control_type
    images = torch.zeros(21, 576, 576, 3).type(torch.float32).to(args.device)
    hints = torch.zeros(21, 576, 576, 3).type(torch.float32).to(args.device)
    for i in range(21):
        idx = (args.istart + i) % 21
        image_path = os.path.join(input_dir, 'images', "{:04}.png".format(idx))
        hint_path = os.path.join(input_dir, control_type, "{:04}.png".format(idx))
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = np.array(image_np, dtype=np.float32) / 255.0
        hint_np = cv2.imread(hint_path)
        hint_np = cv2.cvtColor(hint_np, cv2.COLOR_BGR2RGB)
        hint_np = np.array(hint_np, dtype=np.float32) / 255.0
        images[idx] = torch.tensor(image_np)

    images = images.permute(0, 3, 1, 2)
    images = images.to(args.device)
    hints = hints.permute(0, 3, 1, 2)
    hints = hints.to(args.device)
    return images, hints


def test(args):
    device = args.device
    opt = config_defaults['big']
    opt.resume = args.checkpoints_path
    opt.num_frames = 21

    os.makedirs(args.output_dir, exist_ok=True)
    output_ply_name = os.path.join(args.output_dir, defaut_name.output_ply_name)
    output_video_name = os.path.join(args.output_dir, defaut_name.output_video_name)

    model = ControlLGM(opt, num_frames=args.num_frames)
    if args.checkpoints_path:
        model.load_state_dict(torch.load(args.checkpoints_path)['model'], strict=False)
    model = model.half()
    model = model.to(device)
    model.eval()

    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=args.device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    images, hints = load_input(args)
    input_image = F.interpolate(images, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    rays_embeddings = model.prepare_default_rays(device, num_frames=opt.num_frames, elevation=-10)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0)
    hints = F.interpolate(hints, size=(opt.input_size, opt.input_size),
                          mode='bilinear', align_corners=False).unsqueeze(0)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image, hints)

        # save gaussians
        model.gs.save_ply(gaussians, output_ply_name)

        # render 360 video
        images = []
        elevation = -10
        if opt.fancy_video:
            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                cam_poses = torch.from_numpy(
                    orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(
                    0).to(device)

                cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2)  # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix  # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3]  # [V, 3]

                scale = min(azi / 360, 1)

                image = \
                    model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0),
                                    scale_modifier=scale)['image']
                images.append(
                    (image.squeeze(1).permute(0, 2, 3, 1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                cam_poses = torch.from_numpy(
                    orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(
                    0).to(device)

                cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2)  # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix  # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3]  # [V, 3]

                image = \
                    model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0),
                                    scale_modifier=1)['image']
                images.append(
                    (image.squeeze(1).permute(0, 2, 3, 1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

    images = np.concatenate(images, axis=0)
    imageio.mimwrite(output_video_name, images, fps=30)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./inputs')
    parser.add_argument('--istart', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoints_path', type=str, default='pretrained/model_fp16.safetensors')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--control_type', type=str, default='smplx')
    parser.add_argument('--num_frames', type=int, default=21)

    args = parser.parse_args()
    test(args)
