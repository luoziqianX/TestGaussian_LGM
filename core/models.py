import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kiui.lpips import LPIPS

from core.gs import GaussianRenderer
from core.options import Options
from core.unet import UNet, ControlUnet, ControlNetwork


class LGM(nn.Module):
    def __init__(
            self,
            opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # unet
        self.unet = UNet(
            9, 14,
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1)  # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5  # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def prepare_default_rays(self, device, num_frames=21, elevation=0):

        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, i, radius=self.opt.cam_radius)
            for i in np.linspace(0, 360, num_frames + 1)[:-1] % 360
        ], axis=0)  # [num_frames, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size,
                                      self.opt.fovy)  # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)  # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(
            device)  # [V, 6, h, w]

        return rays_embeddings

    def forward_gaussians(self, images):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        images = images.view(B * V, C, H, W)

        x = self.unet(images)  # [B*4, 14, h, w]
        x = self.conv(x)  # [B*4, 14, h, w]

        x = x.reshape(B, V, 14, self.opt.splat_size, self.opt.splat_size)

        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)

        pos = self.pos_act(x[..., 0:3])  # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)  # [B, N, 14]

        return gaussians

    def forward(self, data, step_ratio=1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        images = data['input']  # [B, 4, 9, h, W], input features

        # use the first view to predict gaussians
        gaussians = self.forward_gaussians(images)  # [B, N, 14]

        results['gaussians'] = gaussians

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image']  # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha']  # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output']  # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output']  # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256),
                              mode='bilinear', align_corners=False),
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256),
                              mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips

        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results


class ControlLGM(nn.Module):
    def __init__(self,
                 opt: Options, num_frames: int = 21, scale_hint: float = 1.0):
        super().__init__()
        self.opt = opt
        self.scale_hint = scale_hint

        # unet
        self.controlunet = ControlUnet(
            9, 3,
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
            num_frames=num_frames,
        )
        self.controlnetwork = ControlNetwork(
            9, 14,
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
            num_frames=num_frames,
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1)  # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5  # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def prepare_default_rays(self, device, num_frames=21, elevation=0):

        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, i, radius=self.opt.cam_radius)
            for i in np.linspace(0, 360, num_frames + 1)[:-1] % 360
        ], axis=0)  # [num_frames, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size,
                                      self.opt.fovy)  # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)  # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(
            device)  # [V, 6, h, w]

        return rays_embeddings

    def control_forward(self, images, hint):
        # images: [B, 3, H, W]
        # hint: [B, 3, H, W]
        # return: [B, 14, H, W]
        hint *= self.scale_hint
        control = self.controlunet(images, hint)
        x = self.controlnetwork(images, control, only_mid_control=False)
        return x

    def forward_gaussians(self, images, hint):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        C_hint = hint.shape[2]
        images = images.view(B * V, C, H, W)
        hint = hint.view(B * V, C_hint, H, W)

        x = self.control_forward(images, hint)  # [B*4, 14, h, w]
        x = self.conv(x)  # [B*4, 14, h, w]

        x = x.reshape(B, V, 14, self.opt.splat_size, self.opt.splat_size)

        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)

        pos = self.pos_act(x[..., 0:3])  # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)  # [B, N, 14]

        return gaussians

    def forward(self, data, step_ratio=1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        images = data['input']  # [B, 4, 9, h, W], input features
        hint = data['hint']  # [B, 3, H, W], hint features

        # use the first view to predict gaussians
        gaussians = self.forward_gaussians(images, hint)  # [B, N, 14]

        results['gaussians'] = gaussians

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image']  # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha']  # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output']  # [B, V, 3, output_size, output_size], ground-truth novel views)

        loss_mse = F.mse_loss(pred_images, gt_images)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256),
                              mode='bilinear', align_corners=False),
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256),
                              mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips

        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results
