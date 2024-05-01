import numpy as np
import torch
from core.dataset import ThumanDataset
from core.models import ControlLGM
from core.unet import UNet, ControlUnet, ControlNetwork
from torch.utils.data import DataLoader
from core.options import config_defaults
from accelerate import Accelerator, DistributedDataParallelKwargs
import argparse
import os
import kiui


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    opt = config_defaults['big']
    opt.num_frames = args.num_frames

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.num_acc,
        # kwargs_handlers=[ddp_kwargs],
    )

    control_lgm = ControlLGM(opt, num_frames=args.num_frames)
    assert os.path.exists(args.ckpt_path)
    control_lgm.load_state_dict(torch.load(args.ckpt_path)['model'], strict=False)

    train_dataset = ThumanDataset(
        opt=opt,
        num_frames=args.num_frames,
        dataset_dir=args.dataset_dir,
        device='cpu',
        use_half=False,
        training=True,
    )
    test_dataset = ThumanDataset(
        opt=opt,
        num_frames=args.num_frames,
        dataset_dir=args.dataset_dir,
        device='cpu',
        use_half=False,
        training=False,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(control_lgm.controlunet.parameters(), lr=args.lr, weight_decay=0.05,
                                  betas=(0.9, 0.95))
    total_steps = args.training_steps * len(dataloader)
    pct_start = 3000 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                                                    pct_start=pct_start)
    control_lgm, optimizer, dataloader, test_dataloader, scheduler = accelerator.prepare(
        control_lgm, optimizer, dataloader, test_dataloader, scheduler
    )

    for epoch in range(args.training_steps):
        control_lgm.train()
        total_loss = 0
        total_psnr = 0
        for i, data in enumerate(dataloader):
            with accelerator.accumulate(control_lgm):
                optimizer.zero_grad()
                step_ratio = (epoch + i / len(dataloader)) / args.training_steps
                out = control_lgm(data, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(control_lgm.parameters(), args.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

            if accelerator.is_main_process:
                if i % 100 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()
                    print(
                        f"[INFO] {i}/{len(dataloader)} mem: {(mem_total - mem_free) / 1024 ** 3:.2f}/{mem_total / 1024 ** 3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}")

                if i % 500 == 0:
                    gt_images = data['images_output'].detach().cpu().numpy()
                    gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                    kiui.write_image(f'{args.output_dir}/train_gt_images_{epoch}_{i}.jpg', gt_images)
                    pred_images = out['images_pred'].detach().cpu().numpy()  # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1,
                                                                               pred_images.shape[1] * pred_images.shape[
                                                                                   3], 3)
                    kiui.write_image(f'{args.output_dir}/train_pred_images_{epoch}_{i}.jpg', pred_images)
        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(dataloader)
            total_psnr /= len(dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")

        accelerator.wait_for_everyone()
        accelerator.save_model(control_lgm, args.output_dir)

        # evel
        with torch.no_grad():
            control_lgm.eval()
            total_psnr = 0
            for i, data in enumerate(test_dataloader):

                out = control_lgm(data)

                psnr = out['psnr']
                total_psnr += psnr.detach()

                # save some images
                if accelerator.is_main_process:
                    gt_images = data['images_output'].detach().cpu().numpy()  # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3],
                                                                           3)  # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{args.output_dir}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy()  # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1,
                                                                               pred_images.shape[1] * pred_images.shape[
                                                                                   3], 3)
                    kiui.write_image(f'{args.output_dir}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

                    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    # kiui.write_image(f'{args.output_dir}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_precision', type=str, default='fp16')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_frames', type=int, default=7)
    parser.add_argument('--output_dir', type=str, default='training_outputs')
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/luoziqian//Works/Thuman_dataset/Thuman_dataset_for_sv3d_1')
    parser.add_argument('--save_name', type=str, default='control_lgm.pth')
    parser.add_argument('--ckpt_path', type=str, default='./control_lgm.pth')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--training_steps', type=int, default=1000)
    parser.add_argument('--num_acc', type=int, default=100)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    args = parser.parse_args()
    main(args)
