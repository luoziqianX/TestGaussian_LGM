import tyro
import time
import os
import random
import argparse
import torch
import numpy as np
import random
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

from core.options import AllConfigs, Options, config_defaults
from core.dataset import ThumanDataset
from core.models import ControlLGM
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import kiui
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from kiui.cam import orbit_camera


def main(args):
    opt = config_defaults['big']
    opt.resume = args.lgm_checkpoints_path
    opt.num_frames = args.num_frames

    train_dataset = ThumanDataset(
        opt=opt,
        num_frames=args.num_frames,
        dataset_dir=args.dataset_dir,
        device=args.device,
        use_half=args.half,
        training=True,
    )
    test_dataset = ThumanDataset(
        opt=opt,
        num_frames=args.num_frames,
        dataset_dir=args.dataset_dir,
        device=args.device,
        use_half=args.falf,
        training=False,
    )

    model = ControlLGM(opt, num_frames=args.num_frames)
    if args.half:
        model = model.half()
        model = model.to(args.device)

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.controlunet.parameters(), lr=args.learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_frames', type=int, default=21)
    parser.add_argument('--output_dir', type=str, default='training_outputs')
    parser.add_argument('--dataset_dir', type=str, default='datasets')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_frames', type=int, default=21)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()
    main(args)
