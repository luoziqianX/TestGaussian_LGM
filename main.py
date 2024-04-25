import tyro
import time
import random
import argparse
import torch
from core.options import AllConfigs, Options,config_defaults
from core.models import LGM_gaussian
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

import kiui

def main(args):
    device = args.device

    opt = config_defaults['big']
    opt.resume=args.lgm_checkpoints_path
    opt.num_frames=args.num_frames
    
    # model
    lgm_model = LGM_gaussian(opt)
    lgm_model = lgm_model.half().to(device)
    lgm_model.eval()

    # resume
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(args.resume, device='cpu')
        else:
            ckpt = torch.load(args.resume, map_location='cpu')


    pass

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args=parser.parse_args()
    main(args)