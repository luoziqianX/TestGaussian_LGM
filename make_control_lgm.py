import torch
from safetensors import safe_open
from core.options import config_defaults
from core.models import ControlLGM, LGM
import argparse
from safetensors.torch import load_file


def main(args):
    opt = config_defaults['big']
    control_lgm = ControlLGM(opt, num_frames=21)
    lgm = LGM(opt)
    ckpt = load_file(args.lgm_path, device='cpu')
    lgm.load_state_dict(ckpt, strict=False)
    key_lib = lgm.state_dict().keys()
    un_know = 0
    model = {}
    for key in control_lgm.state_dict().keys():
        x = ''
        if key in key_lib:
            x = key
        elif 'unet' + key[len('controlunet'):] in key_lib:
            x = 'unet' + key[len('controlunet'):]
        elif 'unet' + key[len('controlnetwork'):] in key_lib:
            x = 'unet' + key[len('controlnetwork'):]
        else:
            un_know += 1
        if x != '':
            model[key] = lgm.state_dict()[x]

    control_lgm.load_state_dict(model, strict=False)
    output_ckpt = args.control_lgm_path
    torch.save({'model': control_lgm.state_dict()}, output_ckpt)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_lgm_path', type=str, default='./control_lgm.pth')
    parser.add_argument('--lgm_path', type=str, default='./pretrained/model_fp16.safetensors')
    args = parser.parse_args()
    main(args)
