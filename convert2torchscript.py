import argparse
import glob
import numpy as np
import os
import torch

from torchvision import transforms
from torch.utils import data

import opt

from lib.io import load_ckpt
from network.softconvmask import SoftConvNotLearnedMaskUNet

parser = argparse.ArgumentParser(description="Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss")
parser.add_argument('--weights', '-w', type=str, required=True, help='Path to the trained CNN weights.')
parser.add_argument('--output-pth', '-o', dest='output_pth', type=str, required=True, help='Path to store model, path + model name.' )
parser.add_argument('--cpu', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    args.train = False

    # use GPU if available.
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('\tUsing device: {}.\n'.format(device))

    model = SoftConvNotLearnedMaskUNet().to(device)
    model.print_network()
    load_ckpt(args.weights, [('model', model)])
    
    model.eval()

      
    example = torch.rand(1, 3, 1080, 1920).to(device)
    print("Tracing model...\n\n")
    traced_script_module = torch.jit.trace(model, [example, example])
    traced_script_module.save(args.output_pth)
    print("Model has been succesfully traced and saved to ")
    
   
