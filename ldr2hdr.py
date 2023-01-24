#%%writefile test.py
import multiprocessing
import time
from progressbar import progressbar
import numpy as np
import ctypes
import multiprocessing_logging
import argparse
import torch
from image_processing import preprocess_image, postprocess_image_torch, postprocess_mask
from common import create_logger, setup_decoder, setup_encoder, PipelineParams, setup_mask_encoder
from lib.img_io import writeLDR  



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-input-file', dest="input", type=str, help='Path to HDR image to transform')
    parser.add_argument('-output-filename', dest="output", default=None, type=str, help='Path to store frames')
    parser.add_argument('-logging-file', dest="log", default="debug_process.log", type=str, help="name of logging file" )
    parser.add_argument('-fps', default=None, type=int)
    parser.add_argument('-sat-threshold',dest='sat_threshold', default=0.95, type=float)
    parser.add_argument('-width', default=None, type=int)
    parser.add_argument('-height',default=None, type=int)
    parser.add_argument('--save-mask', dest='save_mask', action='store_true', help="Encode video of mask for each frame.")

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == '__main__':

  #multiprocessing.set_start_method('spawn')
  opt = parse_opt(True)
  params = PipelineParams(None, opt.input, opt.output, fps=opt.fps, sat_threshold=opt.sat_threshold, width=opt.width, height=opt.height, save_mask=opt.save_mask, logger_name=opt.log)

  print("wh: ", params.width, "  ", params.height, "  fps: ", params.fps)
  decoder = setup_decoder(params.input_pth, params.width, params.height)
  encoder = setup_encoder(params.output_pth, params.width, params.height, params.fps, params.max_luminance)

  t1 = time.time()
  for i in progressbar(range(params.n_frames)):
    in_bytes = decoder.stdout.read(params.size)
    img = np.frombuffer(in_bytes, np.uint8).reshape([params.height, params.width, 3])
    img.transpose(2, 0, 1)
    img, mask = preprocess_image(img, params.sat_threshold)
    mask = np.ones_like(mask)

    img = postprocess_image_torch(torch.from_numpy(img), torch.from_numpy(img), torch.from_numpy(mask), device="cpu").numpy()
    img = img.transpose(0, 2, 3, 1)
    #print("img type: ", type(img))

    #print("img shape: ", img.shape)
    encoder.stdin.write(img.astype(np.uint16).tobytes())
    t = time.time()-t1

  encoder.stdin.close()
  encoder.wait()
  print("----- Inference Completed -----")
  print("Avg. Per Image: ", str(t/params.n_frames))
  print("Complete video: ", str(t))
  print("-------------------------------")


