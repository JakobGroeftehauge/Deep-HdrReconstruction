#%%writefile test.py
import multiprocessing
import time
from progressbar import progressbar
import numpy as np
import ctypes
import multiprocessing_logging
import argparse
import torch
from image_processing import preprocess_image, postprocess_image, postprocess_mask
from common import create_logger, setup_decoder, setup_encoder, PipelineParams, setup_mask_encoder
from lib.img_io import writeLDR  

def get_index_indicator(idx_indicator):
    """
    get_index_indicator - finds unoccopied index of list of shared memory 

    :param idx_indicator: list of bool indicating occupied spaces 
    :return: free index
    """ 
    while True:
      tmp_ind = np.frombuffer(idx_indicator, dtype=ctypes.c_long)
      for idx, i in enumerate(tmp_ind): 
        if i == 0: 
          idx_indicator[idx] = 1
          return idx
    return 

def launch_inference(params, ind, frames_buffer, mask_buffer, decQ, encQ):  
    dec = multiprocessing.Process(target=preprocess, args=(params, ind, frames_buffer, mask_buffer, decQ, ))
    dec.start()
    net = multiprocessing.Process(target=DeepHDR, args=(params, frames_buffer, mask_buffer, encQ, decQ, ))
    net.start()
    enc = multiprocessing.Process(target=postprocess, args=(params, frames_buffer, mask_buffer, ind, encQ, ))
    enc.start()
    
    dec.join()
    net.join()
    enc.join()
    return 


def preprocess(params, ind, frames_buffer, mask_buffer, decQ): 
    logger = create_logger(params.logger_name)
    logger.info('Pre-Process/decode process started')
    decoder = setup_decoder(params.input_pth, params.width, params.height)

    for i in progressbar(range(params.n_frames)):
      logger.debug("Frame decoding initiated")
      in_bytes = decoder.stdout.read(params.size)
      logger.debug('Frame decoded')
      
      img = np.frombuffer(in_bytes, np.uint8).reshape([params.height, params.width, 3])
      img.transpose(2, 0, 1)
      img, mask = preprocess_image(img, params.sat_threshold)
      #print("mask stats - min: ", mask.min(), "  max: ", mask.max(), " mean: ", np.mean(mask))
      idx = get_index_indicator(ind)
      np.copyto(frames_buffer[idx], img)
      np.copyto(mask_buffer[idx], mask)


      decQ.put(idx)
      logger.debug("Index added to decodeQueue")
    
    decQ.put(None) # Indicate that there is no more data to process. 
    decoder.wait()
    return 

def postprocess(params, frames_buffer, mask_buffer, ind, encQ):
    logger = create_logger(params.logger_name)
    logger.info('Post-Process/encode process started')
    encoder = setup_encoder(params.output_pth, params.width, params.height, params.fps, params.max_luminance)
    if params.save_mask:
      mask_out_pth = params.output_pth.split(".")[0] + "_mask.mp4"
      #print("mask_out_pth: ", mask_out_pth)
      mask_encoder = setup_mask_encoder(mask_out_pth, params.width, params.height, params.fps)



    while True:
      idx = encQ.get() 
      logger.debug("Index retrieved from encodeQueue")
      
      if idx is None: 
        break
      
      pred = np.frombuffer(frames_buffer[idx], dtype=np.float32).reshape([1] + params.arr_shape)
      pred = pred.transpose(0, 2, 3, 1)

      if params.save_mask:
        mask = np.frombuffer(mask_buffer[idx], dtype=np.float32).reshape([1] + params.arr_shape)
        #writeLDR(mask.transpose(0, 2, 3, 1)[0,:,:,:], "/content/test2.png")
        mask = mask.transpose(0, 2, 3, 1)[0, :, :, : ]

        mask = postprocess_mask(mask)
        #mask = mask.transpose(0, 2, 3, 1)
        mask_encoder.stdin.write(mask.astype(np.uint8).tobytes())
        #print("write to mask encoder") 


      logger.debug('Write to encoder initiated')
      encoder.stdin.write(pred.astype(np.uint16).tobytes())
      logger.debug('Write to encoder finshed')
      ind[idx] = 0

    logger.debug("Initiated closing of encoder")
    encoder.stdin.close()
    encoder.wait()
    if params.save_mask: 
      mask_encoder.stdin.close()
      mask_encoder.wait()
    logger.debug("Encoder closed")
    return 


def DeepHDR(params, frames_buffer, mask_buffer, encQ, decQ): 
    from DeepHDRmodel import DeepHDRModel

    logger = create_logger(params.logger_name)
    logger.info('DeepHDR process started')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not params.disable_model:
      model = DeepHDRModel(params.model_pth, device, half=params.half)

    while True: 
      logger.debug("decQ size: {} encQ size: {}".format(decQ.qsize(), encQ.qsize()))
      idx = decQ.get()

      if idx is None: 
        break
      
      frame = np.frombuffer(frames_buffer[idx], dtype=np.float32).reshape([1] + params.arr_shape)
      mask  = np.frombuffer(mask_buffer[idx], dtype=np.float32).reshape([1] + params.arr_shape)
      #writeLDR(mask.transpose(0, 2, 3, 1)[0,:,:,:], "/content/test.png")
      logger.debug('Inference started')
      if params.disable_model:
        import time
        output = frame.reshape(params.arr_shape)
        time.sleep(0.5)
      else:
        output = model.run_model(frame, mask)

      logger.debug('Inference Stopped')
      np.copyto(frames_buffer[idx], output)

      encQ.put(idx)

    encQ.put(None) # Indicate that there is no more data to process. 
    return  

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-input-file', dest="input", type=str, help='Path to HDR image to transform')
    parser.add_argument('-output-filename', dest="output", default=None, type=str, help='Path to store frames')
    parser.add_argument('-model', type=str, help="path to inference ONNX-model")
    parser.add_argument('-logging-file', dest="log", default="debug_process.log", type=str, help="name of logging file" )
    parser.add_argument('-fps', default=None, type=int)
    parser.add_argument('-sat-threshold',dest='sat_threshold', default=0.95, type=float)
    parser.add_argument('-width', default=None, type=int)
    parser.add_argument('-height',default=None, type=int)
    parser.add_argument('--disable-model', dest='disable_model', action='store_true', help="disable onnx for debugging on computer wit limited resources")
    parser.add_argument('--half', dest='half', action='store_true', help="Indicate that model is in half precision")
    parser.add_argument('--save-mask', dest='save_mask', action='store_true', help="Encode video of mask for each frame.")

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == '__main__':

  #multiprocessing.set_start_method('spawn')
  opt = parse_opt(True)
  params = PipelineParams(opt.model, opt.input, opt.output, fps=opt.fps, sat_threshold=opt.sat_threshold, width=opt.width, height=opt.height, half=opt.half, save_mask=opt.save_mask, logger_name=opt.log, disable_model=opt.disable_model)

  print("wh: ", params.width, "  ", params.height, "  fps: ", params.fps)
  multiprocessing_logging.install_mp_handler()
  logger = create_logger(params.logger_name)
  logger.debug("Start of program2")
  
  decodeQueue = multiprocessing.Queue(maxsize=3)
  encodeQueue = multiprocessing.Queue(maxsize=3)

  indicator= multiprocessing.Array(ctypes.c_long,[0] * params.N_numbers, lock=False)

  image_arrays = []
  image_arrays_np = []
  mask_arrays = []
  mask_arrays_np = []

  for i in range(params.N_numbers): 
      arr_image = multiprocessing.RawArray(ctypes.c_float, int(params.size))
      arr_mask = multiprocessing.RawArray(ctypes.c_float, int(params.size))

      image_arrays.append(arr_image)
      image_arrays_np.append(np.frombuffer(arr_image, dtype=np.float32).reshape(params.arr_shape))
      mask_arrays.append(arr_mask)
      mask_arrays_np.append(np.frombuffer(arr_mask, dtype=np.float32).reshape(params.arr_shape))

  t1 = time.time()
  launch_inference(params, indicator, image_arrays_np, mask_arrays_np, decodeQueue, encodeQueue)
  t2 = time.time()
  t = t2-t1

  print("----- Inference Completed -----")
  print("Avg. Per Image: ", str(t/params.n_frames))
  print("Complete video: ", str(t))
  print("-------------------------------")
