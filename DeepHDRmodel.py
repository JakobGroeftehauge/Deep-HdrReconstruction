import numpy as np
import torch
from image_processing import postprocess_image_torch

class DeepHDRModel:
  def __init__(self, model_pth, device, half=False):
    self.model = torch.jit.load(model_pth)
    self.device = device
    self.half = half
      
  def run_model(self, frame, mask):
    with torch.no_grad():
      image_ = torch.from_numpy(frame).to(self.device)
      mask_ = torch.from_numpy(mask).to(self.device)
      if self.half: 
          mask_ = mask_.half()
          image_ = image_.half()

      pred_image = self.model(image_, mask_)

      pred_image = postprocess_image_torch(image_, pred_image, mask_)

      return pred_image.cpu().numpy()


