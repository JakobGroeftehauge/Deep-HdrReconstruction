import numpy as np

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def saturated_channel_(im, th):
    return np.minimum(np.maximum(0.0, im - th) / (1 - th), 1)

def get_saturated_regions(im, th=0.95):
    w,h,ch = im.shape

    mask_conv = np.zeros_like(im)
    for i in range(ch):
        mask_conv[:,:,i] = saturated_channel_(im[:,:,i], th)

    return mask_conv#, mask

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD) + torch.Tensor(MEAN)
    x = x.transpose(1, 3)
    return x

def preprocess_image(image): 
    image = image/255.0

    # Calculate mask 
    mask = 1 - get_saturated_regions(image)

    #mask = torch.from_numpy(mask).permute(2,0,1)
    #mask = torch.unsqueeze(mask, 0)
    mask = np.expand_dims(mask.transpose((2, 0, 1)), axis=0)


    # Preprocess image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    image = transform(image)
    image = torch.unsqueeze(image, 0).numpy()

    return image, mask 


def postprocess_image(image, pred_image, mask, sc=20, max_luminance=1000): 
    #image = unnormalize(image).permute(0,2,3,1).numpy()[0,:,:,:]
    image = unnormalize(torch.from_nuumpy(image)).permute(0,2,3,1).numpy()[0,:,:,:]
    #mask = mask.permute(0,2,3,1).numpy()[0,:,:,:]
    mask = mask.transpose(0, 2, 3, 1)[0,:,:,:]
    #pred_img = torch.from_numpy(pred_image.cpu()).permute(0,2,3,1).numpy()[0,:,:,:]
    pred_img = pred_img.transpose(0, 2, 3, 1)[0,:,:,:]

    y_predict = np.exp(pred_img)-1
    gamma = np.power(image, 2)

    H = mask*gamma + (1-mask)*y_predict
    img = transformPQ(H * sc, MAX_LUM=max_luminance)
    img = img * 65535
    return img

def sRGB2linear(img):
    img = img / 255
    return np.where(img <= 0.04045, img / 12.92, np.power((img+0.055) / 1.055, 2.4))

def transformPQ(arr, MAX_LUM=1000.0): 
    L = MAX_LUM #max Luminance
    m = 78.8438
    n = 0.1593
    c1 = 0.8359
    c2 = 18.8516
    c3 = 18.6875
    Lp = np.power(arr/L, n)
    return np.power((c1 + c2*  Lp) / (1 + c3*Lp), m)    
  