#%%
from math import pi

import torch
from torch import nn 
from torchvision.transforms.functional import resize
from torch.optim import Adam
import kornia
from torchinfo import summary as torch_summary

utils = True
try: from utils import device, init_weights, k, delete_these
except: 
    utils = False; k = 20
    def delete_these(verbose, *args): pass
    
    
    
def shared_cnn(in_channels = 3):
    cnn = nn.Sequential(
        nn.Conv2d(
            in_channels = in_channels,
            out_channels = 64,
            kernel_size = 3,
            padding = 1,
            padding_mode = "reflect"),
        nn.Dropout(.2),
        nn.PReLU(),
        nn.MaxPool2d(kernel_size = 2),
        nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            padding = 1,
            padding_mode = "reflect"),
        nn.Dropout(.2),
        nn.PReLU(),
        nn.MaxPool2d(kernel_size = 2))
    return(cnn)
    
    
    
class G(nn.Module):
    
    def __init__(self, k):
        super().__init__()
        
        self.subletter = "a" 
        self.num = 1
        self.name = "g{}{}_{}".format("a", 1, str(k+1).zfill(3))
        self.k = k
                
        self.cnn_rgb    = shared_cnn()
        self.cnn_hsv    = shared_cnn()
        self.cnn_hls    = shared_cnn()
        self.cnn_xyz    = shared_cnn()
        self.cnn_ycbcr  = shared_cnn()
        self.cnn_gray   = shared_cnn(1)
        
        example = torch.zeros((1, 3, 84, 84))
        example = self.cnn_rgb(example).flatten(1)
        quantity = example.shape[-1]
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = quantity * 6,
                out_features = 100),
            nn.LogSoftmax(1))
        
        if(utils): 
            self.cnn_rgb.apply(   init_weights)
            self.cnn_hsv.apply(   init_weights)
            self.cnn_hls.apply(   init_weights)
            self.cnn_xyz.apply(   init_weights)
            self.cnn_ycbcr.apply( init_weights)
            self.cnn_gray.apply(  init_weights)
            self.lin.apply(       init_weights)
        self.opt = Adam(self.parameters())
        
    def forward(self, x):
        if(utils): x = x.to(device)
        
        x = x.permute(0, -1, 1, 2)
        rgb    = x*2 - 1
        
        hsv    = kornia.color.rgb_to_hsv(x)
        hsv[:,0,:,:]  = hsv[:,0,:,:]   * (1/pi) - 1
        hsv[:,1:,:,:] = hsv[:,1:2,:,:] * 2      - 1
        
        hls    = kornia.color.rgb_to_hls(x)
        hls[:,0,:,:]  = hls[:,0,:,:]   * (1/pi) - 1
        hls[:,1:,:,:] = hls[:,1:2,:,:] * 2      - 1
        
        xyz    = kornia.color.rgb_to_xyz(x)*2   - 1
        ycbcr  = kornia.color.rgb_to_ycbcr(x)*2 - 1
        gray   = kornia.color.rgb_to_grayscale(x)*2 - 1

        rgb    = self.cnn_rgb(rgb      ).flatten(1)
        hsv    = self.cnn_hsv(hsv      ).flatten(1)
        hls    = self.cnn_hls(hls      ).flatten(1)
        xyz    = self.cnn_xyz(xyz      ).flatten(1)
        ycbcr  = self.cnn_ycbcr(ycbcr  ).flatten(1)
        gray   = self.cnn_gray(gray    ).flatten(1)
        
        x = torch.cat([rgb, hsv, hls, xyz, ycbcr, gray], 1)
        
        y = self.lin(x)
        delete_these(False, x, rgb, hsv, hls, xyz, ycbcr, gray)
        return(y.cpu())
    
g_dict = {}
k_list = []

for k_ in range(k):
    k_list.append(G(k_))
g_dict["g"] = k_list
    
if __name__ == "__main__":
    for k, v in g_dict.items():
        print()
        print(k)
        print(v[0])
        print()
        print(torch_summary(v[0], (10, 84, 84, 3)))
# %%
