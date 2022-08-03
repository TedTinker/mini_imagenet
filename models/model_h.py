#%%
from math import pi

import torch
from torch import nn 
import torchgan.layers as gnn
from torchvision.transforms.functional import resize
from torch.optim import Adam
import kornia
from torchinfo import summary as torch_summary

utils = True
try: from utils import device, init_weights, k, delete_these
except: 
    utils = False; k = 20
    def delete_these(verbose, *args): pass
    
    

    
    
class Color(nn.Module):
    def __init__(self, channels = 3):
        super().__init__()
        
        self.cnn = nn.Sequential(
        nn.Conv2d(
            in_channels = channels,
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
        
        example = torch.zeros((1, channels, 84, 84))
        example = self.cnn(example).flatten(1)
        quantity = example.shape[-1]
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = quantity,
                out_features = 256),
            nn.PReLU())
        
        if(utils): 
            self.cnn.apply(init_weights)
            self.lin.apply(init_weights)
        
    def forward(self, x):
        x = self.cnn(x).flatten(1)
        x = self.lin(x)
        return(x)
    
    
class H(nn.Module):
    
    def __init__(self, k):
        super().__init__()
        
        self.subletter = "a" 
        self.num = 1
        self.name = "h{}{}_{}".format("a", 1, str(k+1).zfill(3))
        self.k = k
                
        self.cnn_rgb    = Color()
        self.cnn_hsv    = Color()
        self.cnn_hls    = Color()
        self.cnn_xyz    = Color()
        self.cnn_ycbcr  = Color()
        self.cnn_gray   = Color(1)
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = 256 * 6,
                out_features = 100),
            nn.LogSoftmax(1))
        
        if(utils): 
            self.lin.apply(init_weights)
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
        
        if(self.training):
            rgb   += torch.normal(0, 0.2, size=rgb.size()).to(device) 
            hsv   += torch.normal(0, 0.2, size=hsv.size()).to(device) 
            hls   += torch.normal(0, 0.2, size=hls.size()).to(device) 
            xyz   += torch.normal(0, 0.2, size=xyz.size()).to(device) 
            ycbcr += torch.normal(0, 0.2, size=ycbcr.size()).to(device) 
            gray  += torch.normal(0, 0.2, size=gray.size()).to(device) 

        rgb    = self.cnn_rgb(rgb    )
        hsv    = self.cnn_hsv(hsv    )
        hls    = self.cnn_hls(hls    )
        xyz    = self.cnn_xyz(xyz    )
        ycbcr  = self.cnn_ycbcr(ycbcr)
        gray   = self.cnn_gray(gray  )
        
        x = torch.cat([rgb, hsv, hls, xyz, ycbcr, gray], 1)
        
        y = self.lin(x)
        delete_these(False, x, rgb, hsv, hls, xyz, ycbcr, gray)
        return(y.cpu())
    
h_dict = {}
k_list = []

for k_ in range(k):
    k_list.append(H(k_))
h_dict["h"] = k_list
    
if __name__ == "__main__":
    for k, v in h_dict.items():
        print()
        print(k)
        print(v[0])
        print()
        print(torch_summary(v[0], (10, 84, 84, 3)))
# %%
