#%%
import torch
from torch import nn 
from torchvision.transforms.functional import resize
from torch.optim import Adam
from torchinfo import summary as torch_summary

utils = True
try: from utils import device, init_weights, k, delete_these
except: 
    utils = False; k = 20
    def delete_these(verbose, *args): pass
    
image_size_dict = {"a" : 84}

out_channels_dict = {1 : 4,
                     2 : 16,
                     3 : 32,
                     4 : 64}
    
class F(nn.Module):
    
    def __init__(self, k, subletter, num):
        super().__init__()
        
        self.subletter = subletter 
        self.num = num
        self.name = "f{}{}_{}".format(subletter, num, str(k+1).zfill(3))
        self.k = k
                
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = out_channels_dict[num],
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(
                in_channels = out_channels_dict[num],
                out_channels = out_channels_dict[num],
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = 2))
        
        example = torch.zeros((1, 3, image_size_dict[subletter], image_size_dict[subletter]))
        example = self.cnn(example).flatten(1)
        quantity = example.shape[-1]
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = quantity,
                out_features = 256),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = 256,
                out_features = 100),
            nn.LogSoftmax(1))
        
        if(utils): 
            self.cnn.apply(init_weights)
            self.lin.apply(init_weights)
        self.opt = Adam(self.parameters())
        
    def forward(self, x):
        if(utils): x = x.to(device)
        x = (x*2) - 1
        x = x.permute(0, -1, 1, 2)
        x = resize(x, (image_size_dict[self.subletter], image_size_dict[self.subletter]))
        x = self.cnn(x).flatten(1)
        y = self.lin(x)
        delete_these(False, x)
        return(y.cpu())
    
f_dict = {}

for subletter in ["a"]:
    for num in [1,2,3,4]:
        k_list = []
        for k_ in range(k):
            k_list.append(F(k_, subletter = subletter, num = num))
        f_dict["f{}{}".format(subletter, num)] = k_list
    
if __name__ == "__main__":
    for k, v in f_dict.items():
        print()
        print(k)
        print(v[0])
        print()
        print(torch_summary(v[0], (10, 84, 84, 3)))
# %%
