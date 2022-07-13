#%%
from torch import nn 
from torchvision.transforms.functional import resize
from torch.optim import Adam
from torchinfo import summary as torch_summary

utils = True
try: from utils import device, init_weights, k, delete_these
except: 
    utils = False; k = 20
    def delete_these(verbose, *args): pass
    
image_size_dict = {"a" : 21,
                   "b" : 42,
                   "c" : 84}

out_features_dict = {1 : 128,
                     2 : 256,
                     3 : 512}
    
class C(nn.Module):
    
    def __init__(self, k, subletter, num):
        super().__init__()
        
        self.subletter = subletter 
        self.num = num
        self.name = "c{}{}_{}".format(subletter, num, str(k+1).zfill(3))
        self.k = k
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = image_size_dict[subletter]*image_size_dict[subletter]*3,
                out_features = out_features_dict[num]),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = out_features_dict[num],
                out_features = out_features_dict[num]),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = out_features_dict[num],
                out_features = 100),
            nn.LogSoftmax(1))
        
        if(utils): 
            self.lin.apply(init_weights)
        self.opt = Adam(self.parameters())
        
    def forward(self, x):
        if(utils): x = x.to(device)
        x = (x*2) - 1
        x = x.permute(0, -1, 1, 2)
        x = resize(x, (image_size_dict[self.subletter], image_size_dict[self.subletter]))
        x = x.flatten(1)
        y = self.lin(x)
        delete_these(False, x)
        return(y.cpu())
    
c_dict = {}

for subletter in ["a", "b", "c"]:
    for num in [1,2,3]:
        k_list = []
        for k_ in range(k):
            k_list.append(C(k_, subletter = subletter, num = num))
        c_dict["c{}{}".format(subletter, num)] = k_list
    
if __name__ == "__main__":
    for k, v in c_dict.items():
        print()
        print(k)
        print(v[0])
        print()
        print(torch_summary(v[0], (10, 84, 84, 3)))
# %%
