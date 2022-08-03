#%%
import os
import torch
import matplotlib.pyplot as plt
import enlighten

from utils import code, device, trained_models, load_model, delete_these
from get_data import get_batch, number_to_name

os.chdir(code)

from models.model_a import a_dict
from models.model_b import b_dict
from models.model_c import c_dict
from models.model_d import d_dict
from models.model_e import e_dict
from models.model_f import f_dict
from models.model_g import g_dict
from models.model_h import h_dict

model_lists = []
for dict in [a_dict, b_dict, c_dict, d_dict, e_dict, f_dict, g_dict, h_dict]:
    for key, v in dict.items():
        model_lists.append(v)
        
model_names = [m[0].name[:-4] for m in model_lists]

for list in model_lists:
    for model in list:
        if(model.name in trained_models):
            model = load_model(model)
            
#%%

def demonstrate(model):
    model = model.to(device)
    model.eval()
    x, y = get_batch(k_ = model.k, test = True)
    predictions = []
    for x_, y_ in zip(torch.split(x, len(x)//25), torch.split(y, len(y)//25)):
        predicted = model(x_)
        predictions.append(predicted)
        delete_these(False, x_, y_)
    model = model.to("cpu")
    delete_these(False, model)
    predictions = torch.cat(predictions)
    first_predictions = torch.argmax(predictions, 1)
    
    rows = 25
    manager = enlighten.get_manager()
    M = manager.counter(total = (100//rows)**2, desc = "Images:", unit = "ticks", color = "red")
    for a in range(100//rows):
        for b in range(100//rows):
            plt.figure(figsize=(100,100))
            for c in range(rows):
                for d in range(rows):
                    e = rows*a + c
                    f = rows*b + d
                    index = None
                    for i in range(len(first_predictions)):
                        if(y[i].item() == e and first_predictions[i].item() == f):
                            index = i; break
                    ax = plt.subplot(rows, rows, c*rows+d+1)
                    ax.xaxis.set_ticks_position('none') 
                    ax.yaxis.set_ticks_position('none') 
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    if(e != f):
                        ax.axis('off')
                    if(index != None):
                        ax.imshow(x[i])
                        ax.set_title("{}\n({})".format(number_to_name[e], number_to_name[f]), fontsize = 15)
            save_name = "plots/{}/{}_{}.png".format(model.name[-7], a, b)
            print(save_name)
            plt.savefig(save_name)
            plt.close()
            M.update()

model_dict = {}
for model_list in model_lists:
    for model in model_list:
        model_dict[model.name] = model

demonstrate(model_dict["ea4_001"])
# %%
