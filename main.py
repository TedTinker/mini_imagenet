#%%
%env JOBLIB_TEMP_FOLDER=/tmp

import os
import enlighten
import time
from math import log10
import itertools
from_iterable = itertools.chain.from_iterable

from utils import code, device, trained_models, load_model, plot_boxes_acc, k, epochs, get_free_mem, delete_these
from train_test import train_test, train_test_short

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
        
train_losses = {m : [] for m in model_names}
test_losses  = {m : [] for m in model_names}
train_acces  = {m : [] for m in model_names}
test_acces   = {m : [] for m in model_names}

#%%

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

for l, n in zip(model_lists, model_names):
    pp = get_n_params(l[0])
    print(n, ":", round(log10(pp), 2), "\t--\t", pp)



for list in model_lists:
    for model in list:
        if(model.name in trained_models):
            model = load_model(model)
            
manager = enlighten.get_manager()
M = manager.counter(total = len(model_lists), desc = "Models:", unit = "ticks", color = "red")
K = manager.counter(total = k,                desc = "K:",      unit = "ticks", color = "green")
E = manager.counter(total = epochs,           desc = "Epochs:", unit = "ticks", color = "blue")

for list, name in zip(model_lists, model_names):
    get_free_mem("{}".format(name))
    for model in list:
        model = model.to(device)
        if(model.name in trained_models):
            train_loss, test_loss, train_acc, test_acc = train_test_short(model)
        else: 
            train_loss, test_loss, train_acc, test_acc = train_test(model, E)
        model = model.to("cpu")
        delete_these(False, model)
        train_losses[name].append(train_loss)
        test_losses[name].append(test_loss)
        train_acces[name].append(train_acc)
        test_acces[name].append(test_acc)
        E.count = 0; E.start = time.time()
        K.update()
        if(K.count == k):
            K.count = 0; K.start = time.time()
            M.update()
get_free_mem("DONE")

plot_boxes_acc(train_acces, test_acces)
plot_boxes_acc(train_acces, test_acces, True)
# %%
