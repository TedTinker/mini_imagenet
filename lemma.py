#%%
%env JOBLIB_TEMP_FOLDER=/tmp

import enlighten
from itertools import product
import time
import os
import pickle
import torch.multiprocessing as mp

from utils import code, device, get_free_mem, delete_these, k, epochs
from lemma_train import train_test, plot_boxes_acc

os.chdir(code)

#epochs = 10

try:
    with open('plots/lambda_dicts.pickle', 'rb') as f:
        old_train_acces, old_test_acces = pickle.load(f)
except: pass

model_names = ["ea4", "ga1"]
lemma_list = [0, .0001, .0005, .001, .005, .01, .1, 1]
all_cases = list(product(model_names,lemma_list))

train_losses = {m : [] for m in [(n, l) for (n,l) in all_cases]}
test_losses  = {m : [] for m in [(n, l) for (n,l) in all_cases]}
train_acces  = {m : [] for m in [(n, l) for (n,l) in all_cases]}
test_acces   = {m : [] for m in [(n, l) for (n,l) in all_cases]}

manager = enlighten.get_manager()
A = manager.counter(total = len(lemma_list)*len(model_names)*k*epochs,  desc = "Everything:", unit = "ticks", color = "white")
L = manager.counter(total = len(lemma_list),  desc = "Lambdas:", unit = "ticks", color = "yellow")
M = manager.counter(total = len(model_names), desc = "Models:",  unit = "ticks", color = "red")
K = manager.counter(total = k,                desc = "K:",       unit = "ticks", color = "green")
E = manager.counter(total = epochs,           desc = "Epochs:",  unit = "ticks", color = "blue")

def thread_function(lemma):
    
    from models.model_e import e_dict
    from models.model_g import g_dict

    model_lists = [e_dict["ea4"], g_dict["g"]]
    
    for list, name in zip(model_lists, model_names):
        if((name, lemma) in old_test_acces):
            train_acces[(name, lemma)] = old_train_acces[(name, lemma)]
            test_acces[ (name, lemma)] = old_test_acces[ (name, lemma)]
            for _ in range(len(model_names)*k*epochs):
                A.update(); L_update = True
        else:
            L_update = False
            for model in list:
                model = model.to(device)
                train_loss, test_loss, train_acc, test_acc = train_test(model, lemma, A, E, epochs)
                model = model.to("cpu")
                delete_these(False, model)
                train_losses[(name,lemma)].append(train_loss)
                test_losses[(name,lemma)].append(test_loss)
                train_acces[(name,lemma)].append(train_acc)
                test_acces[(name,lemma)].append(test_acc)
                E.count = 0; E.start = time.time()
                K.update()
                if(K.count == k):
                    K.count = 0; K.start = time.time()
                    M.update()
        if(L_update): L.update()
        if(M.count == len(model_lists)):
            M.count = 0; M.start = time.time()
            L.update()
        get_free_mem(name)
    

for l in lemma_list:
    thread_function(l)


os.chdir(code)
with open('plots/lambda_dicts.pickle', 'wb') as f:
    pickle.dump((train_acces, test_acces), f)
            
plot_boxes_acc(train_acces, test_acces)
plot_boxes_acc(train_acces, test_acces, True)
        
# %%
