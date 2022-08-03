#%%
import torch

import pandas as pd
import pickle 
import os
from random import shuffle, seed
import itertools
from_iterable = itertools.chain.from_iterable

from utils import k, plot_image, mini_imagenet
os.chdir(mini_imagenet)

def load():
    with open("train.pkl",'rb') as f: train = pickle.load(f)
    with open("test.pkl",'rb')  as f: test  = pickle.load(f)
    with open("val.pkl",'rb')   as f: val   = pickle.load(f)
    return train, test, val

train, test, val = load()
train_x, train_y = torch.from_numpy(train["image_data"]), train["class_dict"]
test_x,  test_y  = torch.from_numpy(test["image_data"]),  test["class_dict"]
val_x,   val_y   = torch.from_numpy(val["image_data"]),   val["class_dict"]

classes = list(train_y.keys()) + list(test_y.keys()) + list(val_y.keys())
classes.sort()
names = pd.read_csv('mapping.txt', sep='\0',header=None)
class_to_name   = {c[:9] : c[10:].split(',')[0] for c in names[0]}
class_to_number = {c : i for i, c in enumerate(classes)}
number_to_class = {i : c for c, i in class_to_number.items()}
number_to_name  = {i : class_to_name[number_to_class[i]] for i in range(len(classes))}

train_y = torch.tensor([[class_to_number[k]]*600 for k in list(train_y.keys())])
test_y  = torch.tensor([[class_to_number[k]]*600 for k in list(test_y.keys())])
val_y   = torch.tensor([[class_to_number[k]]*600 for k in list(val_y.keys())])
train_y = train_y.reshape((train_y.shape[0] *600))
test_y  = test_y.reshape( (test_y.shape[0]  *600))
val_y   = val_y.reshape(  (val_y.shape[0]   *600))

xs = torch.cat([train_x, test_x, val_x])/255
ys = torch.cat([train_y, test_y, val_y])
data_len = len(xs)
    
    
    
indexes = [i for i in range(24*25)]

test_indexes = []
for i in range(k):
    test_k_indexes = []
    for j in range(int(data_len/k)):
        test_k_indexes.append(i+k*j)
    test_indexes.append(test_k_indexes)
train_indexes = []
for k_ in range(k):
    train_indexes.append(list(from_iterable(test_indexes[k__] for k__ in range(k) if k__ != k_)))


def get_batch(k_, batch_size = 128, test = False):
    if(test):
        indexes = test_indexes[k_]
        x = xs[indexes]
        y = ys[indexes]
        return(x, y)
    indexes = train_indexes[k_]
    shuffle(indexes)
    batch = indexes[:batch_size]
    x = xs[batch]
    y = ys[batch]
    return(x, y)

if __name__ == "__main__":
    x, y = get_batch(0)
    print(x.shape, y.shape)
    x, y = get_batch(0, test = True)
    print(x.shape, y.shape)
    for i in range(10):
        plot_image(x[i], number_to_name[y[i].item()])
    for i in range(100):
        print(i, " : ", len(y[y == i]))
# %%
