#%%
import torch
from torch import nn

from keras.datasets import mnist  # Keras is a version of tensorflow, not torch. I'm only importing it for MNIST data. 
from keras.utils.np_utils import to_categorical   # And for this helpful function. 
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torchinfo import summary as torch_summary

# For some reason, plotting with matplotlib can cause errors without this.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")

# Monitor GPU memory.
from math import log10
def get_free_mem(string = ""):
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    try: print("{}: \t{}, {}.".format(string, round(log10(f),2), f))
    except: print("{}: \t{}, {}.".format(string, "()", f))

# Remove from GPU memory.
def delete_these(verbose = False, *args):
    if(verbose): get_free_mem("Before deleting")
    del args
    torch.cuda.empty_cache()
    if(verbose): get_free_mem("After deleting")
    
get_free_mem("TO BEGIN")

#%%

# First, let's collect MNIST data.
# x is a 28-by-28 image of a handwritten digit from 0 to 9. y is the correct label.
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Next, let's 'clean' the data. 
# Training is easier when inputs are floats in [-1, 1], not integers in [0, 255].
train_x = train_x/255
test_x  = test_x /255

# Convolution requires a 'channel.'
# Grayscale images only have one channel.
# Red, green, and blue would be three channels.
train_x = np.expand_dims(train_x, 1)
test_x  = np.expand_dims(test_x,  1)

# Making numpy arrays into torch tensors.
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x  = torch.from_numpy(test_x)
test_y  = torch.from_numpy(test_y)

# Let's take a look!
print("Training data: \n\tX: {}\n\tY: {}".format(train_x.shape, train_y.shape))
print("Testing data:  \n\tX: {}\n\tY: {}".format(test_x.shape,  test_y.shape))

def show_digit(train = True, num = 0):
    digit = train_x[num] if train else test_x[num]
    digit = np.squeeze(digit,0)
    label = train_y[num] if train else test_y[num]
    label = label.cpu()
    label = np.argmax(label)
    plt.imshow(digit, cmap='gray')
    plt.title("Handwritten {}".format(label))
    plt.show()
    plt.close()
    
print("\n\nOne handwritten digit:")
show_digit()

get_free_mem("AFTER GETTING DATA")

#%%

# Now, let's make a predictive model.

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(                     # Convolution is for data like images, where relative positions matter.
                in_channels = 1, 
                out_channels = 64, 
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.MaxPool2d(kernel_size = 2), # Maxpooling reduces dimensionality by only keeping the most helpful data.
            nn.PReLU(),                # LeakyReLU is an 'activation,' an alternative to being just linear.
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.MaxPool2d(kernel_size = 2),
            nn.PReLU())
        
        # This is a 'kludge' to see the shape of the data after convolution.
        example = torch.zeros([1,1,28,28])
        example = self.conv(example).flatten(1)
        shape_after_conv = example.shape[-1]

        self.lin = nn.Sequential(
            nn.Linear(                         # A linear layer is the most common kind of neural network. 
                in_features = shape_after_conv,
                out_features = 256),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = 256,
                out_features = 10),
            nn.LogSoftmax(1))
                
    # This is the function called by 'model()'
    def forward(self, images):
        images = images.to(device).float()
        images = (images*2) - 1
        x = self.conv(images).flatten(1)
        guesses = self.lin(x)
        delete_these(False, images, x)
        return(guesses.cpu())
    
model = Model()
optimizer = Adam(model.parameters())

get_free_mem("AFTER MAKING MODEL")

model = model.to(device)

get_free_mem("AFTER MODEL TO DEVICE")

print(model)
print()
print(torch_summary(model, (1,1,28,28)))

get_free_mem("AFTER MODEL SUMMARY")



#%%


def epoch(batch_size):
    index = [i for i in range(len(train_x))]
    shuffle(index)
    index = index[:batch_size]
    x = train_x[index]
    y = train_y[index]
    get_free_mem("AFTER DATA")
    
    model.train()
    optimizer.zero_grad()
    get_free_mem("BEFORE RUN")
    guesses = model(x)   # Losing some GPU data
    get_free_mem("BEFORE LOSS")
    loss = F.nll_loss(guesses, y)
    get_free_mem("BEFORE BACK")
    loss.backward()      # Gaining lots of GPU data
    get_free_mem("AFTER BACK")
    optimizer.step()
    
    absolute_guesses = torch.argmax(guesses, 1)
    accuracy = sum([absolute_guesses[i] == y[i] for i in range(batch_size)]) / batch_size
    l = loss.item(); a = accuracy.item()
    get_free_mem("BEFORE DELETE")
    delete_these(False, loss, accuracy, guesses, absolute_guesses)   # Losing lots of GPU data
    get_free_mem("AFTER DELETE")
    return(l, a)
    
def test():
    get_free_mem("BEFORE TEST")
    with torch.no_grad():
        model.eval()
        guesses = model(test_x)
        loss = F.nll_loss(guesses, test_y)
        
        absolute_guesses = torch.argmax(guesses, 1)
        accuracy = sum([absolute_guesses[i] == test_y[i] for i in range(len(test_x))]) / len(test_x)
    l = loss.item(); a = accuracy.item()
    delete_these(False, loss, accuracy, guesses, absolute_guesses)
    get_free_mem("AFTER TEST")
    return(l, a)

def plot_loss_and_acc(train_loss, train_acc, test_loss, test_acc, epochs_per_test):
    train_xs = [i for i in range(1, 1+len(train_loss))]
    test_xs  = [train_xs[i*epochs_per_test -1] for i in range(1, 1+len(test_loss))]
    fig, ax1 = plt.subplots() 
    ax2 = ax1.twinx() 
    ax1.plot(train_xs, train_loss, color = "cyan",  label = "Train Loss")
    ax1.plot(test_xs,  test_loss,  color = "pink",   label = "Test Loss")
    ax2.plot(train_xs, train_acc,  color = "blue",  label = "Train Acc")
    ax2.plot(test_xs,  test_acc,   color = "red",  label = "Test Acc")
    ax1.legend(loc = 'lower left')
    ax2.legend(loc = 'upper left')
    ax2.set_ylim([0, 1])
    plt.title("Accuracy and Loss")
    plt.xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    plt.show()
    plt.close()
    
def train(epochs = 250, batch_size = 256, epochs_per_test = 10):
    train_loss = []; train_acc = []
    test_loss =  []; test_acc =  []
    for e in range(epochs):
        torch.cuda.synchronize()
        get_free_mem("\nBEFORE EPOCH")
        loss, accuracy = epoch(batch_size)
        get_free_mem("AFTER EPOCH")
        train_loss.append(loss)
        train_acc.append(accuracy)
        if((e%epochs_per_test == 0 and e!=0) or e == epochs-1):
            loss, accuracy = test()
            test_loss.append(loss)
            test_acc.append(accuracy)
            plot_loss_and_acc(train_loss, train_acc, test_loss, test_acc, epochs_per_test)
            
train()
# %%
