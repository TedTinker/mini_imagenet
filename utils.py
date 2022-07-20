#%%

k      = 25
epochs = 500

mini_imagenet = r"/home/ted/Desktop/mini_imagenet"
code          = r"/home/ted/Desktop/mini_imagenet_models"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")

def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass
    
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
    
    
    
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

os.chdir(code)
if not os.path.exists('plots'): os.makedirs('plots')
models = os.listdir("models")
models = [m[6:-3] for m in models if m != "__pycache__"]
models.sort()

os.chdir(mini_imagenet)
if not os.path.exists('trained'): os.makedirs('trained')
trained_models = os.listdir("trained")
trained_models = [m[:-3] for m in trained_models if m != "__pycache__"]
trained_models.sort()

os.chdir(code)
for m in models:
    if not os.path.exists('plots/{}'.format(m)): os.makedirs('plots/{}'.format(m))

for m in models:
    for file in os.listdir("plots/{}".format(m)):
        if(not file[:-4] in trained_models):
            os.remove("plots/{}/{}".format(m, file))

import matplotlib.pyplot as plt

def plot_image(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.show()
    plt.close()

def plot_loss_acc(model, e, train_losses, test_losses, train_acc, test_acc):
    fig, ax1 = plt.subplots()
    fig.set_figsize=(7,7)
    ax2 = ax1.twinx()
    ax1.plot(train_losses, color = "b", label = 'Train loss')
    ax1.plot(test_losses,  color = "r", label = 'Test loss')
    ax1.set_ylabel("Loss")
    ax2.plot(train_acc, color = "c", label = 'Train acc')
    ax2.plot(test_acc,  color = "m", label = 'Test acc')
    ax2.set_ylabel("Accuracy")
    plt.title("{}: {} epochs loss and accuracy".format(model.name, e))
    ax2.set_ylim((0,100))
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower left')
    
    plt.savefig("plots/{}/{}".format(model.name[0], model.name))
    #plt.show()
    plt.close()
    
def get_betweens(k_test):
    between_letters = []
    between_subletters = []
    ongoing_letters = ""
    for i, name in enumerate(k_test):
        if(ongoing_letters != "" and name[0] != ongoing_letters[0]):
            between_letters.append(i+.5)
            ongoing_letters = name[:-1]
        if(name[:-1] != ongoing_letters):
            between_subletters.append(i+.5)
            ongoing_letters = name[:-1]
    for x in between_letters:
        plt.axvline(x=x, color = "black", linewidth = 2, linestyle = "-")
    for x in between_subletters:
        plt.axvline(x=x, color = "black", linewidth = 1, linestyle = "--")
    
def plot_boxes_loss(train_losses, test_losses):
    plt.figure(figsize=(20,7))
    train_c = (0,0,1,.1)
    k_train = list(train_losses.keys())
    v_train = list(train_losses.values())
    k_train, v_train = zip(*sorted(zip(k_train, v_train)))
    train = plt.boxplot(v_train, vert = True, widths = .75,
        patch_artist=True,
        boxprops=dict(facecolor=train_c, color=train_c),
        capprops=dict(color=train_c),
        whiskerprops=dict(color=train_c),
        flierprops=dict(color=train_c, markeredgecolor=train_c),
        medianprops=dict(color=train_c))
    
    test_c = (1,0,0,.5)
    k_test = list(test_losses.keys())
    v_test = list(test_losses.values())
    k_test, v_test = zip(*sorted(zip(k_test, v_test)))
    test = plt.boxplot(v_test, vert = True, widths = .25,
        patch_artist=True,
        boxprops=dict(facecolor=test_c, color=test_c),
        capprops=dict(color=test_c),
        whiskerprops=dict(color=test_c),
        flierprops=dict(color=test_c, markeredgecolor=test_c),
        medianprops=dict(color=test_c))
    
    plt.xticks(ticks = [i for i in range(1, len(k_test)+1)], labels = k_test)
    plt.title("Model losses")
    
    get_betweens(k_test)
    plt.legend([train["boxes"][0], test["boxes"][0]], ['Train losses', 'Test losses'], loc='upper left')

    plt.savefig("plots/boxes_loss")
    plt.show()
    plt.close()
    
def plot_boxes_acc(train_acc, test_acc):
    plt.figure(figsize=(20,7))
    train_c = (0,0,1,.1)
    k_train = list(train_acc.keys())
    v_train = list(train_acc.values())
    k_train, v_train = zip(*sorted(zip(k_train, v_train)))
    train = plt.boxplot(v_train, vert = True, widths = .75,
        patch_artist=True,
        boxprops=dict(facecolor=train_c, color=train_c),
        capprops=dict(color=train_c),
        whiskerprops=dict(color=train_c),
        flierprops=dict(color=train_c, markeredgecolor=train_c),
        medianprops=dict(color=train_c))
    
    test_c = (1,0,0,.5)
    k_test = list(test_acc.keys())
    v_test = list(test_acc.values())
    k_test, v_test = zip(*sorted(zip(k_test, v_test)))
    test = plt.boxplot(v_test, vert = True, widths = .25,
        patch_artist=True,
        boxprops=dict(facecolor=test_c, color=test_c),
        capprops=dict(color=test_c),
        whiskerprops=dict(color=test_c),
        flierprops=dict(color=test_c, markeredgecolor=test_c),
        medianprops=dict(color=test_c))
    
    plt.xticks(ticks = [i for i in range(1, len(k_test)+1)], labels = k_test)
    plt.title("Model accuracies")
    
    get_betweens(k_test)
    plt.legend([train["boxes"][0], test["boxes"][0]], ['Train accuracies', 'Test accuracies'], loc='upper left')
    
    minimums = [min(l) for l in list(train_acc.values()) + list(test_acc.values())]
    minimum = min(minimums)
    #plt.ylim((minimum-3,100))

    plt.savefig("plots/boxes_acc")
    plt.show()
    plt.close()
    
def save_model(model):
    model = model.to("cpu")
    os.chdir(mini_imagenet)
    torch.save(model.cpu().state_dict(), "trained/{}.pt".format(model.name))
    delete_these(False, model)
    os.chdir(code)
    
def load_model(model):
    os.chdir(mini_imagenet)
    model.load_state_dict(torch.load("trained/{}.pt".format(model.name)))
    os.chdir(code)
    return(model)
# %%
