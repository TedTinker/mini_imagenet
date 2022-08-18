#%%
import torch
import torch.nn.functional as F

#torch.autograd.set_detect_anomaly(True)

from utils import delete_these, save_model, epochs
from get_data import get_batch

def train_test(model, lemma, A, E, epochs = epochs, batch_size = 128, show_after = 999999):
    train_losses = []; test_losses = []
    train_acc = [];    test_acc = []
    
    for e in range(1,epochs+1):
        E.update(); A.update()
        model.train()
        x, y = get_batch(k_ = model.k, batch_size = batch_size, test = False)
        predicted = model(x)
        
        loss = F.nll_loss(predicted, y)
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + (lemma * l2_norm)
        
        model.opt.zero_grad()
        loss.backward()
        torch.cuda.empty_cache()
        model.opt.step()
        train_losses.append(loss.item())
        accurate = [True if torch.argmax(p).item() == y[i].item() else False for i, p in enumerate(predicted)]
        train_acc.append(100*sum(accurate)/len(accurate))
        delete_these(False, x, y, predicted, loss)
        
        with torch.no_grad():
            model.eval()
            x, y = get_batch(k_ = model.k, batch_size = batch_size, test = True)
            total_loss = 0
            accurates = []
            for x_, y_ in zip(torch.split(x, len(x)//25), torch.split(y, len(y)//25)):
                predicted = model(x_)
                
                loss = F.nll_loss(predicted, y_)
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + (lemma * l2_norm)
                
                total_loss += loss.item()
                accurate = [True if torch.argmax(p).item() == y_[i].item() else False for i, p in enumerate(predicted)]
                accurates += accurate
                delete_these(False, x_, y_, predicted, loss)
            test_losses.append(total_loss)
            test_acc.append(100*sum(accurates)/len(accurates))

    save_model(model)
    torch.cuda.synchronize()
    return(train_losses[-1], test_losses[-1], train_acc[-1], test_acc[-1])










letters_to_name = {
    "aa1" : "Linear_21",
    "ab1" : "Linear_42",
    "ac1" : "Linear_84",
    
    "ba1" : "Multilayer_21_128",
    "ba2" : "Multilayer_21_256",
    "ba3" : "Multilayer_21_512",
    "bb1" : "Multilayer_42_128",
    "bb2" : "Multilayer_42_256",
    "bb3" : "Multilayer_42_512",
    "bc1" : "Multilayer_84_128",
    "bc2" : "Multilayer_84_256",
    "bc3" : "Multilayer_84_512",
    
    "ca1" : "Multilayer_2_21_128",
    "ca2" : "Multilayer_2_21_256",
    "ca3" : "Multilayer_2_21_512",
    "cb1" : "Multilayer_2_42_128",
    "cb2" : "Multilayer_2_42_256",
    "cb3" : "Multilayer_2_42_512",
    "cc1" : "Multilayer_2_84_128",
    "cc2" : "Multilayer_2_84_256",
    "cc3" : "Multilayer_2_84_512",
    
    "da1" : "Conv_42_4",
    "da2" : "Conv_42_16",
    "da3" : "Conv_42_32",
    "da4" : "Conv_42_64",
    "db1" : "Conv_84_4",
    "db2" : "Conv_84_16",
    "db3" : "Conv_84_32",
    "db4" : "Conv_84_64",
    
    "ea1" : "Conv_2_4",
    "ea2" : "Conv_2_16",
    "ea3" : "Conv_2_32",
    "ea4" : "Conv_2_64",
    
    "fa1" : "Conv_3_4",
    "fa2" : "Conv_3_16",
    "fa3" : "Conv_3_32",
    "fa4" : "Conv_3_64",
    
    "ga1" : "Colorspace",
    "ha1" : "Colorspace_2"
}

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
    for y in between_letters:
        plt.axhline(y=y, color = "black", linewidth = 2, linestyle = "-")
    for y in between_subletters:
        plt.axhline(y=y, color = "black", linewidth = 1, linestyle = "--")

import matplotlib.pyplot as plt

def plot_boxes_acc(train_acc, test_acc, training_too = False):
    #plt.figure(figsize=(8,20))
    
    if(training_too):
        train_c = (0,0,1,.1)
        k_train = list(train_acc.keys())
        v_train = list(train_acc.values())
        k_train, v_train = zip(*sorted(zip(k_train, v_train)))
        k_train = list(k_train); k_train.reverse()
        v_train = list(v_train); v_train.reverse()
        train = plt.boxplot(v_train, vert = False, widths = .25,
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
    k_test = list(k_test); k_test.reverse()
    v_test = list(v_test); v_test.reverse()
    test = plt.boxplot(v_test, vert = False, widths = .75,
        patch_artist=True,
        boxprops=dict(facecolor=test_c, color=test_c),
        capprops=dict(color=test_c),
        whiskerprops=dict(color=test_c),
        flierprops=dict(color=test_c, markeredgecolor=test_c),
        medianprops=dict(color=test_c))
    
    label_list = [letters_to_name[k] + " (lambda {})".format(l) for k, l in k_test]
    
    plt.yticks(ticks = [i for i in range(1, len(k_test)+1)], labels = label_list)
    plt.title("Model accuracies")
    
    get_betweens(k_test)
    if(training_too):
        plt.legend([train["boxes"][0], test["boxes"][0]], ['Train accuracies', 'Test accuracies'], loc='lower right')
    else:
        plt.legend([test["boxes"][0]], ['Test accuracies'], loc='lower right')
    
    minimums = [min(l) for l in list(train_acc.values()) + list(test_acc.values())]
    minimum = min(minimums)
    #plt.ylim((minimum-3,100))

    plt.axvline(x=0, color = "black", linewidth = 2)
    plt.axvline(x=1, color = "gray",  linewidth = 1, linestyle = "--")
        
    plt.savefig("plots/lambda_boxes_acc{}".format("_with_training" if training_too else ""), bbox_inches='tight')
    plt.show()
    plt.close()