#%%
import torch
import torch.nn.functional as F

from utils import delete_these, plot_loss_acc, save_model, epochs
from get_data import get_batch

def train_test(model, E, batch_size = 128, show_after = 999999):
    train_losses = []; test_losses = []
    train_acc = [];    test_acc = []
    text_batch_size = batch_size * 10
    
    for e in range(1,epochs+1):
        E.update()
        model.train()
        x, y = get_batch(k_ = model.k, batch_size = batch_size, test = False)
        predicted = model(x)
        loss = F.nll_loss(predicted, y)
        model.opt.zero_grad()
        loss.backward()
        model.opt.step()
        train_losses.append(loss.item())
        accurate = [True if torch.argmax(p).item() == y[i].item() else False for i, p in enumerate(predicted)]
        train_acc.append(100*sum(accurate)/len(accurate))
                
        with torch.no_grad():
            model.eval()
            x, y = get_batch(k_ = model.k, batch_size = text_batch_size, test = True)
            predicted = model(x)
            loss = F.nll_loss(predicted, y)
            test_losses.append(loss.item())
            accurate = [True if torch.argmax(p).item() == y[i].item() else False for i, p in enumerate(predicted)]
            test_acc.append(100*sum(accurate)/len(accurate))

            if(e%show_after == 0 or e==epochs):
                plot_loss_acc(model, e, train_losses, test_losses, train_acc, test_acc)

    save_model(model)
    delete_these(False, model)
    return(train_losses[-1], test_losses[-1], train_acc[-1], test_acc[-1])



def train_test_short(model, batch_size = 128):
    text_batch_size = batch_size * 10
    model.train()
    x, y = get_batch(k_ = model.k, batch_size = batch_size, test = False)
    predicted = model(x)
    loss = F.nll_loss(predicted, y)
    train_loss = loss.item()
    accurate = [True if torch.argmax(p).item() == y[i].item() else False for i, p in enumerate(predicted)]
    train_acc = 100*sum(accurate)/len(accurate)
    
    with torch.no_grad():
        model.eval()
        x, y = get_batch(k_ = model.k, batch_size = batch_size, test = False)
        predicted = model(x)
        test_loss = loss.item()
        accurate = [True if torch.argmax(p).item() == y[i].item() else False for i, p in enumerate(predicted)]
        test_acc = 100*sum(accurate)/len(accurate)
    delete_these(False, model)
    return(train_loss, test_loss, train_acc, test_acc)
# %%
