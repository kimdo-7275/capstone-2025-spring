import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def eval(loader, model, device, n_class) :
    trues, preds = [], []
    model.eval()
    with torch.no_grad() :
        for images, signals, targets in loader :
            images, signals, targets = images.to(device), signals.to(device), targets.to(device)
            
            outputs = model(images, signals)
            outputs = torch.sigmoid(outputs)
            trues.append(targets.cpu().numpy())
            preds.append(outputs.cpu().numpy())
    trues = np.vstack(trues)
    preds = np.vstack(preds)

    accs, aucs, f1s = [], [], []
    for i in range(n_class) :
            accs.append(accuracy_score(trues[:,i], (preds[:,i]>0.5).astype(int)))
            aucs.append(roc_auc_score(trues[:,i], preds[:,i]))
            f1s.append(f1_score(trues[:,i], (preds[:,i]>0.5).astype(int), average= 'binary'))
    
    return {'acc':accs, 'acc_avg':np.average(accs),
            'auc':aucs, 'auc_avg':np.average(aucs),
            'f1':f1s, 'f1_avg':np.average(f1s)}