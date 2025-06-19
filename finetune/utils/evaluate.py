import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def evaluate(device, loader, model, mode, num_classes):
    trues, preds = [], []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, signals, labels in loader:
            images, signals, labels = images.to(device), signals.to(device), labels.to(device)
            
            if mode == 'fusion':
                outputs = model(images, signals)
            elif mode == 'signal':
                outputs = model(signals)
            elif mode == 'image':
                outputs = model(images)

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"NaN or Inf detected in outputs at batch{i}")
                print(outputs)
                break
                
            trues.append(labels.cpu().numpy())
            preds.append(torch.sigmoid(outputs).cpu().numpy())
    trues = np.vstack(trues)
    preds = np.vstack(preds)

    accs, aucs, f1s = [], [], []
    for i in range(num_classes):
        accs.append(accuracy_score(trues[:,i], (preds[:,i]>0.5).astype(int)))
        aucs.append(roc_auc_score(trues[:,i], preds[:,i]))
        f1s.append(f1_score(trues[:,i], (preds[:,i]>0.5).astype(int), average= 'binary'))

    return {
        'acc':accs, 'acc_avg':np.average(accs),
        'auc':aucs, 'auc_avg':np.average(aucs),
        'f1':f1s, 'f1_avg':np.average(f1s)
        }