import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, auc, roc_curve, confusion_matrix
import warnings 
warnings.filterwarnings('ignore')

def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='\r')
    if (current == total):
        print()

def disparate_impact(y_true, y_pred_probs):
    # priv / unpriv
    n_class = y_pred_probs.shape[1]
    di = {}
    for i in range(n_class):
        # for this ith class is priv rest other are unpriv
        priv_ind = np.where(y_true == i)[0]
        npriv_ind = np.where(y_true != i)[0]
        
        priv_prob_mean = np.mean(y_pred_probs[priv_ind,i])
        npriv_prob_mean = np.mean(y_pred_probs[npriv_ind,i])
        
        di[i] = priv_prob_mean / npriv_prob_mean
    return di

def degree_of_bias(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm / cm.sum(axis=1)
    class_wise_acc = {}
    for i in range(cm.shape[0]):
        class_wise_acc[i] = cm[i][i]
    
    return np.std(list(class_wise_acc.values()))

def std_pre(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm / cm.sum(axis=0)
    cm[np.isnan(cm)] = 0.0
    class_wise_acc = {}
    for i in range(cm.shape[0]):
        class_wise_acc[i] = cm[i][i]
    
    return np.std(list(class_wise_acc.values()))
        
def metrices(y_true, y_pred, y_pred_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_prob = np.array(y_pred_prob)
    
    n_class = len(np.unique(y_true))
    
    print(classification_report(y_true, y_pred))
    
    dob = degree_of_bias(y_true, y_pred)
    print(f"degree of bias: {dob:.3f}")
    
    pre_std = std_pre(y_true, y_pred)
    print(f"std of precision: {pre_std:.3f}")
    
    print(confusion_matrix(y_true, y_pred))
    
    binarize_labels = label_binarize(y_true, classes = [i for i in range(n_class)])
    
    auc_vals = {}
    
    for i in range(n_class):
        fpr, tpr, _ = roc_curve(binarize_labels[:,i], y_pred_prob[:,i])
        auc_vals[i] = [fpr, tpr, auc(fpr, tpr)]
        
    di = disparate_impact(y_true, y_pred_prob)
    print('-----------------------------------------')
    for i in di.keys():
        print(f'DI for class {i}: {di[i]:.3f}')
    print('-----------------------------------------')

    return auc_vals

def cur_acc(scores, target):
    scores = F.softmax(scores,dim = 1)
    _,predicted = torch.max(scores,dim = 1)
    correct = (predicted == target).sum()
    samples = scores.shape[0]
    
    return correct / samples
        
def evaluate_mtl_metrics(model, loader, device, return_logs=False):
    y_true = []
    y_pred = []
    y_pred_prob = []
    
    curacc1 = 0; curacc2=0;curacc3=0;curaccens=0
    model.eval()
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(text, img, y) in enumerate(loader):
            text = text.to(device)
            img = img.to(device)
            y = y.to(device)

            scores, scores_cnn, scores_lstm = model(text, img)
            avg_logits = (scores + scores_cnn + scores_lstm)/3
            
            curacc1 += cur_acc(scores, y)/loader_len
            curacc2 += cur_acc(scores_cnn, y)/loader_len
            curacc3 += cur_acc(scores_lstm, y)/loader_len
            curaccens += cur_acc(avg_logits, y)/loader_len
            
            pred_probs = F.softmax(scores, dim=1)
            _,preds = pred_probs.max(dim=1)
            
            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
            y_pred_prob.extend(pred_probs.detach().cpu().numpy())
            
            if return_logs:
                progress(idx+1,loader_len)
                
    print(f"acc1: {curacc1:.3f}")
    print(f"acc2: {curacc2:.3f}")
    print(f"acc3: {curacc3:.3f}")
    print(f"acc_ens: {curaccens:.3f}")
    
    return metrices(y_true, y_pred, y_pred_prob)

def evaluate_mtl_metrics_pyr(model, loader, device, return_logs=False):
    y_true = []
    y_pred = []
    y_pred_prob = []
    
    curacc1 = 0; curacc2=0;curacc3=0;curaccens=0
    model.eval()
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            scores, scores_cnn, scores_lstm = model(x)
            avg_logits = (scores + scores_cnn + scores_lstm)/3
            
            curacc1 += cur_acc(scores, y)/loader_len
            curacc2 += cur_acc(scores_cnn, y)/loader_len
            curacc3 += cur_acc(scores_lstm, y)/loader_len
            curaccens += cur_acc(avg_logits, y)/loader_len
            
            pred_probs = F.softmax(scores, dim=1)
            _,preds = pred_probs.max(dim=1)
            
            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
            y_pred_prob.extend(pred_probs.detach().cpu().numpy())
            
            if return_logs:
                progress(idx+1,loader_len)
                
    print(f"acc1: {curacc1:.3f}")
    print(f"acc2: {curacc2:.3f}")
    print(f"acc3: {curacc3:.3f}")
    print(f"acc_ens: {curaccens:.3f}")
    
    return metrices(y_true, y_pred, y_pred_prob)

def evaluate_metrics(model, loader, device, return_logs=False):
    
    y_true = []
    y_pred = []
    y_pred_prob = []
    
    cur_acc1 = 0
    model.eval()
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            cur_acc1 += cur_acc(scores, y) / loader_len
            
            pred_probs = F.softmax(scores, dim=1)
            _,preds = pred_probs.max(dim=1)
            
            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
            y_pred_prob.extend(pred_probs.detach().cpu().numpy())
            
            if return_logs:
                progress(idx+1,loader_len)
                
    print(f"acc: {cur_acc1:.3f}")
    
    return metrices(y_true, y_pred, y_pred_prob)

def evaluate_metrics_lstmcnn(model, loader, device, return_logs=False):
    
    y_true = []
    y_pred = []
    y_pred_prob = []
    
    cur_acc1 = 0
    model.eval()
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(text, img, y) in enumerate(loader):
            text = text.to(device)
            img = img.to(device)
            y = y.to(device)

            scores = model(text, img)
            cur_acc1 += cur_acc(scores, y) / loader_len
            
            pred_probs = F.softmax(scores, dim=1)
            _,preds = pred_probs.max(dim=1)
            
            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
            y_pred_prob.extend(pred_probs.detach().cpu().numpy())
            
            if return_logs:
                progress(idx+1,loader_len)
                
    print(f"acc: {cur_acc1:.3f}")
    
    return metrices(y_true, y_pred, y_pred_prob)

def roc_plot(data, save_path):
    plt.figure(figsize=(5,4))
    for i in data.keys():
        plt.plot(data[i][0], data[i][1], label=f'{i}: {data[i][2]:.3f}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    
    

if __name__ == "__main__":
    pass
