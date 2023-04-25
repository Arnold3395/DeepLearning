import torch
import seaborn as sns

import matplotlib.pyplot as plt

def accuracy(y_pred,y_true):
    if y_pred.shape[1] == 1:
        y_pred = torch.round(y_pred)
        result = torch.sum(y_pred == y_true)/y_true.shape[0]
    else:
        y_pred = torch.argmax(y_pred,dim=1).reshape(y_pred.shape[0],1)
        y_true = torch.argmax(y_true,dim=1).reshape(y_true.shape[0],1)
        result = torch.sum(y_pred == y_true)/y_true.shape[0]
    return result

def recall(y_pred,y_true):
    tp_fn = torch.sum(y_true)
    tp = torch.sum(y_pred*y_true)
    recall_score = tp/tp_fn
    return recall_score

def precision(y_pred,y_true):
    tp_fp = torch.sum(y_pred)
    tp = torch.sum(y_pred*y_true)
    precision_score = tp/tp_fp
    return precision_score

def f1_socre(y_pred,y_true):
    recall_score = recall(y_pred,y_true)
    precision_score = precision(y_pred,y_true)
    f1score = 2*recall_score*precision_score/(recall_score+precision_score)
    return f1score

def f_score(y_pred,y_true,beta=1.0):
    recall_score = recall(y_pred,y_true)
    precision_score = precision(y_pred,y_true)
    fscore = (1+beta**2)*precision_score*recall_score/(beta**2*precision_score+recall_score)
    return fscore

def confusion_matrix(y_pred,y_true):
    result_matrix = torch.zeros(torch.unique(y_true).shape[0],torch.unique(y_true).shape[0])
    for i in range(y_true.shape[0]):
        ix = int(y_true[i][0])
        iy = int(y_pred[i][0])
        result_matrix[ix][iy] += torch.tensor(1.0)
    return result_matrix

def plot_confusion_matrix(y_pred,y_true):
    con_mat = confusion_matrix(y_pred,y_true)
    sns.heatmap(con_mat,annot=True,fmt='.20g',cmap=plt.cm.Blues)
    plt.tick_params(top=True, labeltop=True,bottom=False,labelbottom=False)
    plt.title("Confusion Matrix")
    plt.show()

