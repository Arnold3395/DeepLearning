#%%
import sys
sys.path.append("./")

import torch
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from learner.metrics import accuracy,plot_confusion_matrix

matplotlib.style.use("ggplot")

class softmaxTrainer():
    def __init__(self,alpha=0.1,iterations=100):
        self.iterations = iterations
        self.alpha = alpha
        self.W = None
        self.losses = []
        self.acc = []

    def fit(self,X,y,verbose = True):
        n_samples = X.shape[0]
        n_dim_X = X.shape[1]
        n_dim_y = y.shape[1]
        self.W = torch.randn(n_dim_X,n_dim_y,dtype=float)
        for i in range(self.iterations):
            y_pred = self.predict_proba(X)

            loss_now = self.loss(y_pred,y)
            acc_now = accuracy(y_pred,y)
            self.losses.append(loss_now)
            self.acc.append(acc_now)
            if verbose:
                print("Epoch: {0}, Loss: {1}, Acc: {2}".format(i, loss_now,acc_now))
            self.W = self.W + self.alpha/n_samples * X.T@(y-y_pred)

    def predict_proba(self,X):
        y_pred = torch.exp(X@self.W) /torch.sum(torch.exp(X@self.W),dim=1).reshape(X.shape[0],1)
        return y_pred

    def predict(self,X):
        y_pred = self.predict_proba(X)
        y_pred = torch.argmax(y_pred,dim=1).reshape(X.shape[0],1)
        return y_pred

    def loss(self,y_pred,y_true):
        loss_now = -1/y_true.shape[0] * torch.sum(torch.sum(y_true*torch.log(y_pred),dim=1).reshape(y_true.shape[0],1))
        return loss_now

    def plot_train_process(self):
        plt.plot(self.acc, label='Accuracy', color='r', linestyle='--')
        plt.plot(self.losses, label='Loss', color='b', linestyle='-')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()


#%%
if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=0, n_clusters_per_class=1,n_classes=4,random_state=42)
    X,y_temp = torch.tensor(X,dtype=float),torch.tensor(y,dtype=float).reshape(y.shape[0],1)
    y_onehot = torch.zeros(y_temp.shape[0],torch.unique(y_temp).shape[0])
    for i in range(y_temp.shape[0]):
        ix = i
        iy = int(y_temp[i][0])
        y_onehot[ix][iy] = 1.0
    trainer = softmaxTrainer(alpha=0.1,iterations=1000)
    trainer.fit(X,y_onehot,verbose=True)
    trainer.plot_train_process()
    y_pred = trainer.predict(X)
    plot_confusion_matrix(y_pred,y_temp)