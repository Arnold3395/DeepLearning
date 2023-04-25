import sys
sys.path.append("./")

import matplotlib
import torch

import numpy as np
import matplotlib.pyplot as plt
from learner.metrics import accuracy,plot_confusion_matrix
from sklearn.datasets import  make_classification

matplotlib.style.use('ggplot')

class perceptronTrainer():
    def __init__(self,iterations=100):
        self.iterations = iterations
        self.w = None
        self.acc = []

    def fit(self,X,y,verbose=True):
        n_samples,n_dim_X = X.shape[0],X.shape[1]
        self.w = torch.rand(n_dim_X,1,dtype=float)
        t = 0
        while t < self.iterations:
            random_rank = torch.randperm(n_samples)
            for i in random_rank:
                y_pred = self.predict(X)
                acc_now = torch.sum(y_pred == y)/n_samples
                self.acc.append(acc_now)
                if verbose:
                    print("Iteration: {0}, Acc: {1}".format(t,acc_now))
                if (y[i]*X[i]@self.w)[0] <= 0:
                    self.w += y[i]*X[i].reshape(n_dim_X,1)
                t += 1
                if t >= self.iterations:
                    break

    def predict(self,X):
        result = X@self.w/torch.abs(X@self.w)
        return result

    def plot_train_process(self,smooth = True):
        if smooth:
            acc_smooth = self.acc.copy()
            for i in range(32,len(self.acc)):
                acc_smooth[i] = np.mean(acc_smooth[i-32:i+1])
            plt.plot(acc_smooth,label='Accuracy Smoothed',color='b',linestyle='-')
        plt.plot(self.acc, label='Accuracy', color='r', linestyle='--')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=0, n_clusters_per_class=1,n_classes=2, random_state=42)
    y = [x if x>0  else -1 for x in y ]
    X,y = torch.tensor(X,dtype=float),torch.tensor(y,dtype=float).reshape(1000,1)
    trainer = perceptronTrainer(iterations=10000)
    trainer.fit(X,y)
    trainer.plot_train_process()