import sys
import numpy as np
import torch
sys.path.append("./")
import matplotlib

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from learner.optim import SGD,Momentum,Adadelta,Adam,Nesterov
from learner.loss import cross_entropy
from learner.activation import sigmoid,relu,tanh,swish,gelu,softplus,prelu,hard_sigmoid,hard_tanh,leaky_relu,elu
from learner.dataset import CustomDataset
from learner.metrics import accuracy

matplotlib.style.use('ggplot')

class fnnNet():
    def __init__(self,hidden:list,activation,loss,optimization,last_act) -> None:
        self.hidden = hidden
        self.loss = loss
        self.optimization = optimization
        self.activation = activation
        self.last_act = last_act
        self.weights = None
        self.bias = None
        self.train_losses = []
        self.train_acces = []
        self.test_losses = []
        self.test_acces = []

    def fit(self,train_dataloader,test_dataloader = None,y_test = None,epochs=100,batch_size=32,verbose=True):
        n_dim_X,n_dim_y = train_dataloader.dataset.get_dimensions()
        self.weights,self.bias = self.generate_weights(n_dim_X=n_dim_X,n_dim_y=n_dim_y)
        
        for epoch in range(epochs):
            loss_train = 0.0
            for (X,y) in train_dataloader:
                with torch.no_grad():
                    y_pred = self.predict(X)
                    loss_train += self.loss(y_pred,y)/len(train_dataloader.dataset)
                    acc_train = accuracy(y_pred,y)
            self.train_losses.append(loss_train)
            self.train_acces.append(acc_train)

            if test_dataloader is not None:
                loss_test = 0.0
                for (X,y) in test_dataloader:
                    with torch.no_grad():
                        y_pred = self.predict(X)
                        loss_test += self.loss(y_pred,y)/len(test_dataloader.dataset)
                        acc_test = accuracy(y_pred,y)
                self.test_losses.append(loss_test)
                self.test_acces.append(acc_test)

            if verbose:
                if test_dataloader is not None:
                    print("Epoch:{0:0>3}\tTrain Loss:{1:<6f}\tTrain Accuracy:{2:<.2f}%\tTest Loss:{3:<6f}\tTest Accuracy:{4:<.2f}%".format(epoch, loss_train,acc_train*100,loss_test,acc_test*100))
                else:
                    print("Epoch:{0:0>3}\tTrain Loss:{1:<6f}\tTrain Accuracy:{2:<.2f}%".format(epoch, loss_train, acc_train*100))
            for (X,y) in train_dataloader:
                y_pred = self.predict(X)
                loss_now = self.loss(y_pred,y)
                loss_now.backward()
                self.optimization.step(self.weights,self.bias)

    def predict(self,X):
        for i in range(len(self.weights)-1):
            X = self.activation(X@self.weights[i] + self.bias[i])
        if self.last_act:
            result = sigmoid(X@self.weights[-1] + self.bias[-1])
        else:
            result = X@self.weights[-1] + self.bias[-1]
        return result
    
    def generate_weights(self,n_dim_X,n_dim_y):
        weight_dim = [n_dim_X] + self.hidden + [n_dim_y]
        weights = []
        bias = []
        for i in range(len(weight_dim)-1):
            weights.append(torch.randn(weight_dim[i],weight_dim[i+1],dtype=float,requires_grad=True))
            bias.append(torch.zeros(1,weight_dim[i+1],dtype=float,requires_grad=True))
        return weights,bias
    
    def plot_train_process(self):
        import matplotlib.pyplot as plt
        plt.plot(self.train_losses,label="Train Loss")
        plt.plot(self.train_acces,label="Train Accuracy")
        if len(self.test_losses) != 0:
            plt.plot(self.test_losses,label="Test Loss")
            plt.plot(self.test_acces,label="Test Accuracy")
        plt.legend()
        plt.show()
    
if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=10, n_informative=2, n_redundant=0, n_clusters_per_class=1,n_classes=2, random_state=42)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    
    X_train, y_train = torch.tensor(X_train,dtype=float),torch.tensor(y_train,dtype=float).reshape(y_train.shape[0],1)
    X_test,y_test = torch.tensor(X_test,dtype=float),torch.tensor(y_test,dtype=float).reshape(y_test.shape[0],1)
    
    train_set = CustomDataset(X_train,y_train)
    test_set = CustomDataset(X_test,y_test)
    
    
    train_dataloader = DataLoader(train_set,batch_size=256,shuffle=True)
    test_dataloader = DataLoader(test_set,batch_size=len(test_set),shuffle=False)

    activation = relu
    loss = cross_entropy
    optimization = Momentum(lr=8e-4)
    last_act = True
    
    fnn = fnnNet([5,5,5,5],activation=activation,loss=loss,optimization=optimization,last_act=last_act)
    fnn.fit(train_dataloader,test_dataloader=test_dataloader,epochs=20)
    fnn.plot_train_process()
    