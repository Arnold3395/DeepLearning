import sys
sys.path.append("./")

import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from learner.metrics import plot_confusion_matrix
matplotlib.style.use('ggplot')

class LogisticTrainer():
    def __init__(self, alpha = 0.01,iterations = 30):
        self.alpha = alpha
        self.iterations = iterations
        self.w = None
        self.accuracy = []
        self.losses = []
    # 拟合
    def fit(self,X,y,verbose = True,theta = 0.5):
        self.w = torch.rand(X.shape[1],1,dtype=float)
        for i in range(self.iterations):
            # 获得预测值
            y_pred = self.predict_proba(X)
            # 获取当前损失值
            loss_now = self.loss(y, y_pred)
            # 获取当前准确率
            y_pred_bi = self.proba2bi(y_pred,theta=theta)
            acc_now = 1 - torch.sum(torch.abs(y_pred_bi - y)) / y.shape[0]
            if verbose:
                print("Epoch: {0}, Loss: {1}, Acc: {2}".format(i, loss_now,acc_now))
            # 储存准确率和损失值
            self.accuracy.append(acc_now)
            self.losses.append(loss_now)
            # 梯度下降
            self.w = self.w + self.alpha * 1/y.shape[0] * torch.sum(X*(y-y_pred),dim=0).reshape(self.w.shape[0],1)
    # 预测连续值
    def predict_proba(self,X):
        if self.w is None:
            raise TypeError
        prediction = 1/(1+torch.exp(-X@self.w))
        return prediction
    # 预测概率值转换
    def proba2bi(self,y_pred,theta=0.5):
        y_pred_bi = torch.tensor(list(map(lambda x: [1] if x[0] >= theta else [0], y_pred)))
        return y_pred_bi
    # 预测二元值
    def predict(self,X,theta = 0.5):
        y_pred = self.predict_proba(X)
        y_pred = self.proba2bi(y_pred,theta=theta)
        return y_pred
    # 损失函数
    def loss(self,y_true,y_pred):
        result = -1/y_true.shape[0]*torch.sum(y_true*torch.log(y_pred)+(1-y_true)*torch.log(1-y_pred))
        return result
    # 画图
    def plot_train_process(self):
        plt.plot(self.accuracy,label='Accuracy',color='r',linestyle = '--')
        plt.plot(self.losses,label='Loss',color = 'b',linestyle = '-')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

if __name__ == '__main__':

    n_samples = 5000
    X,y = make_classification(n_samples=n_samples,n_features=3,n_informative=2,n_redundant=0,n_classes=2,random_state=2)
    X,y = torch.tensor(X,dtype=float),torch.tensor(y,dtype=float).reshape(n_samples,1)

    trainer = LogisticTrainer(0.1,100)
    trainer.fit(X,y)
    trainer.plot_train_process()

    y_pred = trainer.predict(X)
    plot_confusion_matrix(y_pred,y)