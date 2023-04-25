import torch
import time

class SGD():
    def __init__(self,lr=0.01) -> None:
        self.lr = lr

    def step(self,weights,bias):
        for i in range(len(weights)):
            weights[i].data -= self.lr * weights[i].grad.data
            bias[i].data -= self.lr * bias[i].grad.data
            weights[i].grad.data.zero_()
            bias[i].grad.data.zero_()

class Adam():
    def __init__(self,lr=0.01,beta1=0.9,beta2=0.999,eps=1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def step(self,weight,bias):
        if self.m is None:
            self.m = []
            self.v = []
            for i in range(len(weight)):
                self.m.append(torch.zeros_like(weight[i]))
                self.v.append(torch.zeros_like(weight[i]))
        for i in range(len(weight)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * weight[i].grad.data
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * weight[i].grad.data ** 2
            m_hat = self.m[i] / (1 - self.beta1)
            v_hat = self.v[i] / (1 - self.beta2)
            weight[i].data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            bias[i].data -= self.lr * bias[i].grad.data / (torch.sqrt(v_hat) + self.eps)
            weight[i].grad.data.zero_()
            bias[i].grad.data.zero_()

class RMSprop():
    def __init__(self,lr=0.01,beta=0.9,eps=1e-8) -> None:
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = None

    def step(self,weight,bias):
        if self.v is None:
            self.v = []
            for i in range(len(weight)):
                self.v.append(torch.zeros_like(weight[i]))
        for i in range(len(weight)):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * weight[i].grad.data ** 2
            v_hat = self.v[i] / (1 - self.beta)
            weight[i].data -= self.lr * weight[i].grad.data / (torch.sqrt(v_hat) + self.eps)
            bias[i].data -= self.lr * bias[i].grad.data / (torch.sqrt(v_hat) + self.eps)
            weight[i].grad.data.zero_()
            bias[i].grad.data.zero_()
            
class Adagrad():
    def __init__(self,lr=0.01,eps=1e-8) -> None:
        self.lr = lr
        self.eps = eps
        self.v = None

    def step(self,weight,bias):
        if self.v is None:
            self.v = []
            for i in range(len(weight)):
                self.v.append(torch.zeros_like(weight[i]))
        for i in range(len(weight)):
            self.v[i] += weight[i].grad.data ** 2
            weight[i].data -= self.lr * weight[i].grad.data / (torch.sqrt(self.v[i]) + self.eps)
            bias[i].data -= self.lr * bias[i].grad.data / (torch.sqrt(self.v[i]) + self.eps)
            weight[i].grad.data.zero_()
            bias[i].grad.data.zero_()
            
class Adadelta():
    def __init__(self,lr=0.01,beta=0.9,eps=1e-8) -> None:
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = None
        self.s = None

    def step(self,weight,bias):
        if self.v is None:
            self.v = []
            self.s = []
            for i in range(len(weight)):
                self.v.append(torch.zeros_like(weight[i]))
                self.s.append(torch.zeros_like(weight[i]))
        for i in range(len(weight)):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * weight[i].grad.data ** 2
            v_hat = self.v[i] / (1 - self.beta)
            weight[i].data -= self.lr * weight[i].grad.data / (torch.sqrt(v_hat) + self.eps)
            bias[i].data -= self.lr * bias[i].grad.data / (torch.sqrt(v_hat) + self.eps)
            weight[i].grad.data.zero_()
            bias[i].grad.data.zero_()
            
            
class Momentum():
    def __init__(self,lr=0.01,beta=0.9) -> None:
        self.delta_weights_grad = []
        self.delta_bias_grad = []
        self.lr = lr
        self.beta = beta

    def step(self,weights,bias):
        if len(self.delta_weights_grad) == 0:
            for i in range(len(weights)):
                self.delta_weights_grad.append(torch.zeros_like(weights[i],dtype=float))
                self.delta_bias_grad.append(torch.zeros_like(bias[i],dtype=float))
        for i in range(len(weights)):
            self.delta_weights_grad[i] = self.beta * self.delta_weights_grad[i] + (1 - self.beta) * weights[i].grad.data
            self.delta_bias_grad[i] = self.beta * self.delta_bias_grad[i] + (1 - self.beta) * bias[i].grad.data
            weights[i].data -= self.lr * self.delta_weights_grad[i]
            bias[i].data -= self.lr * self.delta_bias_grad[i]
            weights[i].grad.data.zero_()
            bias[i].grad.data.zero_()

        # print(weights)
        # time.sleep(10)
        
        
class Nesterov():
    def __init__(self,lr=0.01,beta=0.9) -> None:
        self.lr = lr
        self.beta = beta
        self.v = None

    def step(self,weight,bias):
        if self.v is None:
            self.v = []
            for i in range(len(weight)):
                self.v.append(torch.zeros_like(weight[i]))
        for i in range(len(weight)):
            self.v[i] = self.beta * self.v[i] + self.lr * weight[i].grad.data
            weight[i].data -= self.beta * self.v[i] + (1 + self.beta) * self.lr * weight[i].grad.data
            bias[i].data -= self.lr * bias[i].grad.data
            weight[i].grad.data.zero_()
            bias[i].grad.data.zero_()
