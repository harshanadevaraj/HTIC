import numpy as np
import matplotlib.pyplot as plt
a=np.array([1,2,3])
weight=0.8
bias=1.5
x=weight*a+bias
def sigmoid(x):
    return 1/(1+np.exp(-x))
print(sigmoid(x))
def siggrad(x):
    return sigmoid(x)*(1-sigmoid(x))
print(siggrad(x))
def tanh(x):
    return np.exp(x)-np.exp(-x)/(np.exp(x)+np.exp(-x))
print(tanh(x))
def tanhgrad(x):
    return 1 - tanh(x)**2
print(tanhgrad(x))
def relu(x):
    return np.maximum(0, x)
print(relu(x))
def relugrad(x):
    return np.where(x > 0, 1, 0)
print(relugrad(x))
plt.plot(a,sigmoid(x) ,label='sigmoid')
plt.show()