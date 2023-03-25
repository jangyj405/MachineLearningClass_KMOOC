import numpy as np
import matplotlib.pyplot as plt

from Sigmoid import sigmoid

def tanh(x):
    return 2*sigmoid(2*x)-1

def tanh_derivative(x):
    return (1 + tanh(x)) * (1-tanh(x))

if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = tanh(x)
    dy = tanh_derivative(x)
    s = sigmoid(x)
    plt.plot(x,y, label='tanh')
    plt.plot(x,dy, label="tanh'")
    plt.plot(x, s, label='sigmoid')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-1.1, 1.1)
    plt.legend(loc='best')
    plt.show()