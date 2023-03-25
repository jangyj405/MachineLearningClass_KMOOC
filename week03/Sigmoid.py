import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))



if __name__ == "__main__":
    x = np.arange(-10.0, 10.0, 0.1)
    y = sigmoid(x)
    dy = sigmoid_derivative(x)

    plt.plot(x,y)
    plt.plot(x,dy)
    plt.axvline(0, color = 'black')
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('h(x)', fontsize = 16)
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(0.0, 1.1, 0.2))
    plt.grid(axis = 'y')
    plt.show()