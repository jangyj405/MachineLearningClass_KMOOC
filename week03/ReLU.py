import numpy as np
import matplotlib.pyplot as plt

def reLU(x):
    return np.maximum(0,x)


if __name__ == "__main__":
    x = np.arange(-10.0, 10.0, 0.1)
    y = reLU(x)

    plt.plot(x,y)
    plt.axvline(0, color = 'black')
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('h(x)', fontsize = 16)
    plt.grid(axis = 'y')
    plt.show()