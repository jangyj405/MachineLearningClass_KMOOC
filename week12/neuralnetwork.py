import numpy as np
import sys
sys.path.insert(1, 'C:/KMOOC/KMOOC-ML/ipynb')
import joy

class NeuralNetwork():
    def __init__(self, net_arch, eta = 0.1, epochs = 10000, random_seed = 1):
        self.layers = len(net_arch)
        self.net_arch = net_arch
        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed
        pass

    def g(self, x):
        return 1/(1+np.exp(-x))
    
    def g_prime(self, x):
        return self.g(x) * (1-self.g(x))
    
    def fit(self, X, Y):
        np.random.seed(self.random_seed)
        W1_shape = (self.net_arch[1], self.net_arch[0])
        W2_shape = (self.net_arch[2], self.net_arch[1])
        self.W1 = 2 * np.random.random(W1_shape)-1
        self.W2 = 2 * np.random.random(W2_shape)-1
        self.cost_ = []

        for _ in range(self.epochs):
            A0 = X
            Z1 = np.dot(self.W1, A0)
            A1 = self.g(Z1)
            Z2 = np.dot(self.W2, A1)
            A2 = self.g(Z2)

            E2 = Y - A2
            E1 = np.dot(self.W2.T, E2)

            dZ2 = E2 * self.g_prime(Z2)
            dZ1 = E1 * self.g_prime(Z1)

            self.W2 += np.dot(dZ2, A1.T)
            self.W1 += np.dot(dZ1, A0.T)
            self.cost_.append(np.sqrt(np.sum(E2*E2)))
        return self
    '''   
    def net_input(self, X):
        if X.shape[0] == self.w.shape[0]:
            return np.dot(X, self.w)
        else:
            return np.dot(X, self.w[1:]) + self.w[0]
    '''    
    def predict(self, X):
        Z1 = np.dot(self.W1, X)
        A1 = self.g(Z1)
        Z2 = np.dot(self.W2, A1)
        A2 = self.g(Z2)
        return A2
        