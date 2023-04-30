import numpy as np

class AdalineGD():
    def __init__(self, eta = 0.01, epochs = 10, random_seed = 1):
        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed
        pass
    
    def net_input(self, X):
        z = np.dot(X, self.w[1:]) + self.w[0]
        return z
    
    def activation(self, X):
        return X
    
    def fit(self, X, y):
        np.random.RandomState(self.random_seed)
        self.w = np.random.random(size=X.shape[1]+1)
        self.maxy = y.max()
        self.miny = y.min()
        self.cost_ = []
        self.w_ = np.array([self.w])

        for i in range(self.epochs):
            Z = self.net_input(X)
            yhat = self.activation(Z)
            errors = y - yhat
            self.w[1:] += self.eta * np.dot(errors,X)
            self.w[0] += self.eta * np.sum(errors)
            cost = 0.5 * np.sum(errors**2)
            self.cost_.append(cost)
            self.w_ = np.vstack([self.w_, self.w])
        return self
    
    def fit_with_momentum(self, X, y):
        np.random.RandomState(self.random_seed)
        self.w = np.random.random(size=X.shape[1]+1)
        self.maxy = y.max()
        self.miny = y.min()
        self.cost_ = []
        self.w_ = np.array([self.w])

        '''Momentum'''
        self.v1 = np.zeros_like(self.w[1:])
        self.v2 = np.zeros_like(self.w[0])
        gamma = 0.5
        for i in range(self.epochs):
            Z = self.net_input(X)
            yhat = self.activation(Z)
            errors = y - yhat

            self.v1 = gamma * self.v1 + self.eta*np.dot(errors, X)
            self.v2 = gamma * self.v2 + self.eta*np.sum(errors)

            self.w[1:] += self.v1
            self.w[0] += self.v2
            cost = 0.5 * np.sum(errors**2)
            self.cost_.append(cost)
            self.w_ = np.vstack([self.w_, self.w])
        return self 
    
    def predict(self, X):
        mid = (self.maxy + self.miny) * 0.5
        Z = self.net_input(X)
        yhat = self.activation(Z)
        return np.where(yhat>mid, self.maxy, self.miny)