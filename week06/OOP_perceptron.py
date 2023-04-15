import numpy as np

class Perceptron:
    def __init__(self, eta=0.1, epochs=10,random_seed=1):
        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed

    def fit(self, X, y, X0=False):
        if X0 == False:
            X = np.c_[np.ones(len(y)),X]
        np.random.seed(self.random_seed)
        self.w = np.random.random(X.shape[1])
        self.maxy, self.miny = y.max(), y.min()
        self.cost_ = []
        self.w_ = np.array([self.w])
        
        for i in range(self.epochs):
            errors = 0
            for xi, yi in zip(X,y):
                yhat = self.activate(xi)
                delta = self.eta * (yi - yhat) * xi
                self.w = self.w + delta
                if (yi != yhat): errors += 1
            self.cost_.append(errors)
            self.w_ = np.vstack([self.w_, self.w])
        return self

    def net_input(self, X):
        if X.shape[0] == self.w.shape[0]:
            z = np.dot(self.w.T, X)
        else:
            z = np.dot(X, self.w[1:]) + self.w[0]
        return z
    
    def activate(self, X):
        mid = (self.maxy + self.miny) * 0.5
        return np.where(self.net_input(X) > mid, self.maxy, self.miny)
    
    def predict(self, X):
        return self.activate(X)
    
