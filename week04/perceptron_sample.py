import numpy as np
import sys
sys.path.insert(1, 'C:/KMOOC/KMOOC-ML/ipynb/code')
from  plot_xyw import  plot_xyw


x = np.array([[1.0, 1.0], [2.0, -2.0], [-1.0, -1.5], [-2.0,-1.0],[-2.0, 1.0],[1.5,-0.5]])
X = np.c_[np.ones(len(x)), x]

y = np.array([1,-1,-1,-1,1,1])
maxlabel, minlabel = y.max(), y.min()
np.random.seed(1)
#w = np.random.random((X.shape[1], 1))
w = np.array([0.0,1.0,0.5])
eta = 0.2
epochs = 1
for _ in range(epochs):
    for xi , yi in zip(X, y):
        xi = xi.reshape(w.shape)
        z = np.dot(w.T, xi)
        yhat = np.where(z >= 0.0, maxlabel, minlabel)
        delta = eta * (yi - yhat) * xi
        print(delta)
        w += delta

print(np.round(w,2))
plot_xyw(X,y,w,X0=True, savefig="JooML/week04/result_hw.png")