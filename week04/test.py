
import sys
sys.path.insert(1, 'C:/KMOOC/KMOOC-ML/ipynb/code')
from  plot_xyw import  plot_xyw
import numpy as np 
from perceptron import perceptron
from perceptronV2 import perceptronV2

x = np.array([[1.0, 1.0], [2.0, -2.0], [-1.0, -1.5], [-2.0,-1.0],[-2.0, 1.0],[1.5,-0.5]])
X = np.c_[np.ones(len(x)), x]
y = np.array([1,-1,-1,-1,1,1])
'''
w = np.array([0, 1.0, 0.5])
W = np.array([w])
epochs = 4
for _ in range(epochs):
    w = perceptron(X, y, w, eta=0.05, epochs=1)
    W = np.vstack([W,w])

plot_xyw(X,y,W,X0=True, savefig="JooML/week04/result.png")
'''

'''
w = np.array([0, 1.0, 0.5])
w = perceptronV2(X, y, w, eta=0.1, epochs=3)
plot_xyw(X,y,w,X0=True, savefig="JooML/week04/result_pv2.png")
'''

