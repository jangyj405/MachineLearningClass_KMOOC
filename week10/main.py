from neuralnetwork import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(1, 'C:/KMOOC/KMOOC-ML/ipynb')
import joy

nn = NeuralNetwork(net_arch=[2,4,1], epochs=1000)

X = np.array([[0,0,1,1],[0,1,0,1]])
Y = np.array([0,1,1,0])
nn.fit(X,Y)

A2 = nn.predict(X)
for x, yhat in zip(X.T, A2.T):
    print(x, np.round(yhat,3))

joy.plot_decision_regions(X.T, Y, lambda z:nn.predict(z.T))
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend(loc='best')
plt.show()