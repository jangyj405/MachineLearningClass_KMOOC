import numpy as np

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


w1 = np.array([0.1, 0.9, 0.5, 0.7, 0.3, 0.7]).reshape((3,2))
w2 = np.array([0.1, 0.9, 0.3, 0.5, 0.7, 0.2]).reshape((2,3))
x = np.array([1,0]).reshape((2,1))

y_hat = relu(np.dot(w2 ,relu(np.dot(w1, x))))
print(y_hat)
##############################################

w1 = np.array([0.1, 0.0, -0.5, 0.7, -0.3, 0.7]).reshape((3,2))
w2 = np.array([0.4, 0.5, -0.2, -0.1, 0.0, -0.3, -0.5, 0.7, -0.2]).reshape((3,3))
w3 = np.array([-0.1, 0.0, -0.3, -0.5, 0.7, -0.2]).reshape(2,3)
x = np.array([1,0]).reshape((2,1))

y_hat = sigmoid(np.dot(w3 ,sigmoid(np.dot(w2 ,sigmoid(np.dot(w1, x))))))
print(y_hat)

