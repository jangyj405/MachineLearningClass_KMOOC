import numpy as np


def perceptron_train(X, y, X0 = False, eta= 0.1, epochs = 1, random_seed = 1):
    if X0 == False:
        X = np.c_[ np.ones(len(X)), X ]
    
    randnum = np.random.RandomState(random_seed) 
    w = randnum.normal(loc=0.0, scale=0.01, size=X.shape[1])
           
    maxlabel, minlabel = y.max(), y.min()                
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = np.dot(xi, w)                           
            yhat = np.where(z > 0.0, maxlabel, minlabel)  
            if yi != yhat:
                delta = eta * (yi - yhat) * xi                
                w += delta                                  
        
    return w

def perceptron_predict(X, w):
    z = np.dot(X, w)
    yhat = np.where(z > 0., 1, 0)
    return yhat


data = np.genfromtxt("joydata.txt")
np.random.seed(1)
np.random.shuffle(data)
devide = int(data.shape[0] * 0.7)
train = data[:devide]
test = data[devide:]


train = np.c_[np.ones(len(train)), train]
test = np.c_[np.ones(len(test)), test]

print(train[:,:3])
print(test.shape)

epo = 2
et = 0.02
rand = 5
w = perceptron_train(train[:,:3], train[:,3], True, epochs=epo, eta=et, random_seed=rand)

m_samples = test.shape[0]
yhat = perceptron_predict(test[:,:3], w)
missed = np.sum(yhat.flatten() != test[:,3])
print('Misclassified:{}/{}'.format(missed, m_samples))

m_samples = train.shape[0]
yhat = perceptron_predict(train[:,:3], w)
missed = np.sum(yhat.flatten() != train[:,3])
print('Misclassified:{}/{}'.format(missed, m_samples))