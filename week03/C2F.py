import numpy as np
import matplotlib.pyplot as plt

def activate(z):
    if(z < 32):
        z = 32
    return z

def C2F(C):
    F = 9/5.0 * C + 32
    return activate(F)

'''
test_c = [-20, -10, 0, 36.5, 40, 50, 100]
test_f = [C2F(c) for c in test_c]
'''
x = np.arange(-100,100, .1)
y = [C2F(ix) for ix in x]

plt.figure()
plt.plot(x,y)
plt.axis([-20,50,0,150])
plt.xlabel('Celcius')
plt.ylabel('Fahrenheit')
plt.show()