import pandas as pd
import sys
sys.path.insert(1, 'C:/KMOOC/KMOOC-ML/ipynb')
import joy
from adaline import *
import matplotlib.pyplot as plt
'''
df = pd.read_csv('https://archive.ics.uci.edu/'
                 'ml/machine-learning-databases/'
                 'iris/iris.data', header=None)

print(df.head())
'''


'''
X, y = joy.iris_data()
ada = AdalineGD(epochs=10, eta=0.1)
ada.fit(X,y)
joy.plot_xyw(X,y,ada.w, savefig='result0')

plt.close()
plt.plot(range(1,len(ada.cost_) + 1), np.log10(ada.cost_), marker='o')
plt.xlabel('Epochs')
plt.ylabel('log(SSE)')
plt.savefig('SSE0')
plt.show()
'''
'''
X, y = joy.iris_data()
ada = AdalineGD(epochs=10, eta=0.0001)
ada.fit(X,y)
joy.plot_xyw(X,y,ada.w, savefig='result1')

plt.close()
plt.plot(range(1,len(ada.cost_) + 1), np.log10(ada.cost_), marker='o')
plt.xlabel('Epochs')
plt.ylabel('log(SSE)')
plt.savefig('SSE1')
plt.show()
'''

'''
Xstd, y = joy.iris_data(standardized=True)
ada = AdalineGD(epochs=10, eta=0.0001)
ada.fit(Xstd,y)
joy.plot_xyw(Xstd,y,ada.w, savefig='result2')

plt.close()
plt.plot(range(1,len(ada.cost_) + 1), np.log10(ada.cost_), marker='o')
plt.xlabel('Epochs')
plt.ylabel('log(SSE)')
plt.savefig('SSE2')
plt.show()
'''

#momentum
Xstd, y = joy.iris_data(standardized=True)
ada = AdalineGD(epochs=1000, eta=0.0001)
ada.fit_with_momentum(Xstd,y)
joy.plot_xyw(Xstd,y,ada.w, savefig='result3')

plt.close()
plt.plot(range(1,len(ada.cost_) + 1), np.log10(ada.cost_), marker='o')
plt.xlabel('Epochs')
plt.ylabel('log(SSE)')
plt.savefig('SSE3')
plt.show()