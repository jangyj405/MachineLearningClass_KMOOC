import sys
sys.path.insert(1, 'C:/KMOOC/KMOOC-ML/ipynb')

import joy
from OOP_perceptron import Perceptron
import matplotlib.pyplot as plt

X,y = joy.joy_data()
#X,y = joy.joy_Ndata()
#X,y = joy.toy_data()
ppn = Perceptron(eta=0.1, epochs=10)
ppn.fit(X,y)

joy.plot_xyw(X,y, ppn.w,savefig="result.png")
#joy.plot_xyw(X,y, ppn.w,savefig="result2.png")
#joy.plot_xyw(X,y, ppn.w,savefig="result3.png")

plt.close()
plt.plot(range(1,len(ppn.cost_) + 1), ppn.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Misclassified Samples')
plt.savefig('Epochs', dpi=150)
 