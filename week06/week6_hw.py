import sys
sys.path.insert(1, 'C:/KMOOC/KMOOC-ML/ipynb')
from matplotlib import pyplot as plt
import joy 
X, y = joy.iris_data()

ppn = joy.Perceptron(eta=0.01, epochs=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.cost_) + 1), ppn.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
joy.plot_xyw(X,y, ppn.w ,savefig="hw")