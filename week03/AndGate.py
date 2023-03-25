import numpy as np

def AND(x1, x2):
    x = np.array([1,x1,x2])
    w = np.array([-0.7,0.5, 0.5])
    return int( np.dot(w,x) > 0 )

print("AND 0 0 = ", AND(0,0))
print("AND 0 1 = ", AND(0,1))
print("AND 1 0 = ", AND(1,0))
print("AND 1 1 = ", AND(1,1))