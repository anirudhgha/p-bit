import pbit
import numpy as np

a = [[1, 2, 3], [4, 5, 6],[7, 8, 9]]
a = np.asarray(a)
a[0:2,0:2] = 1

print(a)