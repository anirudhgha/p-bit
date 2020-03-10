import numpy as np


a = np.ones((1, 5))[0] * 3
b = np.ones((1, 4))[0] * 4
m = np.diag(a, 0) + np.diag(b, -1) + np.diag(b, 1)

print(m)