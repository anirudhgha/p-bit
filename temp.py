import numpy as np


J = np.ndarray.astype(np.array([[0, -2, -2],
                        [-2, 0, 1],
                        [-2, 1, 0]]), 'float64')
Jflat = np.ndarray.flatten(J)
print(Jflat)