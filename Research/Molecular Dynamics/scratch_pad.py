import pbit
import numpy as np
import matplotlib.pyplot as plt

J = [[0, -2, -2],
     [-2, 0, 1],
     [-2, 1, 0]]
h = [2, -1, -1]
myp = pbit.pcircuit(J=J, h=h)

samples = myp.generate_samples(1e5, ret_base=10)

plt.hist(samples)
plt.show()
