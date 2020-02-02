import pbit
import matplotlib.pyplot as plt
import numpy as np


J = [[0, -2, -2],
     [-2, 0, 1],
     [-2, 1, 0]]
h = [2, -1, -1]

my_pcircuit = pbit.pcircuit(J, h)
m = my_pcircuit.runFor(1000)
decimal = pbit.convertToBase10(m)
histcpsl = [0 for i in range(2**3)]
for i in range(1000):
    histcpsl[decimal[i]] += 1

plt.bar(np.arange(2**3), histcpsl)
plt.show()
