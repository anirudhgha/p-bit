"""
Created on Tue Jan 28 16:54:53 2020

@author: anirudh_ghantasala
compare_psl.py replicated using pbit package
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import pbit

random.seed(0)
np.random.seed(0)

# set parameters
Nm = 3
Nt = 100000
I0 = 1
d_t = 1 / (2 * Nm)

# construct weights
J = np.add(np.random.rand(Nm, Nm) * 2, -1)
J = (J + J.T) / 2
np.fill_diagonal(J, 0)
h = np.zeros(Nm)

J = [[0, -2, -2],
     [-2, 0, 1],
     [-2, 1, 0]]
h = [2, -1, -1]

# main
myPcircuit = pbit.pcircuit(Nm, J, h, beta=I0, model="cpsl")  # build a p-circuit
mcpsl = myPcircuit.runFor(Nt)  # save Nt snapshots of the network
myPcircuit.setModel("ppsl", dt=d_t)  # modify the p-circuit to use the ppsl model
mppsl = myPcircuit.runFor(Nt)  # save Nt snapshots

print(mcpsl.shape)

histboltz = myPcircuit.getBoltzmann() * Nt

histcpsl = np.array([0 for i in range(2 ** Nm)])
histppsl = np.array([0 for i in range(2 ** Nm)])

for i in range(Nt):
    histcpsl[pbit.convertToBase10(mcpsl[i, :])] += 1
    histppsl[pbit.convertToBase10(mppsl[i, :])] += 1

# interchange histboltz, histcpsl, and histppsl to test different functions
barWidth = 0.25
x1 = np.arange(2 ** Nm)
x2 = np.array([x + barWidth for x in x1])
x3 = np.array([x + barWidth for x in x2])

plt.bar(x1, histcpsl, width=barWidth, edgecolor='white', label='cpsl')
plt.bar(x2, histppsl, width=barWidth, edgecolor='white', label='ppsl')
plt.bar(x3, histboltz, width=barWidth, edgecolor='white', label='boltz')
plt.legend()
plt.show()
