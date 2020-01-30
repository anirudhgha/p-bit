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
d_t = 1 / (3 * Nm)

# construct weights
J = np.add(np.random.rand(Nm, Nm) * 2, -1)
J = (J + J.T) / 2
np.fill_diagonal(J, 0)
h = np.zeros(Nm)

J = [[0, -2, -2],
     [-2, 0, 1],
     [-2, 1, 0]]
h = [2, -1, -1]

my_pcircuit = pbit.pcircuit(Nm, J, h, beta=I0, model="cpsl")  # build a p-circuit
mcpsl = my_pcircuit.runFor(Nt)  # run network for Nt timesteps
my_pcircuit.setModel("ppsl", dt=d_t)  # modify the p-circuit to use the ppsl model
mppsl = my_pcircuit.runFor(Nt)  # run network for Nt timesteps

histboltz = my_pcircuit.getBoltzmann()
histcpsl = np.array([0 for i in range(2 ** Nm)])
histppsl = np.array([0 for i in range(2 ** Nm)])

for i in range(Nt):
    histcpsl[pbit.convertToBase10(mcpsl[i, :])] += 1  # build histogram of Nt states
    histppsl[pbit.convertToBase10(mppsl[i, :])] += 1

histcpsl = histcpsl / np.sum(histcpsl)
histppsl = histppsl / np.sum(histppsl)

# calculate error from boltz
ErrorPSL_cpsl = np.sqrt(np.divide(np.sum(np.sum((np.abs(np.subtract(histboltz, histcpsl))) ** 2)),
                                  np.sum(np.sum((np.abs(histboltz)) ** 2))))
ErrorPSL_ppsl = np.sqrt(np.divide(np.sum(np.sum((np.abs(np.subtract(histboltz, histppsl))) ** 2)),
                                  np.sum(np.sum((np.abs(histboltz)) ** 2))))

print("cpsl mse: ", ErrorPSL_cpsl, "\nppsl mse: ", ErrorPSL_ppsl)

# plot
barWidth = 0.25
x1 = np.arange(2 ** Nm)
x2 = np.array([x + barWidth for x in x1])
x3 = np.array([x + barWidth for x in x2])
plt.bar(x1, histcpsl, width=barWidth, edgecolor='white', label='cpsl')
plt.bar(x2, histppsl, width=barWidth, edgecolor='white', label='ppsl')
plt.bar(x3, histboltz, width=barWidth, edgecolor='white', label='boltz')
plt.legend()
plt.show()
