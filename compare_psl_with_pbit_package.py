# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:54:53 2020

@author: alasg

compare_psl.py replicated using pbit package
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import random
import pbit

random.seed(0)

I0 = 1
Nt, Nm = 10000, 6  # Nm <= b size limit yields boltzmann curve as well
d_t = 1/Nm
boltz_size_lim = 20

J = np.add(np.random.rand(Nm, Nm) * 2, -1)  # matrix of values between [-1, 1)
J = (J + J.T)/2
np.fill_diagonal(J, 0)
h = np.zeros(Nm)  # h is zero vector
m = np.sign(np.add(np.random.rand(Nm) * 2, -1))


def boltzmann(J, H, Nt, Nm):
    all_state = [0 for i in range(pow(2, Nm))]
    for cc in range(pow(2, Nm)):
        b = [int(x) for x in bin(cc)[2:]]
        state = [0] * (Nm - len(b))
        state.extend(b)
        state = np.array(state)  # convert to nd.array
        state = np.subtract(2*state, 1)  # make [-1,1] from [0,1]
        all_state[cc] = state
    return np.array(all_state)


# main
myPcircuit = pbit.pcircuit(Nm, J, h, beta=I0, model="cpsl")
mcpsl = np.array(myPcircuit.saveSteps(Nt))
myPcircuit.setModel("ppsl", dt=d_t)
mppsl = np.array(myPcircuit.saveSteps(Nt))

decimal_boltz = [0 for i in range(2**Nm)]
histboltz = boltzmann(J, h, Nt, Nm)


# increment hist[val] by 1 where val is decimal of m
decimal_cpsl = [0 for i in range(Nt)]
decimal_ppsl = [0 for i in range(Nt)]


histcpsl = [0 for i in range(2**Nm)]
histppsl = [0 for i in range(2**Nm)]
Look = 2**np.arange(Nm)

for i in range(Nt):
    decimal_cpsl[i] = np.dot(Look, mcpsl[i, :])
    histcpsl[int(decimal_cpsl[i])] += 1
    decimal_ppsl[i] = np.dot(Look, mppsl[i, :])
    histppsl[int(decimal_ppsl[i])] += 1

# plot ppsl vs cpsl histograms
plt.hist([decimal_cpsl, decimal_ppsl])
plt.legend(["cpsl", "ppsl"])
plt.show(block=True)
# plot bar graphs with boltzmann
#plt.bar(np.arange(2**Nm), histcpsl)
#plt.bar(np.arange(2**Nm), histppsl)

#fig = px.bar([np.arange(2**Nm), histcpsl,histppsl])
# fig.show()
