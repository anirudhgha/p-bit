# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:50:03 2020

@author: alasg

Sample code that runs a simple p-circuit using the pbit package
"""
import pbit
from timeit import default_timer as timer

J = [[1, 2], [3, 4]]
Nm = 2
h = [1, 1]
Nt = 100000
beta = 0


pcircuit = pbit.pcircuit()
pcircuit.buildRandomNetwork(Nm=3, J_max_weight=10)

J, h = pcircuit.getWeights()

decimal_states = [0 for i in range(Nt)]

m = pcircuit.saveSteps(Nt)
for i in range(Nt):
    decimal_states[i] = pbit.convertToBase10(m[i])
