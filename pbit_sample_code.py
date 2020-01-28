# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:50:03 2020

@author: alasg

Sample code that runs a simple p-circuit using the pbit package
"""
import pbit

J = [[1, 2], [3, 4]]
Nm = 2
h = [1, 1]
Nt = 100
beta = 0


pcircuit = pbit.pcircuit()
pcircuit.buildRandomNetwork(Nm=3, J_max_weight=10)

J,h = pcircuit.getWeights()
print(J)
print(h)

m = pcircuit.saveSteps(Nt)

Look = 2**np.arange(Nm)



