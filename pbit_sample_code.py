# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:50:03 2020

@author: alasg

Sample code that runs a simple p-circuit using the pbit package
"""
import pbit
from timeit import default_timer as timer

J = [[0, -0.4407, 0, 0, 0, 0],
              [-0.4407, 0, -0.4407, 0, 0, 0],
              [0, -0.4407, 0, 0, 0, 0],
              [0, 0, 0, 0, -0.4407, 0],
              [0, 0, 0, -0.4407, 0, -0.4407],
              [0, 0, 0, 0, -0.4407, 0]]
h = [2, -1, -1]

Nm = 20
h = [1, 1]
Nt = 100000
beta = 0

my_pcircuit = pbit.pcircuit(Nm, J, h)
my_pcircuit.buildRandomNetwork(Nm)
my_pcircuit.draw(labels=True)