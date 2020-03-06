import pbit
import numpy as np

"""
build a j matrix for some travelling salesman problem graph. 

designing a travelling salesman J:
Rule 1: 1 between pbits of same city
Rule 2: 1 between pbits of same order
Rule 3: negative distances as weights between rows (ex. all p-bits in city-1-row to all cities in city-3-row)
Rule 4: 0 connections from city_n-order_n to itself
J looks like:
                 city1-order1 city1-order2 ... city2-order1 ... cityN-orderN
    city1-order1 
    city1-order2
    .
    .
    .
    city2-order1
    .
    .
    .
    cityN-orderN
    
    See excel sheet (building_J_for_tsp in shark tank 2020 purdue onedrive)
"""
city_graph = [[0, 510, 480, 490],
              [510, 0, 240, 370],
              [480, 240, 0, 220],
              [490, 370, 220, 0]]

Nm = len(city_graph[0])
J = np.zeros((Nm ** 2, Nm ** 2))
J = np.asarray(J)
# Rule 3: negative distances from one city to another
for i in range(Nm):
    for j in range(Nm):
        J[j * Nm: j * Nm + Nm, i * Nm: i * Nm + Nm] = city_graph[j][i]  # set the weight from city i to city j

# rule 1 - 1 between pbits of same city
for i in range(Nm):
    J[i * Nm:i * Nm + Nm, i * Nm:i * Nm + Nm] = 1  # dif

# rule 2 - 1 between pbits of same order
for i in range(Nm**2):
    for j in range(Nm**2):
        if i == j % Nm or j == i % Nm:
            J[i, j] = 1

# rule 4: 0s on the diagonal
np.fill_diagonal(J, 0)

h = np.zeros(Nm**2)
myp = pbit.pcircuit(J=J, h=h, beta=0.005)
samples = myp.generate_samples(1e6, gpu=True)
pbit.live_heatmap(samples, 50)
