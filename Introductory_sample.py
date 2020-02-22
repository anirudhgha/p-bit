"""
Simple sample code running an and gate p-circuit
"""
import pbit
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# build p-circuit
J = [[0, -2, -2],
     [-2, 0, 1],
     [-2, 1, 0]]
h = [2, -1, -1]

myp = pbit.pcircuit(J, h)  # or, use myp.load('and')

# run p-circuit
myp.draw()
start = timer()
samples = myp.generate_samples(100000, gpu=True, ret_base='decimal')
print("Generated samples in ", timer() - start, 's.')

# plot
plt.hist(samples)
plt.show()
