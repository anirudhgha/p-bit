"""
Simple sample code running an and gate p-circuit
"""
import pbit
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# build p-circuit
myp = pbit.pcircuit()
myp.load_random(10000)

# run p-circuit
# myp.draw()
start = timer()
samples = myp.generate_samples(1e3, gpu=True)
print("Generated samples in ", timer() - start, 's.')

# plot
# plt.hist(samples)
# plt.show()
