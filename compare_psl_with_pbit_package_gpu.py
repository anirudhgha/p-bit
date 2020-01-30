from numba import jit, cuda, njit
import numpy as np
from timeit import default_timer as timer
import pbit

# normal function to run on cpu


J = [[1, 2], [3, 4]]
Nm = 2
h = [1, 1]
Nt = 10000
beta = 1

my_pcircuit = pbit.pcircuit(Nm, J, h, beta=beta, model="ppsl")
start = timer()
mcpsl = my_pcircuit.runFor(Nt, gpu=False)
print("without GPU:", timer()-start)

start = timer()
mcpsl = my_pcircuit.runFor(Nt, gpu=True)
print("with GPU:", timer()-start)

print("\n\nCurrently un-optimized gpu code, working but needs work. ")
