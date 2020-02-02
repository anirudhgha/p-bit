import numpy as np
import pbit
from numba import int64
import matplotlib.pyplot as plt
from timeit import default_timer as timer

start = timer()

J = [[0, -2, -2],
     [-2, 0, 1],
     [-2, 1, 0]]
h = [2, -1, -1]

Nm = 3
myp = pbit.pcircuit(J, h)
Nt = int64(1000000)
start = timer()
m = myp.runFor(Nt, gpu=True)
print("recieved samples: ", timer()-start)

decicpsl = pbit.convertToBase10(m)
histcpsl = np.zeros(2 ** Nm)
for i in range(Nt):
    histcpsl[decicpsl[i]] += 1

# plot
barWidth = 0.25
x1 = np.arange(2 ** Nm)

plt.bar(x1, histcpsl, width=barWidth, label='cpsl')
plt.legend()
plt.show()
