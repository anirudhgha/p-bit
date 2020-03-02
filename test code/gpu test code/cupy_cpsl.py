"""
gpu cpsl builder
"""
import pbit
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cupy as cp


def predetermine_beta():
    return


"""
all_beta is an array of every beta value from i=0:Nt. beta will be pre-determined and sent to the gpu ahead of time
goal: parallelize the following function...
"""


def cpsl_core_gpu(samples, m, NmAndNt, J, h, rand_vals):

    gpu_NmAndNt = cp.asarray(NmAndNt)
    gpu_samples = cp.asarray(samples)
    gpu_J = cp.asarray(J)
    gpu_h = cp.asarray(h)
    gpu_m = cp.asarray(m)

    gpu_rand_vals = cp.asarray(rand_vals)

    for j in range(int(gpu_NmAndNt[1])):
        for i in range(int(gpu_NmAndNt[0])):
            gpu_J_slice = gpu_J[i * gpu_NmAndNt[0]: i * gpu_NmAndNt[0] + gpu_NmAndNt[0]]
            gpu_temp = -1 * gpu_NmAndNt[2] * (cp.add(cp.dot(gpu_m, gpu_J_slice), gpu_h[i]))
            gpu_m[i] = cp.sign(gpu_rand_vals[j*gpu_NmAndNt[0]+i] + cp.tanh(gpu_temp))
        gpu_samples[j*gpu_NmAndNt[0]:j*gpu_NmAndNt[0]+gpu_NmAndNt[0]] = gpu_m
    samples = cp.asnumpy(gpu_samples)
    return samples





# set parameters

J = np.ndarray.astype(np.array([[0, -2, -2],
                                [-2, 0, 1],
                                [-2, 1, 0]]), 'float64')
Jflat = np.ndarray.flatten(J)

h = np.ndarray.astype(np.array([2, -1, -1]), 'float64')
Nt = int(1e5)
Nm = int(len(J))
beta = int(1)
NmAndNt = np.array([Nm, Nt, beta], dtype=np.int)
m = np.sign(np.add(np.random.rand(int(Nm)) * 2, -1))
m_all = np.zeros(Nm*Nt)
start = timer()
rand_vals = np.random.uniform(-1, 1, Nt * Nm)

m_all = cpsl_core_gpu(m_all, m, NmAndNt, Jflat, h, rand_vals)
print("Samples generated in ", timer()-start, 's')

m_all[m_all < 0] = 0
m_all = np.array(m_all).reshape((int(Nt), int(Nm)))
deci = pbit.bi_arr2de(m_all)
histdeci = np.zeros(2 ** int(Nm))
for i in range(int(Nt)):
    histdeci[deci[i]] += 1

# plot
x1 = np.arange(2 ** Nm)
plt.bar(x1, histdeci)
plt.show()
