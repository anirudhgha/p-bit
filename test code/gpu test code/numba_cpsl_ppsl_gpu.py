"""
Using numba to both compile at time using JIT and run code on cuda-enabled gpu. This requires mapping out which
blocks and threads do what.
"""
import pbit
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cupy as cp
from numba import jit, cuda, float64, int64
import torch
import numba


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device is cuda:0
print(torch.cuda.get_device_name(0))

@cuda.jit('void(float64[:, :], float64[:, :], float64[:], float64[:], float64[:, :], float64[:], float64[:], float64)', device=True)
def ppsl_core_gpu(samples, gpu_samples, gpu_m, gpu_NmNtbeta, gpu_J, gpu_h, gpu_rand_vals, dt):
    for i in range(int(gpu_NmNtbeta[1])):
        x = -1 * cp.add(cp.dot(gpu_m, gpu_J), gpu_h)
        p = cp.exp(-1 * dt * cp.exp(cp.multiply(gpu_m, x)))
        gpu_m = cp.multiply(gpu_m, cp.sign(cp.add(p, -1 * gpu_rand_vals[int(i * gpu_NmNtbeta[0]): int(i * gpu_NmNtbeta[0] + gpu_NmNtbeta[0])])))
        gpu_samples[i, :] = gpu_m
    samples[:, :] = cp.asnumpy(gpu_samples)


myp = pbit.pcircuit()
myp.load_random(10000, J_max_weight=1)
# myp.load('and')
J,h = myp.getWeights()



Nt = int(1e3)
Nm = int(len(J))
beta = int(1)
NmNtbeta = np.array([Nm, Nt, beta], dtype=int64)
m = np.sign(np.add(np.random.rand(int(Nm)) * 2, -1))
samples = np.zeros((Nt,Nm))
rand_vals = np.random.uniform(-1, 1, Nt * Nm)
gpu_samples = np.zeros((Nt,Nm))


start = timer()
ppsl_core_gpu(samples, gpu_samples, m, NmNtbeta, J, h, rand_vals)
print("Samples generated in ", timer()-start, 's')
#
# samples[samples < 0] = 0
# samples = np.array(samples).reshape((int(Nt), int(Nm)))
# deci = pbit.bi_arr2de(samples)
# histdeci = np.zeros(2 ** int(Nm))
# for i in range(int(Nt)):
#     histdeci[deci[i]] += 1
#
# # plot
# x1 = np.arange(2 ** Nm)
# plt.bar(x1, histdeci)
# plt.show()
