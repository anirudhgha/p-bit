"""
gpu cpsl builder
"""
import pbit
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cupy as cp
import torch
from numba import jit, cuda
import numba


def predetermine_beta():
    return


"""
all_beta is an array of every beta value from i=0:Nt. beta will be pre-determined and sent to the gpu ahead of time
goal: parallelize the following function...
"""


def cpsl_core_gpu(samples, gpu_samples, gpu_m, gpu_NmNtbeta, gpu_J, gpu_h, gpu_rand_vals):

    for j in range(int(gpu_NmNtbeta[1])):
        for i in range(int(gpu_NmNtbeta[0])):
            gpu_temp = -1 * gpu_NmNtbeta[2] * (cp.add(cp.dot(gpu_m, gpu_J[i, :]), gpu_h[i]))
            gpu_m[i] = cp.sign(gpu_rand_vals[j*gpu_NmNtbeta[0]+i] + cp.tanh(gpu_temp))
        gpu_samples[j,:] = gpu_m
    samples[:,:] = cp.asnumpy(gpu_samples)

def ppsl_core_gpu(samples, gpu_samples, gpu_m, gpu_NmNtbeta, gpu_J, gpu_h, gpu_rand_vals):
    for i in range(int(gpu_NmNtbeta[1])):
        x = -1 * cp.add(cp.dot(gpu_m, gpu_J), gpu_h)
        p = cp.exp(-1 * 1 / 6 * cp.exp(cp.multiply(-1* gpu_m, x)))
        gpu_m = cp.multiply(gpu_m, cp.sign(cp.add(p, -1 * gpu_rand_vals[int(i * gpu_NmNtbeta[0]): int(i * gpu_NmNtbeta[0] + gpu_NmNtbeta[0])])))
        gpu_samples[i, :] = gpu_m
    samples[:, :] = cp.asnumpy(gpu_samples)

print(torch.cuda.get_device_name(0))

myp = pbit.pcircuit()
myp.load('and')
# myp.load_random(100)
J,h = myp.getWeights()



Nt = int(1e3)
Nm = int(len(J))
beta = int(1)
NmNtbeta = np.array([Nm, Nt, beta], dtype=np.int)
m = np.sign(np.add(np.random.rand(int(Nm)) * 2, -1))
samples = np.zeros((Nt,Nm))
rand_vals = np.random.uniform(-1, 1, Nt * Nm)

gpu_NmNtbeta = cp.asarray(NmNtbeta)
gpu_samples = cp.asarray(samples)
gpu_J = cp.asarray(J)
gpu_h = cp.asarray(h)
gpu_m = cp.asarray(m)
gpu_rand_vals = cp.asarray(rand_vals)

start = timer()
ppsl_core_gpu(samples, gpu_samples, gpu_m, gpu_NmNtbeta, gpu_J, gpu_h, gpu_rand_vals)
print("Samples generated in ", timer()-start, 's')

samples[samples < 0] = 0
samples = np.array(samples).reshape((int(Nt), int(Nm)))
deci = pbit.bi_arr2de(samples)
histdeci = np.zeros(2 ** int(Nm))
for i in range(int(Nt)):
    histdeci[deci[i]] += 1



myp.setModel('ppsl')
m_all = myp.generate_samples(1e3, gpu=True, ret_base=10)
plt.hist(m_all)
plt.show()

#
#
# # plot
# x1 = np.arange(2 ** Nm)
# plt.bar(x1, histdeci)
# plt.show()
