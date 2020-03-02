"""
gpu cpsl builder
"""
import pbit
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device is cuda:0
print(torch.cuda.get_device_name(0))

# Initialize parameters and send each to cuda device
# J = torch.FloatTensor([[0, -2, -2],
#                       [-2, 0, 1],
#                       [-2, 1, 0]])
# h = torch.FloatTensor([2, -1, -1])

myp = pbit.pcircuit(model='ppsl', delta_t=1/6)
# myp.load('and')
myp.load_random(10000, J_max_weight=1)
J, h = myp.getWeights()
J = torch.FloatTensor(J)
h = torch.FloatTensor(h)

# samples_correct = myp.generate_samples(1e3, gpu=True, ret_base='decimal')

Nt, Nm, beta = int(1e3), int(len(h)), 1
NmNtBeta = torch.IntTensor([Nm, Nt, beta])
m = torch.FloatTensor(np.sign(np.add(np.random.rand(Nm) * 2, -1)))
gpu_samples = torch.zeros(Nt, Nm)
rand_vals = torch.FloatTensor(2 * torch.rand(Nt * Nm) - 1)

gpu_samples.cuda(device)
gpu_J = J.cuda(device)
gpu_h = h.cuda(device)
gpu_NmNtBeta = NmNtBeta.cuda(device)
gpu_rand_vals = rand_vals.cuda(device)

gpu_m = m.cuda(device)
start = timer()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CPSL GPU CODE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# for j in range(gpu_NmNtBeta[1]):
#     for i in range(gpu_NmNtBeta[0]):
#         gpu_temp = -1 * gpu_NmNtBeta[2] * (torch.dot(gpu_m, gpu_J[:, i]) + gpu_h[i])
#         gpu_m[i] = torch.sign(gpu_rand_vals[j * gpu_NmNtBeta[0] + i] + torch.tanh(gpu_temp))
#     gpu_samples[j, :] = gpu_m
# samples = gpu_samples.cpu()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PPSL GPU CODE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
gpu_m = gpu_m.unsqueeze(0)
# print(gpu_J)
# print(gpu_h)
# print(gpu_m)
# print(gpu_rand_vals)
# print('gpu rand vals size: ', gpu_rand_vals.size())
for i in range(gpu_NmNtBeta[1]):
    x = -1*torch.add(torch.mm(gpu_m, gpu_J), gpu_h)
    p = torch.exp(-1 * 1/6 * torch.exp(torch.mul(gpu_m, x)))
    gpu_m = torch.mul(gpu_m, torch.sign(torch.add(p, -1*gpu_rand_vals[i*gpu_NmNtBeta[0]: i*gpu_NmNtBeta[0] + gpu_NmNtBeta[0]])))
    gpu_samples[i, :] = gpu_m
samples = gpu_samples.cpu()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("Samples generated in ", timer() - start, 's')

# samples[samples < 0] = 0
# deci = pbit.bi_arr2de(samples)
# histdeci = np.zeros(2 ** Nm)
# for i in range(Nt):
#     histdeci[deci[i]] += 1
#
# histdeci_fast = np.zeros(2 ** Nm)
# for i in range(Nt):
#     histdeci_fast[samples_correct[i]] += 1
#
# # plot
# x1 = np.arange(2 ** Nm)
# x2 = np.array([x + 0.25 for x in x1])
# plt.bar(x1, histdeci, width=0.25, edgecolor='white', label='gpu_ppsl')
# plt.bar(x2, histdeci_fast, width=0.25, edgecolor='white', label='jit_ppsl')
# plt.legend()
# plt.show()
