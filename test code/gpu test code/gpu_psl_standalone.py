"""
gpu cpsl builder
"""
import pbit
from numba import jit, cuda, float64, njit, int32
import numba
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def predetermine_beta():
    return

"""
all_beta is an array of every beta value from i=0:Nt. beta will be pre-determined and sent to the gpu ahead of time
goal: parallelize the following function...
"""


@jit(float64[:](float64[:], int32, float64[:], float64[:], int32, float64[:]), nopython=True)
def cpsl_core_gpu(m, Nm, J, h, Nt, rand_vals):
    m_all = np.zeros(Nt * Nm)  # [[0 for xx in range(a)] for yy in range(b)]
    m = np.ascontiguousarray(m)
    J = np.ascontiguousarray(J)
    for j in range(int(Nt)):
        for i in range(int(Nm)):
            xx = -1 * 1 * (np.dot(m,  J[i * Nm:i * Nm + Nm]) + h[i])
            m[i] = np.sign(rand_vals[j*Nm+i] + np.tanh(xx))
        m_all[j * Nm:j * Nm + Nm] = m
    m_all[m_all < 0] = 0
    # m_all = np.reshape(m_all, (Nt, Nm))
    return m_all

@jit(float64[:](float64[:], int32, float64[:], float64[:], int32, float64), nopython=True)
def ppsl_core_gpu(m, Nm, J, h, Nt, dt):
    a = int(Nt)
    b = int(Nm)
    m_all = np.zeros(a * b)
    m = np.ascontiguousarray(m)
    J = np.ascontiguousarray(J)
    J = np.reshape(J, (b, b))
    for i in range(a):
        x = np.multiply(np.add(np.dot(J, m), h), -1 * 1)
        p = np.exp(-1 * dt * np.exp(np.multiply(-1 * m, x)))
        m = np.multiply(m, np.sign(np.subtract(p, np.random.rand(b))))
        m_all[i * b:i * b + b] = m
    m_all[m_all < 0] = 0
    return m_all


# set parameters
J = np.ndarray.astype(np.array([[0, -2, -2],
                                [-2, 0, 1],
                                [-2, 1, 0]]), 'float64')
Jflat = np.ndarray.flatten(J)
h = np.ndarray.astype(np.array([2, -1, -1]), 'float64')
Nt_cpsl = 10000000
Nt_ppsl = 1
Nm = len(J)
m = np.sign(np.add(np.random.rand(Nm) * 2, -1))
dt = float64(1/9)
rand_vals_cpsl = np.random.uniform(-1, 1, Nt_cpsl*Nm)

start = timer()
# cpsl
mcpsl = cpsl_core_gpu(m, Nm, Jflat, h, Nt_cpsl, rand_vals_cpsl)
print("getting all m back:", timer() - start)

mcpsl = np.reshape(mcpsl, (Nt_cpsl, Nm))
decicpsl = pbit.convertToBase10(mcpsl)
histcpsl = np.zeros(2 ** Nm)
for i in range(Nt_cpsl):
    histcpsl[decicpsl[i]] += 1

# # ppsl
# mppsl = ppsl_core_gpu(m, Nm, Jflat, h, Nt_ppsl, dt)
# mppsl = np.reshape(mppsl, (Nt_ppsl, Nm))
# decippsl = pbit.convertToBase10(mppsl)
# histppsl = np.zeros(2 ** int(Nm))
# for i in range(int(Nt_ppsl)):
#     histppsl[decippsl[i]] += 1


# plot
barWidth = 0.25
x1 = np.arange(2 ** Nm)
x2 = np.array([x + barWidth for x in x1])

plt.bar(x1, histcpsl, width=barWidth, label='cpsl')
# plt.bar(x2, histppsl, width=barWidth, label='ppsl')
plt.legend()
plt.show()
