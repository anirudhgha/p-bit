"""
gpu cpsl builder
"""
import pbit
from numba import jit, cuda, float64, njit
import numba
import numpy as np
import matplotlib.pyplot as plt


def predetermine_beta():
    return


"""
all_beta is an array of every beta value from i=0:Nt. beta will be pre-determined and sent to the gpu ahead of time
goal: parallelize the following function...
"""


@jit(float64[:](float64[:], float64, float64[:], float64[:], float64), nopython=True)
def cpsl_core_gpu(m, Nm, J, h, Nt):
    a = int(Nt)
    b = int(Nm)
    m_all = np.zeros(a*b)#[[0 for xx in range(a)] for yy in range(b)]
    m = np.ascontiguousarray(m)
    J = np.ascontiguousarray(J)
    for j in range(int(Nt)):
        for i in range(int(Nm)):
            arr1 = J[i * int(Nm):i * int(Nm) + int(Nm)]
            xx = -1 * 1 * (np.dot(m, arr1) + h[i])
            m[i] = np.sign(np.random.uniform(-1, 1) + np.tanh(xx))
        m_all[j*b:j*b+Nm] = m
    m_all[m_all < 0] = 0
    return m_all


# set parameters

J = np.ndarray.astype(np.array([[0, -2, -2],
                                [-2, 0, 1],
                                [-2, 1, 0]]), 'float64')
Jflat = np.ndarray.flatten(J)

h = np.ndarray.astype(np.array([2, -1, -1]), 'float64')
Nt = float64(100000)
Nm = float64(len(J))
m = np.sign(np.add(np.random.rand(int(Nm)) * 2, -1))
m_all = cpsl_core_gpu(m, Nm, Jflat, h, Nt)
m_all[m_all < 0] = 0
m_all = np.array(m_all).reshape((int(Nt), int(Nm)))
deci = pbit.convertToBase10(m_all)
histdeci = np.zeros(2 ** int(Nm))
for i in range(int(Nt)):
    histdeci[deci[i]] += 1

# plot
x1 = np.arange(2 ** Nm)
plt.bar(x1, histdeci)
plt.show()
