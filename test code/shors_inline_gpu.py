import pbit
import scipy.io
import random
import numpy as np
import cmath
import matplotlib.pyplot as plt
from numba import jit, float64, int64, complex128
import numba.cuda as cuda


@jit(complex128[:](complex128[:], int64, complex128[:], complex128[:], float64, int64, float64[:]), nopython=True)
def qcpsl_gpu(m, Nm, J, h, beta, Nt, rand_vals):
    m_all = np.zeros((Nt, Nm), dtype=np.complex_)
    cuda.to_device(m_all)
    m = np.ascontiguousarray(m)
    m = m.astype(complex)
    J = np.ascontiguousarray(J)
    for j in range(int(Nt)):
        for i in range(int(Nm)):
            xx = -1 * beta * (np.dot(m, J[i * Nm:i * Nm + Nm]) + h[i])
            m[i] = np.sign(rand_vals[j * Nm + i] + np.tanh(xx))
        m_all[j * Nm:j * Nm + Nm] = m
        E = es + np.multiply(0.5, np.dot(np.dot(m, J[i * Nm:i * Nm + Nm]), m)) + np.dot(np.expand_dims(m, 0), h)
        phi = cmath.phase(np.exp(-1 * betaPSL * E))
        # X2 = pbit.bi_arr2de(m[[19, 16, 13, 10, 23]])
        a = m[[19, 16, 13, 10, 23]]
        a[a < 0] = 0
        arr = np.flip(np.array(a))
        length = len(arr[0]) if arr.ndim == 2 else len(arr)
        Look = 2 ** np.arange(length)
        X2 = np.round(np.dot(arr, Look)).astype("int")
        Probs[X2] = Probs[X2] + np.exp(1j * phi)


    return m_all


mat = scipy.io.loadmat('JH_MATLAB.mat')
J, h, Jr, hr, es, Prefac = mat['J'], mat['h'], mat['Jr'], mat['hr'], mat['es'], mat['Prefac']
betaPSL = 1
Nm = len(h)
Nt = 1000
numTotal_qbits = 5
m = np.sign(np.add(np.random.rand(Nm) * 2, -1))
m = m.astype(complex)
Probs = np.zeros(2 ** numTotal_qbits, dtype=np.complex_)
m_all = np.zeros((Nt, Nm), dtype=np.complex_)


for j in range(Nt):
    for i in np.random.permutation(Nm):
        xx = -1 * betaPSL * (np.dot(m, J[:, i]) + h[i])
        m[i] = np.sign(random.uniform(-1, 1) + np.tanh(xx))



    m_all[j, :] = m



    m_all[m_all < 0] = 0
    m_all = np.reshape(m_all, (Nt, Nm))

# now we have results, just need to plot
Probs2 = np.multiply(Probs, Prefac)[0]
PSL = np.divide(Probs2, np.sqrt(np.sum(np.square(np.abs(Probs2)))))[1:31:2]
XX = 2*np.square(np.abs(PSL))
print('Complete...')
plt.stem(np.arange(15), XX, use_line_collection=True)
plt.show()



