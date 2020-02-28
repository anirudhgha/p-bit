# DESCRIPTION:
#
# Executes shor's algorithm using the pbit module. We should should see peaks at 0 and 8, which are ideally 0.5 high.
# This method follows a post-processing approach, in which all the samples are first generated by the gpu, providing the
# magnitude for the qubits we are trying to model. The phase of each qubit is then calculated in post-processing via the
# cmath.phase function. Depending on the phase of each p-circuit state, the state may interfere constructively or
# destructively with the existing sum. This interference eventually reveals the correct peaks.

# Authors: Python code: Anirudh Ghantasala, MATLAB code and concepts, Shuvro Chowdhury


import scipy.io
import numpy as np
import pbit
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from numba import jit, complex128, int64, float64, cuda


@jit((complex128[:], complex128[:], int64, int64, complex128[:, :], complex128[:], complex128[:], complex128, int64[:],
      float64[:]), nopython=True)
def post_processing(samples, mm, Nm, Nt, J, h, probs, es, a, Look):
    betaPSL = 1
    indices = [23, 10, 13, 16, 19]  # [19, 16, 13, 10, 23]
    for i in range(Nt):
        mm[:] = samples[i * Nm:i * Nm + Nm]
        E = es + np.multiply(0.5, np.dot(np.dot(mm, J), mm)) + np.dot(mm, h)
        phi = np.angle(np.exp(-1 * betaPSL * E))
        a = [np.real(mm[ii]) for ii in indices]  # mm[[19, 16, 13, 10, 23]]
        for ii in range(5):
            if a[ii] < 0:
                a[ii] = 0
        arr = np.array(a)
        X2 = int(np.round(np.dot(arr, Look)))
        probs[X2] = probs[X2] + np.exp(1j * phi)


# these values taken from matlab file python_helper_isingshor.m which uses shuvro's IsingGraphShor2.m
mat = scipy.io.loadmat('JH_MATLAB.mat')
J, h, Jr, hr, es, Prefac = mat['J'], mat['h'], mat['Jr'], mat['hr'], mat['es'], mat['Prefac']
betaPSL = 1
Nm = len(h)
Nt = 1e6
Nt = int(Nt)
numTotal_qbits = 5

print('Building Network...')
myp = pbit.pcircuit(Jr, hr, model='cpsl')
print('Generating samples...')
start = timer()
m_all = myp.generate_samples(Nt, gpu=True)
time_collected = timer() - start
print('Samples collected in ', time_collected, 's')
m_all[m_all == 0] = -1
probs = np.ascontiguousarray(np.zeros(2 ** numTotal_qbits, dtype=np.complex_))
E = complex(0, 0)
m = np.sign(np.add(np.random.rand(Nm) * 2, -1))

print('Post-processing...')
gpu_es = es[0, 0]
gpu_look = np.ascontiguousarray(2 ** np.arange(5, dtype=np.float64))
gpu_mm = np.ascontiguousarray(np.ones(Nm, dtype=np.complex_))
gpu_J = np.ascontiguousarray(J)
gpu_m_all = np.ascontiguousarray(np.array(np.ndarray.flatten(m_all), dtype=np.complex_))
gpu_h = np.ascontiguousarray(np.array(np.ndarray.flatten(h), dtype=np.complex_))
gpu_looker = np.ascontiguousarray(np.ones(5, dtype=np.int64))
post_processing(gpu_m_all, gpu_mm, Nm, Nt, gpu_J, gpu_h, probs, gpu_es, gpu_looker, gpu_look)

# for i in range(int(Nt)):
#     m[:] = m_all[i, :]
#     E = es + np.multiply(0.5, np.dot(np.dot(m, J), m)) + np.dot(m, h)
#     phi = np.angle(np.exp(-1 * betaPSL * E))
#     X2 = pbit.bi_arr2de(m[[19, 16, 13, 10, 23]])
#     probs[X2] = probs[X2] + np.exp(1j * phi)

print('Processing complete in ', timer() - time_collected, 's')

# now we have results, just need to plot
probs2 = np.multiply(probs, Prefac)[0]
PSL = np.divide(probs2, np.sqrt(np.sum(np.square(np.abs(probs2)))))[1:31:2]
XX = 2 * np.square(np.abs(PSL))
print('Complete...')
plt.stem(np.arange(15), XX, use_line_collection=True)
plt.show()
