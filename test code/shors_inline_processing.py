import pbit
import scipy.io
import random
import numpy as np
import cmath
import matplotlib.pyplot as plt
from timeit import default_timer as timer


mat = scipy.io.loadmat('JH_MATLAB.mat')
J, h, Jr, hr, es, Prefac = mat['J'], mat['h'], mat['Jr'], mat['hr'], mat['es'], mat['Prefac']
betaPSL = 1
Nm = len(h)
Nt = 100000
numTotal_qbits = 5
m = np.sign(np.add(np.random.rand(Nm) * 2, -1))
m = m.astype(complex)
Probs = np.zeros(2 ** numTotal_qbits, dtype=np.complex_)
m_all = np.zeros((Nt, Nm), dtype=np.complex_)
start = timer()
print('Post-processing...')

for j in range(Nt):
    for i in np.random.permutation(Nm):
        xx = -1 * betaPSL * (np.dot(m, Jr[:, i]) + hr[i])
        m[i] = np.sign(random.uniform(-1, 1) + np.tanh(xx))
    E = es + np.multiply(0.5, np.dot(np.dot(m, J), m)) + np.dot(np.expand_dims(m, 0), h)
    phi = cmath.phase(np.exp(-1 * betaPSL * E))
    X2 = pbit.bi_arr2de(m[[19, 16, 13, 10, 23]])
    Probs[X2] = Probs[X2] + np.exp(1j * phi)

time_collected = timer()-start
print('Q Samples collected in ', time_collected, 's')

# now we have results, just need to plot
Probs2 = np.multiply(Probs, Prefac)[0]
PSL = np.divide(Probs2, np.sqrt(np.sum(np.square(np.abs(Probs2)))))[1:31:2]
XX = 2*np.square(np.abs(PSL))
print('Complete...')
plt.stem(np.arange(15), XX, use_line_collection=True)
plt.show()



