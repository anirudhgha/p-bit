import pbit
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(linewidth=300)

mat = scipy.io.loadmat('J_h_tsp.mat')
J, h = mat['Hc_temp'], mat['Hs_temp'] #J_h.mat uses J and h, J_h_tsp.mat uses Hc_temp and Hs_temp
# print(J)
# print(h)

print(len(J))
myp = pbit.pcircuit(J=J, h=h)
step_size = 0.4
beta = 0
for i in range(40):
    myp.setBeta(beta)
    samples = myp.generate_samples(1e5)
    beta += step_size

print('final beta: ', beta)
pbit.live_heatmap(samples, num_samples_to_plot=20, hold_time=0.2)
# final_sample = np.expand_dims(samples[-1, :]).reshape((3,3))
plt.imshow(samples[-1, :].reshape(4, 4))
plt.show()
