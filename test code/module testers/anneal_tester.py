import pbit
import matplotlib.pyplot as plt
import numpy as np


Nt = 10
myp = pbit.pcircuit()
myp.load_image_as_ground_state('ground_state.png')

print('size: ', myp.getSize(), 'p-bits')
all_samples = np.zeros((10*Nt, myp.getSize()))
beta = 0
for i in range(10):
    print(i)
    myp.setBeta(beta)
    samples = myp.generate_samples(Nt, gpu=True)
    all_samples[i*Nt:i*Nt+Nt, :] = samples[:]
    beta += 0.1


print(all_samples.shape)
pbit.live_heatmap(all_samples, num_samples_to_plot=20, hold_time=0.2)
plt.imshow(samples[-1, :].reshape(4, 4))
plt.show()