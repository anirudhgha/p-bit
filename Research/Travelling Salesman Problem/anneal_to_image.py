import pbit
import matplotlib.pyplot as plt
import numpy as np

myp = pbit.pcircuit()
myp.load_image_as_ground_state('ground_state_shark_tank.png')

all_samples = np.zeros((10000, myp.getSize()))
beta = 0
for i in range(10):
    myp.setBeta(beta)
    samples = myp.generate_samples(1e3)
    all_samples[i*1000:i*1000+1000, :] = samples[:]
    beta += 0.1

print(all_samples.shape)
pbit.live_heatmap(all_samples, num_samples_to_plot=30, hold_time=0.5)
plt.imshow(samples[-1, :].reshape(4, 4))
plt.show()