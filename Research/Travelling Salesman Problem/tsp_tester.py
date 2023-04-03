import pbit
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(linewidth=300)

city_graph = np.array([[0, 510, 480, 490],
                      [510, 0, 240, 370],
                      [480, 240, 0, 220],
                      [490, 370, 220, 0]])
# city_graph = np.array([[0, 2, 3],
#                        [2, 0, 1],
#                        [3, 1, 0]])
city_graph *= -1

myp = pbit.pcircuit()
myp.load('tsp', city_graph, tsp_modifier=0.33)

J, h = myp.getWeights()
print(J)
step_size = 0.1
beta = 0
for i in range(100):
    myp.setBeta(beta)
    samples = myp.generate_samples(1e4)
    beta += step_size
print('ending beta: ', myp.getBeta())
samples = myp.generate_samples(1e5)

# pbit.live_heatmap(samples, num_samples_to_plot=20, hold_time=0.2)
# final_sample = np.expand_dims(samples[-1, :]).reshape((3,3))
plt.imshow(samples[-1, :].reshape((len(city_graph), len(city_graph[0]))))
plt.show()
#sol: 2 3 4 1
