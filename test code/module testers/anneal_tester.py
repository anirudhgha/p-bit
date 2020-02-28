import pbit
import matplotlib.pyplot as plt
import numpy as np


myp = pbit.pcircuit(anneal='linear', start_beta=1, end_beta=100)
myp.load_image_as_ground_state("32x32.png")
samples = myp.generate_samples(1000)
pbit.live_heatmap(samples, num_samples_to_plot=100, hold_time=0.1)


