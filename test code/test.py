import pbit


myp = pbit.pcircuit()
myp.load('and')

samples = myp.generate_samples(1000, gpu=True, ret_base='binary')
pbit.live_heatmap(samples, num_samples_to_plot=100, hold_time=0.2)

