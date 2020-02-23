import pbit

Nt = 10000

myp = pbit.pcircuit()
myp.load_image_as_ground_state("32x32.png")
m_all = myp.generate_samples(Nt, gpu=True)

pbit.live_heatmap(m_all, num_samples_to_plot=50, hold_time=0.2)
