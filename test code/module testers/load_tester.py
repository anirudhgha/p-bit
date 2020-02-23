import pbit
import matplotlib.pyplot as plt

# Set up a skeletal p-circuit for which weights can then be changed
my_pcircuit = pbit.pcircuit()

# load in weights
my_pcircuit.load('not')

# generate samples with p-circuit
boltz = my_pcircuit.generate_samples(100000, ret='decimal')

# build histogram from samples
plt.hist(boltz)
plt.show()
