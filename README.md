# p-bit Python Package
A comprehensive p-bit python package that simplifies execution of p-circuits. See www.purdue.edu/p-bit to learn about p-bits. 

* [To Do](#To-Do)
* [Getting Started](#Getting-Started)
* [Variable Definitions](#Variable-Definitions)
* [Methods](#Methods)


## To Do
Pre-compilation works! Running an And gate on MATLAB for 1e7 samples takes ~46s, takes ~2s on laptop gtx1060. Python cpsl takes 283s to run a 24 p-bit network for 1e6 samples without gpu and 2.18s with gpu
- [ ] cupy may be easier to use, its a direct alternative to numpy. use pytorch to have your code actually run on the gpu. Pytorch has matmul and dot/ other functions already optimized for the gpu. need to transfer data to and from gpu memory etc...
- [ ] build quantum cpsl function which processes imaginary J/h components and returns a complex array of every state
- [ ] Optimize the GPU function, see https://numba.pydata.org/numba-doc/dev/cuda/memory.html to understand how to manage memory efficiently
- [ ] talk with fariah to incorporate new ppsl model
- [ ] talk with jan to incorporate learning into the package
- [ ] implement annealing properly, it doesn't work currently. modify increment annealing to send an entire beta (eventually it'll be sending batches of some 1e6 beta values and recalculating
the next batch at the end of that length)
- [ ] make J optimal for k-nearest neighbor setups by having Nm rows and each row only has k columns, one for each neighbor, may require modifying cpsl and ppsl functions
to only convolve with certain p-bits associated with each column
- [ ] NOTE: CAN HOLD OFF ON SHORS, it seems it may not be optimal for p-bits afterall, get shors algorithm incorporate somehow
- [x] ~~input image to act as ground state for p-circuit (load_image)~~
- [x] ~~Make a live update addition to generate_samples which lets you see color-flipping checkerboard of the p-bits for nearest neighbor, or color-flipping squares
arranged in a circle for any other topology, https://stackoverflow.com/questions/25385216/python-real-time-varying-heat-map-plotting~~
- [ ] update readme with new load() function
- [x] ~~make a sublcass of pbit which contains an assortment of J/h's already stored~~
- [ ] see how to import c++ functions into python to be able to communicate with aws fpga, check out http://www.swig.org/papers/PyTutorial98/PyTutorial98.pdf
- [ ] add shor's algorithm example to possible load function options
- [ ] overhaul load function to allow selecting preloading variable size networks, including using shors algorithm for dividing arbitrary numbers
- [ ] finish quantum gate copy of matlab code
- [ ] Option of running without returning every intermediate state (only final)
- [ ] Option to sample every Nth timestep, and returns Nt/N samples though it runs for Nt timesteps
- [x] ~~Need to integrate gpu cpsl into pbit module, currently works as a standalone unit~~ Integrated cpsl and ppsl gpu functions into pbit module
- [x] ~~add pbit.convertMatToCSV to convert matlab to python readable easily, maybe could 
just figure out how to export mat files as csv.~~ EXISTS IN SCIPY: See scipy example in shor's algorithm
- [x] ~~incorporate annealing (constant, linear, geometric to begin)~~ test annealing, doesn't seem to be making a difference
- [x] ~~extend class with draw function to draw the current p-circuit object~~ pcircuit.draw()
- [x] ~~cpsl and ppsl are not matching Boltzmann~~, error was with pbit.bi_arr2de
- [x] ~~bi_arr2de needs to accept either 1D array or 2D array where each row is a sample, and convert every row to base 10 
and return a 1D array of base 10 values.~~ 
- [x] ~~gpu speeds are slower than cpu speeds, need to optimize cpsl and ppsl functions to better use gpu~~ 1060 wrecks i7 cpu now (~60x)
- [ ] introduce more post-processing functions for analysing data, maybe quantum functions
- [ ] possible bug in draw function (_not_drawn() does not correctly keep track of drawn labels). Draw function isn't too reliable as is, may need to update it to make it robust. 

## Getting Started
P-circuits are quite simple to work with. The general use flow follows the same pattern for any use case.  

__Step 1)__ Initialize a p-circuit, either with J and h or without if you have yet to load some weights, following 
If you don't have a specific J and h...
```python 
my_pcircuit = pbit.pcircuit()
```
If you know your J and h
```python 
my_pcircuit = pbit.pcircuit(J = your_2D_J_Array, h = your_1D_h_vector)
```
The full initialization is provided below. For any parameter left empty, that parameter will be initialized with the provided value. 
```python
my_pcircuit = pbit.pcircuit(self, J=[[]], h=[], beta=1, Nm=0, model="cpsl", delta_t=None, start_beta=1, end_beta=2,
                 growth_factor=1.001, anneal="constant"):
``` 
See [Variable Definitions](#Variable-Definitions) for what each variable means. The basics of what constitutes a p-circuit is explained in detail at https://www.purdue.edu/p-bit/blog.html.  

__Step 2:__ Use generate samples to run the network for some number of timesteps and provide the state of the network at each of those timesteps. 
```python
samples = my_pcircuit.generate_samples(Nt=100000)
```  
See the full function definition [below](#generate_samples)  

__Step 3)__ Visualize the results. Histograms are a great way to see which states your network preferred over the course of its run. To plot a histogram, it is important to have generate_samples return a decimal value for each sample...
```python
samples = my_pcircuit.generate_samples(Nt=100000, ret_base='decimal')
plt.hist(samples)
plt.show()
```

Another visualization technique is to plot the heatmap of the resulting samples from generate_samples. A heatmap gives a color for each 1 in a sample and a different color to each 0 in a sample. See the [heatmap function](#live_heatmap). 

## Variable Definitions

* __Nm__ - Number of p-bits
* __J__ - A 2D adjacency matrix of the network, also known as the weight matrix
* __h__ - A 1D vector of biases
* __beta__ - When running the p-circuit, the input to p-bits will be scaled up by beta before being fed into the activation function. This controls the 'temperature' of the network.
* __model__ - A parallel psl model ("ppsl") and the classical psl model("cpsl") are supported. The model defines which set of algorithms are used to update the p-bits. 
* __Nt__ - number of timesteps a network is to run for (i.e number of samples to generate)
* __delta_t__ - percent of p-bits to update per timestep (i.e d_t=0.3 means each p-bit has a 30% chance of updating (could flip or not flip) per timestep). Only applicable to the ppsl model, since cpsl updates each p-bit sequentially. 


## Methods
* [set variable state](#setWeights)
* [get variable state](#getWeights)
* [reset](#reset)
* [load_random](#load_random)
* [generate_samples](#generate_samples)
* [getBotlzmann](#getBoltzmann)
* [draw](#draw)
* [binary array to decimal](#bi_arr2de)
* [animated heatmap](#live_heatmap)
* [load image as ground state](#load_image_as_ground_state)


### setWeights
    my_pcircuit.setWeights(J,h)
Set the weights J and h
### setSize
    my_pcircuit.setSize(Nm)
Set Nm
### setBeta
    my_pcircuit.setBeta(beta)
### setModel
    my_pcircuit.setModel("ppsl") or my_pcircuit.setModel("cpsl")
### setState
    my_pcircuit.setState(m)
### getWeights
    J,h = my_pcircuit.getWeights()
returns J and h
### getSize
    my_pcircuit.getSize()
returns Nm
### getBeta
    my_pcircuit.getBeta()
returns beta
### getModel
    my_pcircuit.getModel()
returns "cpsl" or "ppsl"
### getState
    my_pcircuit.getState()
returns m
### reset
    my_pcircuit.reset()
resets the internal m state, randomizing each p-bit in the Nm-size p-circuit to +-1.

### load_random
```python
my_pcircuit.load_random(Nm, weight_type="float", J_max_weight=5, random_h=False, h_max_weight=5)
```
each value defaults to those shown above. They can each be provided by the user to customize the random p-circuit setup. 

use case, creating a randomly connected 20 p-bit network:
```python
my_pcircuit = pbit.pcircuit()
my_pcircuit.load_random(20)
```

note, all load_random networks are fully connected, no weight is set to 0.

### load_image_as_ground_state
```python
    myp.load_image_as_ground_state("desired_ground_state.png")
```
Sets the image desired_ground_state.png as the ground state for the p-circuit. There exist a set (infinite technically) of weights for which the global minimum is the provided pattern. This function calculates such a weight matrix from the image. Any image will be grayscaled, and converted to black and white before setting the weight. 


### generate_samples
    m = myPcircuit.generate_samples(Nt, gpu=False)

generate_samples returns an Nt * Nm matrix (ex. Nm=3, [0 0 1]). Each row of m contains the state of each Nm p-bits as the p-circuit ran Nt times.
Setting gpu to True will make use of an available Nvidia gpu to accelerate execution. 

generate_samples will execute following the model provided when the pcircuit was constructed or that was set using setModel(). 

### getBoltzmann
    
    histboltz = myPcircuit.getBoltzmann()

myPcircuit contains the J, h, and Nm of your current pcircuit. These parameters are used to find the Boltzmann statistics of which states myPcircuit should occupy. 

Returns a normalized-to-1 np array of length 2^Nm with the exact probability of occupation for each state. 

Note, for Nm > 15, it may take a very long time for getBoltzmann() to execute. It must calculate a probability for each of 2^Nm states. 

Ex.   
>  histboltz = myPcircuit.getBoltzmann()  
>  print(histboltz)

prints the following for Nm = 3:  

> [0.04573869 0.14532606 0.22552337 0.08341188 0.08341188 0.22552337 0.14532606 0.04573869]
### draw

    my_pcircuit.draw(labels=True)

Draws the pcircuit. Dots are pbits and lines are weights. Setting labels to False can allow faster drawing and reduce clutter for 
larger networks. 

### Out of Class Methods
#### bi_arr2de
    b = bi_arr2de(a, inputBase=2)
A handy function for converting p-bit output from generate_samples into decimal values that can be plotted in a histogram. Takes as input a list or array of binary values ex. [1,0,0,1,0] and returns a decimal integer ex. 18.

possible use case:
```python
m = pcircuit.generate_samples(Nt)
for i in range(Nt):
    decimal_states[i] = pbit.bi_arr2de(m[i, :])
```
#### live_heatmap
```python
pbit.live_heatmap(samples, num_samples_to_plot=Nt, hold_time=Nm)
```

Plot an animated sequence of heatmaps that shows how the network changed over time. The input takes a binary (0,1) array with some Nt rows and some Nm columns. The function finds the most 'square' representation of Nm p-bits and aranges the heatmap such that each pixel is a p-bit. When the function is run, black is 0 and gold is 1.  

Example use case: 
```python
myp = pbit.pcircuit()
myp.load_image_as_ground_state("32x32.png")
samples = myp.generate_samples(Nt=1000)
pbit.live_heatmap(samples, num_samples_to_plot=50, hold_time=0.2)
```

#### errorMSE
```python
error = pbit.errorMSE(arr1, arr2)
```
finds the mean squared error between two 1D arrays. 

