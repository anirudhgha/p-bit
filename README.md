# p-bit Python Package
A comprehensive p-bit python package that simplifies execution of p-circuits. See www.purdue.edu/p-bit to learn about p-bits. 

* [To Do](#To-Do)
* [Getting Started](#Getting-Started)
* [Methods](#Methods)
* [Variable Definitions](#Variable-Definitions)

## To Do
GPU works! Running an And gate on MATLAB for 1e7 samples takes ~46s, takes ~2s on laptop gtx1060
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
- [x] ~~cpsl and ppsl are not matching Boltzmann~~, error was with pbit.convertToBase10
- [x] ~~convertToBase10 needs to accept either 1D array or 2D array where each row is a sample, and convert every row to base 10 
and return a 1D array of base 10 values.~~ 
- [x] ~~gpu speeds are slower than cpu speeds, need to optimize cpsl and ppsl functions to better use gpu~~ 1060 wrecks i7 cpu now (~60x)
- [ ] introduce more post-processing functions for analysing data, maybe quantum functions
- [ ] possible bug in draw function (_not_drawn() does not correctly keep track of drawn labels). Draw function isn't too reliable as is, may need to update it to make it robust. 
## Getting Started
To get started, initialize a p-circuit with the necessary parameters following:
```python
 my_pcircuit = pbit.pcircuit(Nm=0, J=[[]], h=[], beta=1, model="cpsl", delta_t=0.01):
``` 
* Nm - Number of p-bits
* J - A 2D adjacency matrix of the network, also known as the weight matrix
* h - A 1D vector of biases
* beta - When running the p-circuit, the input to p-bits will be scaled up by beta before being fed into the activation function. 
* model - A parallel psl model ("ppsl") and the classical psl model (default "cpsl") are supported. 
  
## Methods
* [set variable state](#setWeights)
* [get variable state](#getWeights)
* [reset](#reset)
* [buildRandomNetwork](#buildRandomNetwork)
* [runFor](#runFor)
* [getBotlzmann](#getBoltzmann)
* [draw](#draw)
* [Out of class methods](#Out-of-Class-Methods)


### setWeights
    my_pcircuit.setWeights(J,h)
### setSize
    my_pcircuit.setSize(Nm)
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
resets the internal m state, randomizing each p-bit state to +-1.

### buildRandomNetwork
```python
my_pcircuit.buildRandomNetwork(Nm, weight_type="float", J_max_weight=5, random_h=False, h_max_weight=5)
```
each value defaults to those shown above. They can each be provided by the user to customize the random p-circuit setup. 

### runFor
    m = myPcircuit.runFor(Nt, gpu=False)

runFor returns an Nt * Nm matrix (ex. Nm=3, [0 0 1]). Each row of m contains the state of each Nm p-bits as the p-circuit ran Nt times.
Setting gpu to True will make use of an available Nvidia gpu to accelerate execution. 

runFor will execute following the model provided when the pcircuit was constructed or that was set using setModel(). 

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
#### convertToBase10
    b = convertToBase10(a, inputBase=2)
A handy function for converting p-bit output from runFor into decimal values that can be plotted in a histogram. Takes as input a list or array of binary values ex. [1,0,0,1,0] and returns a decimal integer ex. 18.

possible use case:
```python
m = pcircuit.runFor(Nt)
for i in range(Nt):
    decimal_states[i] = pbit.convertToBase10(m[i, :])
```

## Variable Definitions

* ##### Nm - Number of p-bits
* ##### J - A 2D adjacency matrix of the network, also known as the weight matrix
* ##### h - A 1D vector of biases
* ##### beta - When running the p-circuit, the input to p-bits will be scaled up by beta before being fed into the activation function. 
* ##### model - A parallel psl model ("ppsl") and the classical psl model are supported. 

