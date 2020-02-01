# p-bit Python Package
A comprehensive p-bit python package that simplifies execution of p-circuits. See www.purdue.edu/p-bit to learn about p-bits. 

* [To Do](#To-Do)
* [Getting Started](#Getting-Started)
* [Methods](#Methods)
* [Variable Definitions](#Variable-Definitions)

## To Do
- [ ] add pbit.convertMatToCSV to convert matlab to python readable easily, maybe could 
just figure out how to export mat files as csv. 
- [x] incorporate annealing (constant, linear, geometric to begin)
- [x] extend class with draw function to draw the current p-circuit object
- [x] ~~cpsl and ppsl are not matching Boltzmann~~, error was with pbit.convertToBase10
- [ ] convertToBase10 needs to accept either 1D array or 2D array where each row is a sample, and convert every row to base 10 
and return a 1D array of base 10 values. 
- [ ] gpu speeds are slower than cpu speeds, need to optimize cpsl and ppsl functions to better use gpu 
- [ ] write function that samples every Nth timestep, and returns Nt/N samples though it runs for Nt timesteps
- [ ] introduce more post-processing functions for analysing data, maybe quantum functions
- [ ] long term goal: incorporate fpga commands into pbit package
- [ ] possible bug in draw function (_not_drawn() does not correctly keep track of drawn labels)

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
pcircuit.buildRandomNetwork(Nm, weight_type="float", J_max_weight=5, random_h=False, h_max_weight=5)
```
each value defaults to those shown above. They can each be provided by the user to customize the random p-circuit setup. 

### runFor
    m = myPcircuit.runFor(Nt)

runFor returns an Nt * Nm matrix. Each row of m contains the state of each Nm p-bits as the p-circuit ran Nt times.

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

