# p-bit Python Package
A comprehensive p-bit python package that simplifies execution of p-circuits. See www.purdue.edu/p-bit to learn about p-bits. 

* [Getting Started](#Getting-Started)
* [Methods](#Methods)
* [Variable Definitions](#Variable-Definitions)


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
* [saveSteps](#saveSteps)



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
    my_pcircuit.getWeights()
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

### saveSteps
    m = myPcircuit.saveSteps(Nt)

saveSteps returns an Nt * Nm matrix. Each row of m contains the state of each Nm p-bits as the p-circuit ran Nt times.

## Variable Definitions

* ##### Nm - Number of p-bits
* #### J - A 2D adjacency matrix of the network, also known as the weight matrix
* #### h - A 1D vector of biases
* #### beta - When running the p-circuit, the input to p-bits will be scaled up by beta before being fed into the activation function. 
* #### model - A parallel psl model ("ppsl") and the classical psl model are supported. 