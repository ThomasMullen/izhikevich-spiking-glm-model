# Fit Generalised Linear Model to spiking data

A summer project where I wanted to fit a generalised linear model to spiking data and systematically evaluate the performance of a recurrent linear-nonlinear Poisson model. To test the performance of the model data is generated from the Izhikevich neuron (a reduced model of the Hodge-Huxley model).

## Generating the data
- Generating the I model - predefined behaviours based on reset parameters
Dictionary of `behaviour_types` such as:
* tonic spiking
* tonic bursting
* 

returns

![Generated tonic spiking data from the Izhikevich neuron. From top to bottom: the fast dynamics of the neuron (V), where the dots mark a spiking event, the slow dynamics of the neuron, the input current injected to the neuron, and the binned spike data using a bin width of 500 ms.](/imgs/tonic_spiking.svg "Generated tonic spiking data from the Izhikevich neuron")
*From top to bottom: the fast dynamics of the neuron (V), where the dots mark a spiking event, the slow dynamics of the neuron, the input current injected to the neuron, and the binned spike data using a bin width of 500 ms.*

## Recurrent Linear-Nonlinear Poisson model


## Building a design matrix
![Design matrix](/imgs/design_mat.png "Design matrix of the spiking neuron")




## Fitting GLM
To find the optimal filter shape to generate data is based on maximal likelihoods. 

Take the log like... to speed up convergence

![Filters](/imgs/filters.svg "Inferred filters of the glm")

![Generated data](/imgs/simulated_plot.svg "Generated data from the inferred filters of the glm")

## Acknowledgements
