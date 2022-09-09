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
Work was based on Weber, A. & Pillow, J. 2017

Weber, A. I., & Pillow, J. W. (2017). Capturing the Dynamical Repertoire of Single Neurons with Generalized Linear Models. In Neural Computation (Vol. 29, Issue 12, pp. 3260–3289). MIT Press - Journals. https://doi.org/10.1162/neco_a_01021

Izhikevich, E. M. (2003). Simple model of spiking neurons. In IEEE Transactions on Neural Networks (Vol. 14, Issue 6, pp. 1569–1572). Institute of Electrical and Electronics Engineers (IEEE). https://doi.org/10.1109/tnn.2003.820440