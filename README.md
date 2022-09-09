# Fit Generalised Linear Model to spiking data

A summer project where I wanted to fit a generalised linear model to spiking data and systematically evaluate the performance of a recurrent linear-nonlinear Poisson model. To test the performance of the model data is generated from the Izhikevich neuron (a reduced model of the Hodge-Huxley model).

## Generating the data
Data was generated by feeding in random step currents into the Izhikevich dynamical model characterized by

$$
\begin{equation}
\begin{cases}
    \dot{v} = 0.04v^{2} + 5v + 140 − u + I(t)\\
    \dot{u} = a(bv − u)
\end{cases}
\end{equation}
$$


$$
\begin{equation}
  v(t) >= 30 
    \begin{cases}
      v (t^{+}) = c \\
      u (t^{+}) = u(t) + d
    \end{cases}       
\end{equation}
$$

This model can describe many ifferent firing behaviour by changing the 4 free parameters $a, b, c$ and $ d$.

The dictionary `behaviour_types` in `Izhikevich.py` contains present parameters and current input for the following behaviours:
* Tonic spiking
* Tonic bursting
* Phasic spiking
* Phasic bursting
* Mixed modes
* Spike frequency adaption
* Type I
* Type II
* Spike latency
* Resonator
* Integrator
* Rebound spikes
* Rebound burst
* Threshold variablility
* Bistability I
* Bistability II
  
This returns a Struct dynamics which characterizes the dynamics, time, and spike events.

![Generated tonic spiking data from the Izhikevich neuron. From top to bottom: the fast dynamics of the neuron (V), where the dots mark a spiking event, the slow dynamics of the neuron, the input current injected to the neuron, and the binned spike data using a bin width of 500 ms.](/imgs/tonic_spiking.svg "Generated tonic spiking data from the Izhikevich neuron")
*From top to bottom: the fast dynamics of the neuron (V), where the dots mark a spiking event, the slow dynamics of the neuron, the input current injected to the neuron, and the binned spike data using a bin width of 500 ms.*

## Recurrent Linear-Nonlinear Poisson model

Assume spike count, $y$, follows a **Poisson Distribution**, which is parametrized by a firing rate, $\lambda$. So,
$$y_{t} | x_{t} \sim Poiss(\lambda)$$
where the distribution of an event is defined as
$$ P(y|x, \theta) = \frac{1}{y!}\lambda^{y}e^{-\lambda} .$$

The count rate $\lambda$ is encoded by a **linear** combination of the stimulus filter $f$ and the history filter $h$. The history filter provides a **recurrent** feedback to the Poisson spike generator. The stimulus filter describes how the time history of the stimulus affects the spike generator. So theta is a combination $\theta = [f, h]^T$.

To ensure the encoded lambda is positive we pass the encoded stimulus and history through an exponential **non-linearity** function. Therefore,
$$\lambda = f(\theta x) = e^{\theta x}.$$



## Building a design matrix
![Design matrix](/imgs/design_mat.png "Design matrix of the spiking neuron")




## Fitting GLM



You have a set of $Y=\{y_{i}\}$ representing the number of spikes at time $t=i$ and corresponding stimulus and history features $X=\{x_{i}\}$. 

To fit the GLM to spiking data we want our parameterised distribution $P_{\theta}(y|x)≈P(y|x)$. This is acheived by finding $\theta$ which maximises our likelihood $P(Y |X, \theta)$.

As each event is assumed to be conditionally independent we can factorise the likelihood probability which yield,
$$P(Y|X,\theta) = ∏^{N}_{i=1}P(y_{i}|x_{i}, \theta)$$

Taking the log of the likelihood will speed up convergence an also reduces the complexity of the analytical solution.
$$Log(P(Y|X,\theta)) = ∑^{N}_{i=1}y_{i}log(\lambda_{i}) -\lambda_{i} -log(y_{t}!)$$
and in the recurrent linear-nonlinear Poisson model,
$$\lambda_{i} = e^{x_{i}^{T}\theta}$$

To find the maxima we take the derivative with respect to the $\theta$ which reduces to,
$$\partial_{\theta} Log(P(Y|X,\theta)) = YX^{T} - X^{T}e^{X^{T}\theta}.$$

![Filters](/imgs/filters.svg "Inferred filters of the glm")

![Generated data](/imgs/simulated_plot.svg "Generated data from the inferred filters of the glm")


## Behaviour and fine tuning
Tonic bursting:
bin_width = 2000
n_filter = 17
n_hist_filter = 7

## Acknowledgements
Work was a replication of works from on Weber, A. & Pillow, J. 2017

Weber, A. I., & Pillow, J. W. (2017). Capturing the Dynamical Repertoire of Single Neurons with Generalized Linear Models. In Neural Computation (Vol. 29, Issue 12, pp. 3260–3289). MIT Press - Journals. https://doi.org/10.1162/neco_a_01021

Izhikevich, E. M. (2003). Simple model of spiking neurons. In IEEE Transactions on Neural Networks (Vol. 14, Issue 6, pp. 1569–1572). Institute of Electrical and Electronics Engineers (IEEE). https://doi.org/10.1109/tnn.2003.820440