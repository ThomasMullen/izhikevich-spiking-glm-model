from ast import main
from keyword import kwlist
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from collections import namedtuple
from functools import partial
from misc import *
from plotting import plot_generated_data

# plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

np.random.seed(1234)

def Izhikevich(v, u, I=None, a=0.02, b=0.2):
    '''
    Given:
       v, u:    fast and slow dimensional variables
       a, b:    parameters defining the dynamical system
       c:       reset voltage
       d:       
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    v_dot = 0.04 * v**2 + 5*v + 140 - u + I
    u_dot = a * (b*v - u)
    return v_dot, u_dot


def euler_integrate(initial_state, derivative_function, dt, timesteps, I, c=-60, d= 6, **kwargs):
    """Integrate 2 variable dynamical system via euler method

    Args:
        initial_state (np.array): 2 element array of initial state
        derivative_function (func): rate equation
        dt (float): time interval
        timesteps (array): timesteps equation is evaluated.

    Returns:
        tuple: two dimensional timeseries
    """
    # Maximal Spike Value
    spike_value = 30                            
    v, u = np.zeros((len(initial_state), len(timesteps)))
    spike_count = np.zeros_like(v)
    spike_times = []
    v[0], u[0] = initial_state
    for i in range(1,len(timesteps)):
        # TODO: flip condition statement
        # if v[i-1] < spike_value:
        # ODE for membrane potential
        v_dot, u_dot = derivative_function(v[i-1], u[i-1], I[i-1], **kwargs)
        v[i] = v[i-1] + v_dot * dt
        u[i] = u[i-1] + u_dot * dt
        # spike reached
        if v[i-1] >= spike_value:
            spike_times.append(timesteps[i-1])
            v[i-1] = spike_value    # set to spike value
            spike_count[i-1] = 1    # set to spike bin
            v[i] = c                # reset membrane voltage
            u[i] = u[i-1] + d       # reset recovery
    return v, u, spike_count, spike_times


def set_rand_step_I(I0, tt, n_steps=2000, lower=23, upper=22):
    """generate a random steps current

    Args:
        I0 (float): height of step
        tt (array): timesteps
        duration (float): duration in seconds
    """
    # generate random periods
    periods = np.random.randint(40000, 50000, n_steps)
    # amplitudes = np.random.randint(I0-lower,I0+upper, n_steps)
    amplitudes = np.random.randint(0,2, n_steps)*I0
    I = np.repeat(amplitudes, periods)
    I[I==I0-3] = 0
    _len = len(tt)
    
    I = I[:_len]
    
    ix = int(_len//16)
    I[-ix:] = 0
    I[:ix] = 0
    
    return I[:_len]


def set_rand_step_I2(I0, tt, n_steps=2000, lower=14, upper=5):
    """generate a random steps current

    Args:
        I0 (float): height of step
        tt (array): timesteps
        duration (float): duration in seconds
    """
    # generate random periods
    periods = np.random.randint(40000, 50000, n_steps)
    amplitudes = np.random.randint(I0-lower,I0+upper, n_steps)
    # amplitudes = np.random.randint(0,2, n_steps)*I0
    I = np.repeat(amplitudes, periods)
    I[I==I0-3] = 0
    _len = len(tt)
    
    I = I[:_len]
    
    ix = int(_len//16)
    I[-ix:] = 0
    I[:ix] = 0
    
    return I[:_len]


def set_rand_step_I(I0, tt, n_steps=2000, lower=23, upper=22):
    """generate a random steps current

    Args:
        I0 (float): height of step
        tt (array): timesteps
        duration (float): duration in seconds
    """
    # generate random periods
    periods = np.random.randint(40000, 50000, n_steps)
    amplitudes = np.random.randint(I0-lower,I0+upper, n_steps)
    I = np.repeat(amplitudes, periods)
    I[I==I0-3] = 0
    _len = len(tt)
    
    I = I[:_len]
    
    ix = int(_len//16)
    I[-ix:] = 0
    I[:ix] = 0
    
    return I[:_len]


def set_step_I(I0, tt):
    """generate a steps current

    Args:
        I0 (float): height of step
        tt (array): timesteps
        duration (float): duration in seconds
    """
    # generate random periods
    
    I = np.ones_like(tt)*I0
    _len = len(tt)
    
    I = I[:_len]
    
    ix = int(_len//16)
    I[-ix:] = 0
    I[:ix] = 0
    
    return I[:_len]

behaviour_types={
    "tonic spiking" : PhaseParams(a=0.02, b=0.2, c=-65, d=6, I0=14),
    "phasic spiking" : PhaseParams(0.02, 0.25, -65, 6, 0.5),
    "tonic bursting" : PhaseParams(0.02, 0.2, -50, 2, 10),
    "phasic bursting" : PhaseParams(0.02, 0.25, -55, 0.05, 0.6),
    "mixed mode" : PhaseParams(0.02, 0.2, -55, 4, 10),
    "spike freq adaptation" : PhaseParams(0.01, 0.2, -65, 5, 20),
    "type I" : PhaseParams(0.02, -0.1, -55, 6, 25),
    "type II" : PhaseParams(0.02, 0.2, -65, 0, 0.5),
    "spike latency" : PhaseParams(0.02, 0.2, -65, 6, 3.49),
    "resonator" : PhaseParams(0.1, 0.26, -60, -1, 0.3),
    "integrator" : PhaseParams(0.02, -0.1, -55, 6, 27.4),
    "rebound spike" : PhaseParams(0.03, 0.25, -60, 4, -5),
    "rebound burst" : PhaseParams(0.03, 0.25, -52, 0, -5),
    "threshold variability" : PhaseParams(0.03, 0.25, -60, 4, 2.3),
    "bistability I" : PhaseParams(1, 1.5, -60, 0, 26.1),
    "bistability II" : PhaseParams(1.02, 1.5, -60, 0, 26.1),
    }



def generate_izhikevich(x_init, params, time_params, bin_size=5000, I=None, I_func=set_rand_step_I2, show_plt=False, filename=None, name="", **kwargs):
    if I is None:
        I = I_func(params.I0, time_params.tt)
    V, U, spike_events, spike_times = euler_integrate(initial_state=x_init, derivative_function=Izhikevich, 
                                   dt=time_params.dt, timesteps=time_params.tt, I=I, c=params.c,
                                   d= params.d, a=params.a, b=params.b)
    # bin and calculate spikes
    assert (len(I)//bin_size)*bin_size==spike_events.size, f"Bin size must factor in "\
                                                            "stimulus len. I: {I.size},"\
                                                            " bin width: {bin_size}, spike "\
                                                            "event size: {spike_events.size}."
    spike_count = spike_events.reshape(int(len(I)// bin_size), bin_size).sum(axis=1)
    # time_seqs = np.r_[time_params.tt[::bin_size], time_params.tt[-1]+(dt*bin_size)]
    # spike_count = np.histogram(spike_times, time_seqs)[0]
    return Dynamics(time_params.tt, V, U, I, spike_times, spike_events, spike_count, time_params.tt[::bin_size], bin_size)


def add_spike_jitter(spikes, time, jitter=0.1):
    # dynamics = dynamics._replace(spike_count=1)
    spike_ix = np.where(spikes == 1)[0]
    jitter = np.round((np.random.rand(len(spike_ix)) -0.5) * 
                       2 * jitter/time.dt).astype(int)
    
    spike_ix += jitter
    spikes = np.zeros_like(spikes)
    spikes[spike_ix] = 1
    return spikes, spike_ix
    
def time_width_to_n_bins(time, time_width):
    """calculate number of bins to count spikes given a fixed period."""
    return int(time.T/time_width)


def resample(data, n):
    from scipy.interpolate import interp1d
    m = len(data)
    xin, xout = np.arange(n, 2*m*n, 2*n), np.arange(m, 2*m*n, 2*m)
    return interp1d(xin, data, 'nearest', fill_value='extrapolate')(xout)


def generate_spikes(dynamics_func, x_init, params, time_params, time_width=10, jitter=None, **kwargs):
    dynamics = dynamics_func(x_init, params, time_params, **kwargs)
    n_bins = time_width_to_n_bins(time_params, time_width)
    if jitter is not None:
        _, spike_ix = add_spike_jitter(params.spike_count, time, jitter=jitter)
        dynamics = dynamics._replace(time.tt[spike_ix])
        
    counts, _ = np.histogram(dynamics.spike_times, bins=n_bins)
    return counts, dynamics


if main == "__main__":
    T = 100000 # units ms
    dt = 0.1
    N = int(T/dt)
    time = Time(T=T, tt=np.arange(0,T, dt), dt=dt)
    x0 = np.array([-70, -14]) # initial state

    time_width=5
    jitter = None
    

    for key, behavior_params in behaviour_types.items():
        print(f"{key}\n====\n")

        # generate spikes    
        dynamics = generate_izhikevich(x0, behavior_params, time, 
                                       time_bin_width=time_width,
                                       show_plt=True, 
                                       filename=f"/Users/tom/Imbizo/Week_03/izhikevich-neuron/plots/izhikevich_dynamics/{key.replace(' ', '_')}", name=key)
        plot_generated_data(dynamics)


    # mplot3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(dynamics.U, dynamics.V, dynamics.tt)
    
    
    