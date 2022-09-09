import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.optimize import minimize
from Izhikevich import *



def make_stim_design_matrix(stim, n_filter=6):
    """make a stimulus design matrix based on current input I`

    Args:
        stim (array): current input
        n_filter (int, optional): size of filter kernel. Defaults to 6.

    Returns:
        array: design matrix shape (stim x filter)
    """
    padded_stim = np.hstack([np.zeros(n_filter-1), stim])
    design_stim_matrix = hankel(padded_stim[:-n_filter+1], stim[-n_filter:])
    return design_stim_matrix


def make_history_matrix(spike_count, n_filter=5, n_lags=1):
    """make a history dependent design matrix based on input spike counts

    Args:
        spike_count (array): past spike counts
        n_filter (int, optional): size of kernel. Defaults to 5.
        n_lags (int, optional): numberof timesteps in past. Defaults to 1.

    Returns:
        array: design matrix shape (stim x filter)
    """
    padded_spike_count = np.hstack([np.zeros(n_filter-1+n_lags), spike_count[:-n_lags]])
    design_hist_matrix = hankel(padded_spike_count[:-n_filter+1], spike_count[-n_filter:])
    return design_hist_matrix


tonic_spiking = behaviour_types['tonic spiking']

T = 100000 # units ms
dt = 0.1
N = int(T/dt)
time = Time(T=T, tt=np.arange(0,T, dt), dt=dt)
x0 = np.array([-70, -14]) # initial state

# define stim
n_steps = 20000
periods = np.random.randint(4000, 5000, n_steps)
# amplitudes = np.random.randint(0,2, n_steps)*tonic_spiking.I0
amplitudes = np.random.randint(tonic_spiking.I0-9,tonic_spiking.I0+5, n_steps)
amplitudes[amplitudes==tonic_spiking.I0-3]
I = np.repeat(amplitudes, periods)
_len = len(time.tt)
I = I[:_len]

ix = int(_len//16)
I[-ix:] = 0
I[:ix] = 0
stim=I

# generate dyamics
v, u, spike_events, spike_times = euler_integrate(x0, Izhikevich, time.dt, 
                                                  time.tt, stim, c=-60, d= 6, 
                                                  a=tonic_spiking.a, 
                                                  b=tonic_spiking.b)

# calculate bin count
bin_size=100
spike_count = spike_events.reshape(int(len(stim)// bin_size), bin_size).sum(axis=1)
time_seqs = np.r_[time.tt[::bin_size], time.tt[-1]+(dt*bin_size)]

# plot
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8,5), sharex=True)
ax1.plot(time.tt, v)
ax1.scatter(spike_times, np.full_like(spike_times, 40))
ax1.set_ylabel("V")
ax2.plot(time.tt, u)
ax2.set_ylabel("U")
ax3.plot(time.tt, I)
ax3.set_ylabel("I")
ax3.set_xlabel("Time")
ax4.stem(time_seqs[:-1], spike_count)
ax4.set_ylabel("Bin spikes")
ax4.set_xlabel("Time")

fig.tight_layout()

# define design matrix
n_filter=35
design_stim_matrix = make_stim_design_matrix(stim=stim[::bin_size], n_filter=n_filter)

# plot
fig, (ax1, ax2) = plt.subplots(
    ncols=2,
    figsize=(6, 8),
    sharey=True,
    gridspec_kw=dict(width_ratios=(5, 1)),
)
ax1.imshow(design_stim_matrix, aspect='auto', interpolation='nearest')
ax1.set(
title="Design matrix with history",
xlabel="regressors",
ylabel="Time point (time bins)",
)
ax2.imshow(spike_count[:,np.newaxis], aspect='auto', interpolation='nearest')
ax1.set(
ylabel="spike counts",
)
fig.tight_layout()

design_matrix_offset = np.hstack((np.ones((len(stim[::bin_size]),1)), design_stim_matrix))


# plot spike trigger average
sta = (design_stim_matrix.T @ spike_count)/np.sum(spike_count)

### Plot it
ttk = np.arange(-n_filter+1,1)*time.dt*bin_size  # time bins for STA (in seconds)
plt.clf()
plt.figure(figsize=[12,8])
plt.plot(ttk,ttk*0, 'k--')
plt.plot(ttk, sta.T, 'bo-')
plt.title('STA')
plt.xlabel('time before spike (s)')
plt.xlim([ttk[0],ttk[-1]])
plt.show()

# defining the nlp-GLM
def neg_log_like_lnp(theta, X, y):
    """calculate the loglikihood for linear non-linear poisson process
    LL = ylog(lambda) - lambda - log(y!)

    Args:
        theta (array): array of parameters
        X (array): stimulus inputs
        y (array): spike counts

    Returns:
        float: log likelihood
    """
    rate = np.exp(X@theta)
    log_likihood = np.sum(y @ np.log(rate) - rate)
    return -log_likihood
    
def fit_lnp(stim, spikes, n_filter=25):
    # Build the design matrix
    y = spikes
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_stim_design_matrix(stim, n_filter)])

    # Use a random vector of weights to start (mean 0, sd .2)
    x0 = np.random.normal(0, .6, n_filter+1)

    # Find parameters that minmize the negative log likelihood function
    res = minimize(neg_log_like_lnp, x0, args=(X, y))
    
    return res['x']

def calc_aic(theta, X, y, n_filter):
    LL_expGLM = neg_log_like_lnp(theta, X, y)
    AIC_expGLM = -2*LL_expGLM + 2*(1+n_filter)
    return AIC_expGLM


def predict_spike_counts_lnp(stim, spikes, theta=None, d=25):
  """Compute a vector of predicted spike counts given the stimulus.

  Args:
    stim (1D array): Stimulus values at each timepoint
    spikes (1D array): Spike counts measured at each timepoint
    theta (1D array): Filter weights; estimated if not provided.
    d (number): Number of time lags to use.

  Returns:
    yhat (1D array): Predicted spikes at each timepoint.

  """
  y = spikes
  constant = np.ones_like(spikes)
  X = np.column_stack([constant, make_stim_design_matrix(stim)])
  if theta is None:  # Allow pre-cached weights, as fitting is slow
    theta = fit_lnp(X, y, d)

  yhat = np.exp(X@theta)
  return yhat


theta_lnp = fit_lnp(stim[::bin_size], spike_count, n_filter=n_filter)

# plot fitted filter
fig, ax = plt.subplots()
ax.plot(theta_lnp/np.linalg.norm(theta_lnp), 'o-', label='poisson GLM stil filt', c='r')
ax.plot(theta_lnp*0, linestyle='--', c='k')

