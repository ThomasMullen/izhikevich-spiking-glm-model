"""
fit a poisson glm to a tonic spiking step function

notes:
----
# generate design matrix

# fit glm and get parameters

# simulate model

# compare with Izhikevich neuron - tonic spike
"""
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.interpolate import interp1d

from Izhikevich import *
from plotting import plot_glm_matrices, plot_matrix

plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

plot_path = "/Users/tom/Imbizo/Week_03/izhikevich-neuron/plots/tonic_spiking"
np.random.seed(87931)
#   T = len(stim)  # Total number of timepoints (hint: number of stimulus frames)
#   X = np.zeros((T, d))
#   for t in range(T):
#       X[t] = padded_stim[t : t + d]


#   return X


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


T = 100000 # units ms
dt = 0.1
N = int(T/dt)
time = Time(T=T, tt=np.arange(0,T, dt), dt=dt)
x0 = np.array([-70, -14]) # initial state


bin_size=5000

n_filter = 10
n_hist_filter = 9

# stim = set_rand_step_I2(tonic_spiking.I0, time.tt)

# generate spikes
dynamics = generate_izhikevich(x0, behaviour_types['tonic spiking'], time, 
                                       bin_size=bin_size, I_func=set_rand_step_I2,
                                       show_plt=True, name="Tonic Spiking",)
# design matrix
design_stim_matrix = make_stim_design_matrix(stim=dynamics.I[::dynamics.bin_wid], n_filter=n_filter)
design_hist_matrix = make_history_matrix(dynamics.spike_counts, 
                                         n_filter=n_hist_filter, n_lags=1)
design_matrix = np.concatenate([design_stim_matrix, design_hist_matrix], axis=1)
# design_matrix = design_stim_matrix

# plot design matrix
fig, (ax1, ax2) = plt.subplots(
    ncols=2,
    figsize=(6, 8),
    sharey=True,
    gridspec_kw=dict(width_ratios=(5, 1)),
)
ax1.imshow(design_matrix, aspect='auto', interpolation='nearest')
ax1.set(
title="Design matrix with history",
xlabel="regressors",
ylabel="Time point (time bins)",
)
ax2.imshow(dynamics.spike_counts[:,np.newaxis], aspect='auto', interpolation='nearest')
ax1.set(
ylabel="spike counts",
)
fig.tight_layout()

# add offsets
design_matrix_offset = np.hstack((np.ones((len(dynamics.I[::dynamics.bin_wid]),1)), design_matrix))


# fit Poisson GLM
glm_poisson_exp = sm.GLM(endog=dynamics.spike_counts, exog=design_matrix_offset,
                         family=sm.families.Poisson())

pGLM_results = glm_poisson_exp.fit(max_iter=1000, tol=1e-6, tol_criterion='params')

# get filters
glm_const = pGLM_results.params[0]
glm_filter_stim = pGLM_results.params[1:n_filter+1]
glm_filter_hist = pGLM_results.params[-n_hist_filter:]

# predict rates
rate_pred_pGLM = np.exp(glm_const + design_matrix @ pGLM_results.params[1:])


# plot filters
fig, (ax1, ax2) = plt.subplots(ncols=2,
                               sharey=True,
                               )
ax1.plot(glm_filter_stim[2:]/np.linalg.norm(glm_filter_stim[2:]), 'o-', label='poisson GLM stil filt', c='r')
ax1.plot(glm_filter_stim[2:]*0, linestyle='--', c='k')
ax1.set(xlabel="Time",
        ylabel="",
        title="Stimulus filter")
ax2.plot(glm_filter_hist/np.linalg.norm(glm_filter_hist), 'o-', label='poisson GLM hist filt', c='r')
ax2.plot(glm_filter_hist*0, linestyle='--', c='k')
ax2.set(xlabel="Time",
        ylabel="",
        title="History filter")
# fig.savefig(f"{plot_path}/filters.svg")


# plot spike counts to predicted
fig, ax = plt.subplots()
markerline,stemlines,baseline = plt.stem(dynamics.spike_counts, label="spike count", linefmt='b-', basefmt='b-')
# ax.stem(spike_counts)
plt.setp(markerline, 'markerfacecolor', 'none')
plt.setp(stemlines, color='b', linewidth=.5)
plt.setp(baseline, color='b', linewidth=.5)
ax.plot(rate_pred_pGLM, color='orange', label="predicted GLM")
ax.set(xlabel="Time",
       ylabel="Counts")
ax.legend()
# fig.savefig(f"{plot_path}/predicted.svg")


# simulate data
num_repeats = 40
gen_spike_counts = np.random.poisson(np.tile(rate_pred_pGLM.T,[num_repeats,1]))
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,
                                    )
ax1.plot(dynamics.tt, dynamics.I) # plot stimulus
ax1.set(xlabel="Time",
        ylabel="Current",)
ax2.scatter(dynamics.tt_sample, dynamics.spike_counts, s=.5) # plot spike data
ax2.set(xlabel="Time",
        ylabel="Spike Events",)
ax3.imshow(gen_spike_counts, cmap='jet', aspect='auto') # plot simulated data
ax3.set(xlabel="Time",
        ylabel="Simulated Trials",)
fig.savefig(f"{plot_path}/simulated_plot.svg")

pGLM_results.summary2()