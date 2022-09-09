import matplotlib.pyplot as plt
import numpy as np

from misc import *

# plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

def plot_generated_data(dynamics: Dynamics, filename: str =None):
  """plot 4 row plot fast V dynamics of Izkevich neuron, slow U dynamics, 
  input stimulus current I, and the binned spike counts form spike trains

  Args:
      dynamics (Dynamics): dynamics class of izkevich neurons
      filename (str, optional): Name of filename wish to export. Defaults to None.
  """
  fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,7), sharex=True)
  ax1.plot(dynamics.tt, dynamics.V, lw=1)
  ax1.scatter(dynamics.spike_times, np.full_like(dynamics.spike_times, 40), s=1)
  ax1.set_ylabel("V")
  ax2.plot(dynamics.tt, dynamics.U, lw=1)
  ax2.set_ylabel("U")
  ax3.plot(dynamics.tt, dynamics.I, lw=1)
  ax3.set_ylabel("I")
  markerline,stemlines,baseline = ax4.stem(dynamics.tt[::dynamics.bin_wid], dynamics.spike_counts)
  markerline.set_markerfacecolor('none')
  stemlines.set(color='b', linewidth=.5)
  baseline.set(color='b', linewidth=.5)
  ax4.set_ylabel("Bin spikes")
  ax4.set_xlabel("Time [ms]")
  # fig.suptitle(f"")
  fig.tight_layout()
  if filename is not None:
    fig.savefig(f"{filename}.svg")
  plt.show()
  return

def plot_design_matrix(design_mat: np.ndarray, dynamics: Dynamics, filename: str = None):
  fig, (ax1, ax2) = plt.subplots(
    ncols=2,
    figsize=(6, 8),
    sharey=True,
    gridspec_kw=dict(width_ratios=(5, 1)),
  )
  ax1.imshow(design_mat, aspect='auto', interpolation='nearest')
  ax1.set(
  title="Design matrix with history",
  xlabel="regressors",
  ylabel="Time point (time bins)",
  )
  cbar = ax2.imshow(dynamics.spike_counts[:,np.newaxis], aspect='auto', interpolation='nearest')
  fig.colorbar(cbar, ax=ax2)
  # ax.set(
  # ylabel="spike counts",
  # )
  fig.tight_layout()
  if filename is not None:
    fig.savefig(f"{filename}.svg")
  plt.show()
  return

def plot_kernels(stim_kernel:np.ndarray, hist_kernel:np.ndarray, filename:str=None):
  fig, (ax1, ax2) = plt.subplots(ncols=2,
                               sharey=True,
                               )
  ax1.plot(stim_kernel/np.linalg.norm(stim_kernel), 'o-', label='poisson GLM stil filt', c='r')
  ax1.plot(stim_kernel*0, linestyle='--', c='k')
  ax1.set(xlabel="Time",
          ylabel="",
          title="Stimulus filter")
  ax2.plot(hist_kernel/np.linalg.norm(hist_kernel), 'o-', label='poisson GLM hist filt', c='r')
  ax2.plot(hist_kernel*0, linestyle='--', c='k')
  ax2.set(xlabel="Time",
          ylabel="",
          title="History filter")
  if filename is not None:
    fig.savefig(f"{filename}.svg")
  plt.show()
  return

def plot_prediction(dynamics:Dynamics, prediction:np.ndarray, filename:str=None):
  fig, ax = plt.subplots()
  markerline,stemlines,baseline = plt.stem(dynamics.tt_sample, dynamics.spike_counts, label="spike count", linefmt='b-', basefmt='b-')
  # ax.stem(spike_counts)
  plt.setp(markerline, 'markerfacecolor', 'none')
  plt.setp(stemlines, color='b', linewidth=.5)
  plt.setp(baseline, color='b', linewidth=.5)
  ax.plot(dynamics.tt_sample, prediction, color='orange', label="predicted GLM")
  ax.set(xlabel="Time",
        ylabel="Counts")
  ax.legend()
  if filename is not None:
    fig.savefig(f"{filename}.svg")
  plt.show()
  return
