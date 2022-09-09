from collections import namedtuple

# Define different behaviour conditions
PhaseParams = namedtuple("PhaseParams", ['a', 'b', 'c', 'd', 'I0'])

# define time params
Time = namedtuple("Time", ["T", "tt", "dt"])

# define dynamics class
Dynamics = namedtuple("Dynamics", ["tt", "V", "U", "I", "spike_times", "spike_events", "spike_counts", "tt_sample", "bin_wid"])
