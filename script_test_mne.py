


import mne
from mne.datasets import sample  # noqa
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
print(raw_fname)

raw = mne.io.read_raw_fif(raw_fname)  # load data
print(raw)
print(raw.info)


print(raw.ch_names)

start, stop = raw.time_as_index([100, 115])  # 100 s to 115 s data segment
data, times = raw[:, start:stop]
print(data.shape)
print(times.shape)
data, times = raw[2:20:3, start:stop]  # access underlying data
raw.plot()



raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels
raw.filter(l_freq=None, h_freq=40.0)  # low-pass filter
events = mne.find_events(raw, 'STI014')  # extract events and epoch data
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,
                     reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
evoked = epochs.average()  # compute evoked
evoked.plot()  # butterfly plot the evoked data
cov = mne.compute_covariance(epochs, tmax=0, method='shrunk')
fwd = mne.read_forward_solution(fwd_fname, surf_ori=True)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2)  # compute inverse operator
stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2=1. / 9., method='dSPM')  # apply it
stc_fs = stc.morph('fsaverage')  # morph to fsaverage
stc_fs.plot()  # plot source data on fsaverage's brain

from mne.time_frequency import tfr_morlet  # noqa
n_cycles = 2  # number of cycles in Morlet wavelet
freqs = np.arange(7, 30, 3)  # frequencies of interest
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                        return_itc=True, decim=3, n_jobs=1)
power.plot([power.ch_names.index('MEG 1332')])



""" test case """
import numpy as np
from matplotlib import pyplot as plt

# ----- generate two signals -----
fs = 1000.0  # sampling rate
N_ts = 500  # number of timestamps
N_trial = 100  # number of trials
t_ini = -0.1  # initial timestamp
f0s = 30  # shared frequency, sync
f0n = 50  # shared frequency, not sync
f1 = 10  # unique frequency for signal1
f2 = 90  # unique frequency for signal1
ts = np.arange(0, N_ts).astype(float) * (1 / fs) + t_ini  # timestamps
phi0s = np.random.rand(N_trial, 1) * np.pi * 2
phi0s_lag_cycle = 0.125
phi0n = np.random.rand(N_trial, 1) * np.pi * 2
phi1 = np.random.rand(N_trial, 1) * np.pi * 2
phi2 = np.random.rand(N_trial, 1) * np.pi * 2
Noise1 = 5
Noise2 = 0.5
t_bin = 0.1

signal1 = np.sin(2 * np.pi * f0s * ts + phi0s) \
          + np.sin(2 * np.pi * f0n * ts + 0) \
          + np.sin(2 * np.pi * f1 * ts + phi1) \
          + np.random.randn(N_trial, N_ts) * Noise1
signal2 = np.sin(2 * np.pi * f0s * ts + phi0s - phi0s_lag_cycle * 2 * np.pi) * np.linspace(0, 2, N_ts) * (
np.linspace(0, 2, N_ts) > 1) \
          + + np.sin(2 * np.pi * f0n * ts + phi0n) \
          + np.sin(2 * np.pi * f2 * ts + phi2) * np.linspace(2, 0, N_ts) \
          + np.random.randn(N_trial, N_ts) * Noise2

X = np.dstack([signal1, signal2])
X = np.swapaxes(X, 1,2)



plt.pcolormesh(signal2)