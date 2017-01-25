"""
Test cases for the package functions
"""

import numpy as np
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt

# package modules
import PyNeuroAna as pna
import PyNeuroPlot as pnp




""" Test case for LFP spectrogram and cohererogram """
if True:

    # ----- generate two signals -----
    fs = 1000.0   # sampling rate
    N_ts = 500    # number of timestamps
    N_trial = 100 # number of trials
    t_ini = -0.1  # initial timestamp
    f0s = 30       # shared frequency, sync
    f0n = 50       # shared frequency, not sync
    f1  = 10       # unique frequency for signal1
    f2  = 90       # unique frequency for signal1
    ts = np.arange(0, N_ts).astype(float) * (1/fs) + t_ini   # timestamps
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
    signal2 = np.sin(2 * np.pi * f0s * ts + phi0s - phi0s_lag_cycle*2*np.pi) * np.linspace(0,2,N_ts)*(np.linspace(0,2,N_ts)>1) \
              + + np.sin(2 * np.pi * f0n * ts + phi0n) \
              + np.sin(2 * np.pi * f2 * ts + phi2) * np.linspace(2,0,N_ts) \
              + np.random.randn(N_trial, N_ts) * Noise2
    [spcg1, spgc_t, spcg_f] = pna.ComputeSpectrogram(signal1, fs=1000, t_bin=t_bin, t_ini=t_ini)
    [spcg2, spgc_t, spcg_f] = pna.ComputeSpectrogram(signal2, fs=1000, t_bin=t_bin, t_ini=t_ini)
    [cohg, spgc_t, spcg_f]  = pna.ComputeCoherogram(signal1, signal2, fs=1000, t_bin=t_bin, t_ini=t_ini, tf_phase=True)

    h_fig, h_ax = plt.subplots(3,3, sharex=True, sharey=False, figsize=[16,9])
    plt.suptitle('Test Spectrogram / coherence calculation and plot \n Signal1 and Signal2 share synced {}Hz (signal2 with {} cycle lag) and  non-synched {}Hz'.format(f0s, phi0s_lag_cycle, f0n))
    plt.axes(h_ax[0, 0])
    pnp.RasterPlot(signal1, ts)
    plt.title('signal_1, stable, noisy')
    plt.axes(h_ax[0, 1])
    pnp.RasterPlot(signal2, ts)
    plt.title('signal_2, changing power')

    plt.axes(h_ax[1, 0])
    pnp.SpectrogramPlot(spcg1, spgc_t, spcg_f, f_lim=[0, 120], tf_colorbar=True, tf_log=False)
    plt.title('signal_1, power, linear scale, colorbar')
    plt.axes(h_ax[1, 1])
    pnp.SpectrogramPlot(spcg2, spgc_t, spcg_f, f_lim=[0, 120], tf_colorbar=True, tf_log=False)
    plt.title('signal_2, power, linear scale')
    plt.axes(h_ax[1, 2])
    pnp.SpectrogramPlot(spcg2, spgc_t, spcg_f, f_lim=[0, 120], tf_colorbar=True, tf_log=False, time_baseline=[-0.1,0.1])
    plt.title('signal_2, power, use baseline time')

    plt.axes(h_ax[2, 0])
    pnp.SpectrogramPlot(spcg1, spgc_t, spcg_f, f_lim=[0, 120], tf_colorbar=False, tf_log=True, rate_interp=None)
    plt.title('signal_1, power, log scale, non-smoothed')
    plt.axes(h_ax[2, 1])
    pnp.SpectrogramPlot(spcg2, spgc_t, spcg_f, f_lim=[0, 120], tf_colorbar=False, tf_log=True, rate_interp=8)
    plt.title('signal_2, power, log scale, smoothed')
    plt.axes(h_ax[2, 2])
    pnp.SpectrogramPlot(spcg2, spgc_t, spcg_f, f_lim=[0, 120], tf_colorbar=False, tf_log=True, rate_interp=8, time_baseline=[-0.1,0.1])
    plt.title('signal_2, power, use baseline time')

    plt.axes(h_ax[0, 2])
    pnp.SpectrogramPlot(cohg, spgc_t, spcg_f, f_lim=[0, 120], tf_phase=True, tf_colorbar=True, tf_log=False, rate_interp=8, name_cmap='viridis')
    plt.title('coherence')

    plt.show()



""" Test case for spike-LFP phase coupling """
if True:

    # ----- generate two signals -----
    fs = 1000.0   # sampling rate
    N_ts = 500    # number of timestamps
    N_trial = 100 # number of trials
    t_ini = -0.1  # initial timestamp
    f0s = 30       # shared frequency, sync
    f0n = 50       # shared frequency, not sync
    f1  = 10       # unique frequency for signal1
    f2  = 90       # unique frequency for signal1
    ts = np.arange(0, N_ts).astype(float) * (1/fs) + t_ini   # timestamps
    phi0s = np.random.rand(N_trial, 1) * np.pi * 2
    phi0n = np.random.rand(N_trial, 1) * np.pi * 2
    phi1 = np.random.rand(N_trial, 1) * np.pi * 2
    phi2 = np.random.rand(N_trial, 1) * np.pi * 2
    Noise1 = 5
    Noise2 = 0.5
    t_bin = 0.1

    signal_spk = np.sin(2 * np.pi * f0s * ts + phi0s) \
              + np.sin(2 * np.pi * f0n * ts + 0) \
              + np.sin(2 * np.pi * f1 * ts + phi1) \
              + np.random.randn(N_trial, N_ts) * Noise1
    signal_spk = signal_spk > np.percentile(signal_spk, 95)
    signal_LFP = np.sin(2 * np.pi * f0s * ts + phi0s) * np.linspace(0,2,N_ts)*(np.linspace(0,2,N_ts)>1) \
              + + np.sin(2 * np.pi * f0n * ts + phi0n) \
              + np.sin(2 * np.pi * f2 * ts + phi2) * np.linspace(2,0,N_ts) \
              + np.random.randn(N_trial, N_ts) * Noise2

    h_fig, h_ax = plt.subplots(3,3, sharex=True, sharey=False, figsize=[16,9])
    plt.suptitle('Test Spectrogram / coherence calculation and plot \n Signal1 and Signal2 share synced {}Hz, non-synched {}Hz'.format(f0s, f0n))
    plt.axes(h_ax[0, 0])
    pnp.RasterPlot(signal_spk, ts)
    plt.title('signal_1, stable, noisy')
    plt.axes(h_ax[0, 1])
    pnp.RasterPlot(signal_LFP, ts)
    plt.title('signal_2, changing power')