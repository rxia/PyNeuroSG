"""
Test cases for the package functions
"""

import numpy as np
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt

# package modules
import PyNeuroAna as pna
import PyNeuroPlt as pnp




""" Test case for spectrogram and cohererogram """
if True:
    fs = 1000.0
    N_ts = 500
    N_trial = 100
    t_ini = -0.1
    ts = np.arange(0, N_ts).astype(float) * (1/fs) + t_ini
    f0 = 30
    f1 = 10
    f2 = 60
    phi0 = np.random.rand(N_trial,1)*np.pi*2
    phi1 = np.random.rand(N_trial, 1) * np.pi * 2
    phi2 = np.random.rand(N_trial, 1) * np.pi * 2
    Noise1 = 5
    Noise2 = 0.5
    t_bin = 0.1

    signal1 = np.sin(2 * np.pi * f0 * ts + phi0) + np.sin(2 * np.pi * f1 * ts + phi1) + np.random.randn(N_trial, N_ts) * Noise1
    signal2 = np.sin(2 * np.pi * f0 * ts + phi0) * np.linspace(0,2,N_ts)*(np.linspace(0,2,N_ts)>1) + np.sin(2 * np.pi * f2 * ts + phi2) * np.linspace(2,0,N_ts) + np.random.randn(N_trial, N_ts) * Noise2
    [spcg1, spgc_t, spcg_f] = pna.ComputeSpectrogram(signal1, fs=1000, t_bin=t_bin, t_ini=t_ini)
    [spcg2, spgc_t, spcg_f] = pna.ComputeSpectrogram(signal2, fs=1000, t_bin=t_bin, t_ini=t_ini)
    [cohg, spgc_t, spcg_f]  = pna.ComputeCoherogram(signal1, signal2, fs=1000, t_bin=t_bin, t_ini=t_ini)
    h_fig, h_ax = plt.subplots(3,3, sharex=True, sharey=False, figsize=[16,12])
    plt.axes(h_ax[0, 0])
    plt.plot(ts, signal1.transpose())
    plt.title('signal_1, stable, noisy')
    plt.axes(h_ax[0, 1])
    plt.plot(ts, signal2.transpose())
    plt.title('signal_2, changing power')
    plt.axes(h_ax[1, 0])
    pnp.SpectrogramPlot(spcg1, spgc_t, spcg_f, f_lim=[0, 100], tf_colorbar=True, tf_log=False)
    plt.title('signal_1, power, linear scale, with colorbar')
    plt.axes(h_ax[1, 1])
    pnp.SpectrogramPlot(spcg2, spgc_t, spcg_f, f_lim=[0, 100], tf_colorbar=False, tf_log=False)
    plt.title('signal_2, power, linear scale, no colorbar')
    plt.axes(h_ax[2, 0])
    pnp.SpectrogramPlot(spcg1, spgc_t, spcg_f, f_lim=[0, 100], tf_colorbar=True, tf_log=True, rate_interp=None)
    plt.title('signal_1, power, log scale, non-smoothed')
    plt.axes(h_ax[2, 1])
    pnp.SpectrogramPlot(spcg2, spgc_t, spcg_f, f_lim=[0, 100], tf_colorbar=False, tf_log=True, rate_interp=8)
    plt.title('signal_2, power, log scale, smoothed')

    plt.axes(h_ax[0, 2])
    pnp.SpectrogramPlot(cohg, spgc_t, spcg_f, f_lim=[0, 100], tf_colorbar=True, tf_log=False, rate_interp=8, name_cmap='viridis')
    plt.title('coherence')

    plt.show()