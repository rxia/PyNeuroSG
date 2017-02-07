"""
Test cases for the package functions
"""

import numpy as np
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt

# package modules
import PyNeuroAna as pna
import PyNeuroPlot as pnp
reload(pna)
reload(pnp)



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
    [spcg1, spcg_t, spcg_f] = pna.ComputeSpectrogram(signal1, fs=fs, t_bin=t_bin, t_ini=t_ini, f_lim=[0, 120])
    [spcg2, spcg_t, spcg_f] = pna.ComputeSpectrogram(signal2, fs=fs, t_bin=t_bin, t_ini=t_ini, f_lim=[0, 120])
    [cohg, spcg_t, spcg_f]  = pna.ComputeCoherogram(signal1, signal2, fs=fs, t_bin=t_bin, t_ini=t_ini, tf_phase=True, f_lim=[0, 120])

    h_fig, h_ax = plt.subplots(3,3, sharex=True, sharey=False, figsize=[16,9])
    plt.suptitle('Test Spectrogram / coherence calculation and plot \n Signal1 and Signal2 share synced {}Hz (signal2 with {} cycle lag) and  non-synched {}Hz'.format(f0s, phi0s_lag_cycle, f0n))
    plt.axes(h_ax[0, 0])
    pnp.RasterPlot(signal1, ts)
    plt.title('signal_1, stable, noisy')
    plt.axes(h_ax[0, 1])
    pnp.RasterPlot(signal2, ts)
    plt.title('signal_2, changing power')

    plt.axes(h_ax[1, 0])
    pnp.SpectrogramPlot(spcg1, spcg_t, spcg_f, f_lim=[0, 120], tf_colorbar=True, tf_log=False, tf_mesh_f=True,)
    plt.title('signal_1, power, linear scale, colorbar')
    plt.axes(h_ax[1, 1])
    pnp.SpectrogramPlot(spcg2, spcg_t, spcg_f, f_lim=[0, 120], tf_colorbar=True, tf_log=False, tf_mesh_f=True,)
    plt.title('signal_2, power, linear scale')
    plt.axes(h_ax[1, 2])
    pnp.SpectrogramPlot(spcg2, spcg_t, spcg_f, f_lim=[0, 120], tf_colorbar=True, tf_log=False, time_baseline=[-0.1,0.1], tf_mesh_f=True,)
    plt.title('signal_2, power, use baseline time')

    plt.axes(h_ax[2, 0])
    pnp.SpectrogramPlot(spcg1, spcg_t, spcg_f, f_lim=[0, 120], tf_colorbar=False, tf_log=True, rate_interp=None, tf_mesh_f=True,)
    plt.title('signal_1, power, log scale, non-smoothed')
    plt.axes(h_ax[2, 1])
    pnp.SpectrogramPlot(spcg2, spcg_t, spcg_f, f_lim=[0, 120], tf_colorbar=False, tf_log=True, rate_interp=8, tf_mesh_f=True,)
    plt.title('signal_2, power, log scale, smoothed')
    plt.axes(h_ax[2, 2])
    pnp.SpectrogramPlot(spcg2, spcg_t, spcg_f, f_lim=[0, 120], tf_colorbar=False, tf_log=True, rate_interp=8, time_baseline=[-0.1,0.1], tf_mesh_f=True,)
    plt.title('signal_2, power, use baseline time')

    plt.axes(h_ax[0, 2])
    pnp.SpectrogramPlot(cohg, spcg_t, spcg_f, f_lim=[0, 120], tf_phase=True, tf_mesh_f=True,
                        tf_colorbar=True, tf_log=False,
                        rate_interp=8, name_cmap='viridis')
    plt.title('coherence')

    plt.savefig('./demo_figs/Showcase_LFP_spectrogram_coherence')
    plt.show()



""" Test case for spike-LFP phase coupling """
if False:

    # ----- generate two signals -----
    fs = 1000.0   # sampling rate
    N_ts = 1000    # number of timestamps
    N_trial = 200 # number of trials
    t_ini = -0.1  # initial timestamp
    f0s = 30       # shared frequency, sync, locked to trial
    f0n = 50       # shared frequency, sync, not locked to trial
    f1  = 10       # unique frequency for signal1
    f2  = 90       # unique frequency for signal1
    ts = np.arange(0, N_ts).astype(float) * (1/fs) + t_ini   # timestamps
    phi0s = np.random.rand(N_trial, 1) * np.pi * 2
    phi0s_lag_cycle = 0.125        # LFP lags spk
    phi0n = np.random.rand(N_trial, 1) * np.pi * 2
    phi1 = np.random.rand(N_trial, 1) * np.pi * 2
    phi2 = np.random.rand(N_trial, 1) * np.pi * 2
    Noise1 = 4.0
    Noise2 = 0.5
    spk_criterion = 0.99  # sparse if close to 1.0, dense if close to 0.0
    t_bin = 0.2

    signal_spk = np.sin(2 * np.pi * f0s * ts) \
              + np.sin(2 * np.pi * f0n * ts + phi0n) \
              + np.sin(2 * np.pi * f1 * ts + phi1) \
              + np.random.randn(N_trial, N_ts) * Noise1
    signal_spk = signal_spk > np.random.rand(*signal_spk)
    signal_LFP = np.sin(2 * np.pi * f0s * ts - phi0s_lag_cycle*2*np.pi) * np.linspace(0,2,N_ts)*(np.linspace(0,2,N_ts)>1) \
              + + np.sin(2 * np.pi * f0n * ts + phi0n) \
              + np.sin(2 * np.pi * f2 * ts + phi2) * np.linspace(2,0,N_ts) \
              + np.random.randn(N_trial, N_ts) * Noise2

    h_fig, h_ax = plt.subplots(3,3, sharex=True, sharey=False, figsize=[16,9])
    plt.suptitle('Test Spectrogram / coherence calculation and plot \n Signal1 and Signal2 share {0}Hz and {1}Hz; but the {0}Hz is locked to trial, {1}Hz is not'.format(f0s, f0n))
    plt.axes(h_ax[0, 0])
    pnp.RasterPlot(signal_spk, ts)
    plt.title('signal_spk, stable, noisy')
    plt.axes(h_ax[0, 1])
    pnp.RasterPlot(signal_LFP, ts)
    plt.title('signal_LFP, changing power')

    [spcg_spk, spcg_t, spcg_f] = pna.ComputeSpectrogram(signal_spk, fs=fs, t_bin=t_bin, t_ini=t_ini, f_lim=[0, 120])
    [spcg_LFP, spcg_t, spcg_f] = pna.ComputeSpectrogram(signal_LFP, fs=fs, t_bin=t_bin, t_ini=t_ini, f_lim=[0, 120])
    [cohg, spcg_t, spcg_f]  = pna.ComputeCoherogram(signal_LFP, signal_spk, fs=fs, t_bin=t_bin, t_ini=t_ini, tf_phase=True, f_lim=[0, 120], tf_vs_shuffle=False)
    [PLV, spcg_t, spcg_f] = pna.ComputeSpkTrnFieldCoupling(signal_LFP, signal_spk, fs=fs, t_bin=t_bin, t_ini=t_ini, measure='PLV', f_lim=[0, 120], tf_vs_shuffle=False)
    [PPC, spcg_t, spcg_f] = pna.ComputeSpkTrnFieldCoupling(signal_LFP, signal_spk, fs=fs, t_bin=t_bin, t_ini=t_ini, measure='PPC', f_lim=[0, 120], tf_vs_shuffle=False)
    [PLV_shuffle, spcg_t, spcg_f] = pna.ComputeSpkTrnFieldCoupling(signal_LFP, signal_spk, fs=fs, t_bin=t_bin, t_ini=t_ini, measure='PLV', f_lim=[0, 120], tf_vs_shuffle=True)
    [PPC_shuffle, spcg_t, spcg_f] = pna.ComputeSpkTrnFieldCoupling(signal_LFP, signal_spk, fs=fs, t_bin=t_bin, t_ini=t_ini, measure='PPC', f_lim=[0, 120], tf_vs_shuffle=True)

    plt.axes(h_ax[1, 0])
    pnp.SpectrogramPlot(spcg_spk, spcg_t, spcg_f, f_lim=[0, 120], tf_colorbar=True, tf_log=False)
    plt.title('spk, power, linear scale, colorbar')
    plt.axes(h_ax[1, 1])
    pnp.SpectrogramPlot(spcg_LFP, spcg_t, spcg_f, f_lim=[0, 120], tf_colorbar=True, tf_log=False)
    plt.title('LFP, power, linear scale')


    plt.axes(h_ax[0, 2])
    pnp.SpectrogramPlot(cohg, spcg_t, spcg_f, f_lim=[0, 120], tf_phase=True, tf_colorbar=True, tf_log=False,
                        rate_interp=8, c_lim_style='from_zero')
    plt.title('coherence')

    plt.axes(h_ax[1, 2])
    pnp.SpectrogramPlot(PLV, spcg_t, spcg_f, f_lim=[0, 120], tf_phase=True, tf_colorbar=True, tf_log=False,
                        rate_interp=8, c_lim_style='from_zero')
    plt.title('PLV')

    plt.axes(h_ax[2, 2])
    pnp.SpectrogramPlot(PPC, spcg_t, spcg_f, f_lim=[0, 120], tf_phase=True, tf_colorbar=True, tf_log=False,
                        rate_interp=8, c_lim_style='from_zero')
    plt.title('PPC')

    plt.axes(h_ax[2, 0])
    pnp.SpectrogramPlot(PLV_shuffle, spcg_t, spcg_f, f_lim=[0, 120], tf_phase=True, tf_colorbar=True, tf_log=False,
                        rate_interp=8, c_lim_style='diverge')
    plt.title('PLV_vs_shuffle')

    plt.axes(h_ax[2, 1])
    pnp.SpectrogramPlot(PPC_shuffle, spcg_t, spcg_f, f_lim=[0, 120], tf_phase=True, tf_colorbar=True, tf_log=False,
                        rate_interp=8, c_lim_style='diverge')
    plt.title('PPC_vs_shuffle')

    plt.savefig('./demo_figs/Showcase_spk_LFP_phase_sync.png')
    plt.show()