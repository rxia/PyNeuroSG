import os
import sys
import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import re                   # regular expression
import time                 # time code execution
import pickle
import warnings
import copy
import h5py

import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import quantities as pq
import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc

import mne
import data_load_DLSH       # package specific for DLSH lab data

def load_data_from_h5(dir_data_save = '/shared/homes/rxia/data', block_type = [], signal_type = []):
    hdf_file_path = '{}/all_data_dexter_{}.hdf5'.format(dir_data_save, block_type)
    with h5py.File(hdf_file_path, 'r') as hf_file:
        dict_data = {}
        for date in list(hf_file.keys()):
            print(date, 'reading_dg, {}'.format(block_type))
            data_df = pd.read_json(hf_file[date]['trial_info_json'][()])
            data_df.sort_index(inplace=True)
            print(date, 'reading_neural, {}'.format(block_type))
            data_neural = dict([])
            data_neural['data'] = hf_file[date][signal_type]['data'][:]
            data_neural['ts'] = hf_file[date][signal_type]['ts'][:]
            data_neural['signal_id'] = hf_file[date][signal_type]['signal_id'][:]
            data_neural['trial_info'] = data_df
            dict_data[date] = data_neural
    return(dict_data)

dir_data_save = '/shared/homes/rxia/data'

data_lfp_att = load_data_from_h5(dir_data_save,'featureMTS','lfp')

channels = np.array([1,2,3,5,6,8,11,17,19,22,23,27,28,31])
days = data_lfp_att.keys()
dates_removed = ['171027','171030']
fs = 1/np.mean(np.diff(data_lfp_att['171016']['ts']))
data_mne = {}
for i in days:
    if i not in dates_removed:
        data_lfp_att_i = signal_align.select_signal(data_lfp_att[i],indx=channels)
        info = mne.create_info(ch_names=list(data_lfp_att_i['signal_id'].astype('str')), sfreq=fs)
        data_mne[i] = mne.EpochsArray(data_lfp_att_i['data'].transpose([0, 2, 1]), info, tmin=data_lfp_att_i['ts'][0])

## Compute single trial spectrogram using wavelet and multitaper
epochs = data_mne['171016']
freqs = np.arange(3., 80., 1.)

n_cycles = freqs/5
time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
power_mt = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,picks=np.arange(14),decim=1,
                       time_bandwidth=time_bandwidth, return_itc=False,average=False)
print('multitaper is done')
# n_cycles = 7
# power_mw = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False,picks=np.arange(14),decim=1,average=False)
# print('wavelet is done')

## Visualize single trial spectrogram in beta band (wavelet & multitaper)
from matplotlib.widgets import Button
medians_mw, medians_mt = [], []
f_selected = (freqs>=5) & (freqs<=50)
for i in range(14):
    # median_value = np.median(power_mw.data[:,i,f_selected,:],axis=[0,2])
    # # median_value = np.ones(np.sum(f_selected))
    # medians_mw.append(median_value)
    median_value = np.median(power_mt.data[:,i,f_selected,:],axis=[0,2])
    # median_value = np.ones(np.sum(f_selected))
    medians_mt.append(median_value)
##
fig, ax = plt.subplots(14,2)
ax = ax.transpose().flatten()
im_mw, im_mt = [], []
for i in range(14):
    # plt.axes(ax[i])
    # im_mw.append(plt.pcolormesh(power_mw.times,power_mw.freqs[f_selected],power_mw.data[0,i,f_selected,:]/medians_mt[i][:,None],cmap='jet',vmin=0, vmax=12))
    # plt.axvline(0, c='w', ls='-')
    # plt.axvline(0.3, c='w', ls='--')
    # plt.axvline(0.8, c='w', ls='-')
    # plt.axvline(1.3, c='w', ls='--')
    # plt.ylabel(power_mw.info['ch_names'][i][-4:-2])
    plt.axes(ax[i+14])
    im_mt.append(plt.pcolormesh(power_mt.times,power_mt.freqs[f_selected],power_mt.data[0,i,f_selected,:]/medians_mt[i][:,None],cmap='jet',vmin=0, vmax=12))
    plt.axvline(0, c='w', ls='-')
    plt.axvline(0.3, c='w', ls='--')
    plt.axvline(0.8, c='w', ls='-')
    plt.axvline(1.3, c='w', ls='--')
plt.suptitle('Trial #: 0')

class Index(object):
    ind = 0
    def update_figure(self):
        for i in range(14):
            # im_mw[i].set_array((power_mw.data[self.ind, i, f_selected, :]/medians_mw[i][:,None]).ravel())
            im_mt[i].set_array((power_mt.data[self.ind, i, f_selected, :]/medians_mt[i][:,None]).ravel())
        plt.suptitle('Trial #: {}'.format(self.ind))
        plt.draw()

    def next_trial(self, event):
        self.ind += 1
        self.update_figure()

    def prev_trial(self, event):
        self.ind -= 1
        self.update_figure()

callback = Index()
prev_trial = plt.axes([0.1, 0.0, 0.15, 0.07])
next_trial = plt.axes([0.3, 0.0, 0.15, 0.07])
b_prev_trial = Button(prev_trial, 'prev_trial')
b_prev_trial.on_clicked(callback.prev_trial)
b_next_trial = Button(next_trial, 'next_trial')
b_next_trial.on_clicked(callback.next_trial)
plt.show()

## Average power spectrum
limit_nonmatch = [data_lfp_att[i]['trial_info']['MatchNot']==0 for i in data_lfp_att.keys()]
limit_match = [data_lfp_att[i]['trial_info']['MatchNot']==1 for i in data_lfp_att.keys()]
f_selected = (power_mt.freqs>=10) & (power_mt.freqs<=50)
fig, ax = plt.subplots(5,3)
ax = ax.transpose().flatten()
sum_mean_match, sum_mean_nonmatch = 0, 0
deduct_baseline = False
for i in range(14):
    plt.axes(ax[i])
    if deduct_baseline:
        t_baseline = (power_mt.times>-0.5)&(power_mt.times<-0.1)
        mean_match = np.mean(10*np.log10(power_mt.data[limit_match[0],i,:,:][:,f_selected,:]) - 10*np.log10(np.median(power_mt.data[limit_match[0],i,:,:][:,f_selected,:][:,:,t_baseline],axis=2)[:,:,None]),axis=0)
        mean_nonmatch = np.mean(10*np.log10(power_mt.data[limit_nonmatch[0],i,:,:][:,f_selected,:]) - 10*np.log10(np.median(power_mt.data[limit_nonmatch[0],i,:,:][:,f_selected,:][:,:,t_baseline],axis=2)[:,:,None]),axis=0)
    else:
        mean_match = 10*np.log10(np.mean(power_mt.data[limit_match[0],i,:,:][:,f_selected,:],axis=0))
        mean_nonmatch = 10*np.log10(np.mean(power_mt.data[limit_nonmatch[0],i,:,:][:,f_selected,:],axis=0))
    plt.pcolormesh(power_mt.times,power_mt.freqs[f_selected],mean_match-mean_nonmatch,cmap='jet')
    sum_mean_match += mean_match
    sum_mean_nonmatch += mean_nonmatch
plt.figure()
plt.plot(power_mt.freqs[f_selected],np.mean(sum_mean_match[:,(power_mt.times>=1.1)&(power_mt.times<=1.4)],axis=1)/14)
plt.plot(power_mt.freqs[f_selected],np.mean(sum_mean_nonmatch[:,(power_mt.times>=1.1)&(power_mt.times<=1.4)],axis=1)/14)

##
def group_ave_spctrm(power,channel='all',trials=None,deduct_baseline=False,flim=[0,100],tlim=[-1,2],plot=False):
    f_selected = (power.freqs>=flim[0])&(power.freqs<=flim[1])
    t_selected = (power.times>=tlim[0])&(power.times<=tlim[1])
    spctrm = []
    if channel=='all':
        for i in range(len(trials)):
            power_i = np.mean(power.data[trials[i]][:,:,f_selected,:][:,:,:,t_selected],axis=0)
            spctrm.append(10*np.log10(power_i))
        times = power.times[t_selected]
        freqs = power.freqs[f_selected]
        spctrm = np.stack(spctrm, axis=0)
        if plot:
            _, ax = plt.subplots(4, 4)
            ax = ax.flatten()
            for i in range(14):
                plt.axes(ax[i])
                spctrm_i = list(spctrm[:,i,:,:])
                pnp.DataFastSubplot(data_list=spctrm_i, data_type='spectrum', layout=(4, 4),
                                clim='basic', xx=times, yy=freqs)

    else:
        for i in range(len(trials)):
            power_i = np.mean(power.data[trials[i]][:,channel,f_selected,:][:,:,t_selected],axis=0)
            spctrm.append(10*np.log10(power_i))
        times = power.times[t_selected]
        freqs = power.freqs[f_selected]
        if plot:
            pnp.DataFastSubplot(data_list=spctrm, data_type='spectrum', layout=(4, 4),
                                clim='basic', xx=times, yy=freqs)
        spctrm = np.stack(spctrm, axis=0)

    return spctrm, times, freqs


trials = list(data_lfp_att['171016']['trial_info'].groupby(['ProbeOrientation','ProbeColor']).indices.values())
spctrm, times, freqs = group_ave_spctrm(power_mt,channel='all',trials=trials,flim=[30,50],tlim=[0.6,1.4],plot=True)




## Compute coherence
limit_orien = [data_lfp_att[i]['trial_info']['FeatureType']==0 for i in data_lfp_att.keys()]
limit_color = [data_lfp_att[i]['trial_info']['FeatureType']==1 for i in data_lfp_att.keys()]
conn_orien_delay, freqs_conn, times_delay, n_epochs_orien, n_tapers_conn = \
    mne.connectivity.spectral_connectivity(epochs.get_data()[limit_orien[0]], method='coh', sfreq=fs, mode='multitaper', fmin=3, fmax=50, tmin=1.0, tmax=1.4, block_size=256, n_jobs=10)
conn_color_delay, freqs_conn, times_delay, n_epochs_color, n_tapers_conn = \
    mne.connectivity.spectral_connectivity(epochs.get_data()[limit_color[0]], method='coh', sfreq=fs, mode='multitaper', fmin=3, fmax=50, tmin=1.0, tmax=1.4, block_size=256, n_jobs=10)
conn_orien_visual, freqs_conn, times_visual, n_epochs_orien, n_tapers_conn = \
    mne.connectivity.spectral_connectivity(epochs.get_data()[limit_orien[0]], method='coh', sfreq=fs, mode='multitaper', fmin=3, fmax=50, tmin=1.5, tmax=1.9, block_size=256, n_jobs=10)
conn_color_visual, freqs_conn, times_visual, n_epochs_color, n_tapers_conn = \
    mne.connectivity.spectral_connectivity(epochs.get_data()[limit_color[0]], method='coh', sfreq=fs, mode='multitaper', fmin=3, fmax=50, tmin=1.5, tmax=1.9, block_size=256, n_jobs=10)
##
_, ax = plt.subplots(14,14,sharey=True)
for i in range(14):
    for j in range(14):
        if i>j:
            plt.axes(ax[i,j])
            plt.plot(freqs_conn[freqs_conn<=50],conn_orien_delay[i,j,freqs_conn<=50])
            plt.plot(freqs_conn[freqs_conn<=50],conn_color_delay[i,j,freqs_conn<=50])

_, ax = plt.subplots(14,14,sharey=True)
for i in range(14):
    for j in range(14):
        if i>j:
            plt.axes(ax[i,j])
            plt.plot(freqs_conn[freqs_conn<=50],conn_orien_visual[i,j,freqs_conn<=50])
            plt.plot(freqs_conn[freqs_conn<=50],conn_color_visual[i,j,freqs_conn<=50])

##
limit_nonmatch = [data_lfp_att[i]['trial_info']['status']==0 for i in data_lfp_att.keys()]
limit_match = [data_lfp_att[i]['trial_info']['status']==1 for i in data_lfp_att.keys()]
conn_nonmatch_visual, freqs_conn, times_visual, n_epochs_orien, n_tapers_conn = \
    mne.connectivity.spectral_connectivity(epochs.get_data()[limit_nonmatch[0]], method='coh', sfreq=fs, mode='multitaper', fmin=3, fmax=50, tmin=1.5, tmax=1.9, block_size=256, n_jobs=10)
conn_match_visual, freqs_conn, times_visual, n_epochs_color, n_tapers_conn = \
    mne.connectivity.spectral_connectivity(epochs.get_data()[limit_match[0]], method='coh', sfreq=fs, mode='multitaper', fmin=3, fmax=50, tmin=1.5, tmax=1.9, block_size=256, n_jobs=10)
conn_nonmatch_visual2, freqs_conn, times_visual2, n_epochs_orien, n_tapers_conn = \
    mne.connectivity.spectral_connectivity(epochs.get_data()[limit_nonmatch[0]], method='coh', sfreq=fs, mode='multitaper', fmin=3, fmax=50, tmin=1.7, tmax=2.1, block_size=256, n_jobs=10)
conn_match_visual2, freqs_conn, times_visual2, n_epochs_color, n_tapers_conn = \
    mne.connectivity.spectral_connectivity(epochs.get_data()[limit_match[0]], method='coh', sfreq=fs, mode='multitaper', fmin=3, fmax=50, tmin=1.7, tmax=2.1, block_size=256, n_jobs=10)
##
_, ax = plt.subplots(14,14,sharey=True)
for i in range(14):
    for j in range(14):
        if i>j:
            plt.axes(ax[i,j])
            plt.plot(freqs_conn[(freqs_conn>=10)&(freqs_conn<=50)],conn_match_visual[i,j,(freqs_conn>=10)&(freqs_conn<=50)])
            plt.plot(freqs_conn[(freqs_conn>=10)&(freqs_conn<=50)],conn_nonmatch_visual[i,j,(freqs_conn>=10)&(freqs_conn<=50)])
plt.figure()
plt.plot(freqs_conn[(freqs_conn>=10)&(freqs_conn<=50)],np.mean(conn_match_visual[:,:,(freqs_conn>=10)&(freqs_conn<=50)],axis=(0,1)))
plt.plot(freqs_conn[(freqs_conn>=10)&(freqs_conn<=50)],np.mean(conn_nonmatch_visual[:,:,(freqs_conn>=10)&(freqs_conn<=50)],axis=(0,1)))
