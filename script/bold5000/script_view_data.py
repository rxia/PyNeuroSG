import store_hdf5
import pandas as pd
import re     # use regular expression to find file names
import numpy as np
import scipy as sp
import math
import time
import mne
from mne.connectivity import spectral_connectivity
import copy
import signal_align
import matplotlib as mpl
import matplotlib.pyplot as plt
import df_ana
import PyNeuroData as pnd
import PyNeuroAna as pna
import PyNeuroPlot as pnp
import GM32_layout
import tkinter
import cv2
from skimage.filters import gabor_kernel

animal_name = 'thor'
task = 'bold5000'
dir_data_save = '/shared/homes/rxia/labdata'
signal_type = 'lfp'

def load_data_from_h5(dir_data_save='/shared/homes/rxia/labdata', name='', block_type='', signal_type='', dates=[]):
    hdf_file_path = '{}/all_data_{}_{}.hdf5'.format(dir_data_save, name, block_type)
    dict_data = dict()
    if len(dates) == 0:
        dates = list(store_hdf5.ShowH5(hdf_file_path).keys())
    for date in dates:
        print('loading {} {} {}'.format(date,block_type,signal_type))
        dict_data[date] = store_hdf5.LoadFromH5(hdf_file_path, h5_groups=[date, block_type, signal_type])
    return(dict_data)


dates = ['181024','181026','181028','181030','181101','181104','181105','181106','181107','181108','181110','181113']
# dates = ['181030','181101','181104','181105','181106','181107','181108','181110','181113']
data_spk_all = load_data_from_h5(dir_data_save, animal_name, task, signal_type, dates)
##
import csv
repeat_files = []
with open('/shared/lab/stimuli/BOLD5000_Stimuli/Scene_Stimuli/repeated_stimuli_113_list.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csvreader:
        repeat_files.append(''.join(row))
##
if signal_type == 'spcg':
    [spcg, spcg_t, spcg_f] = pna.ComputeSpectrogram(data_spk_all['181030']['data'],
                                                    fs=data_spk_all['181030']['signal_info']['sampling_rate'][0],
                                                    t_ini=data_spk_all['181030']['ts'][0], t_bin=0.150, t_step=0.01,
                                                    t_axis=1, batchsize=100,
                                                    f_lim=[5, 100])
f_range = [5,15]
spcg_1D = np.moveaxis(np.mean(spcg[:,(spcg_f>=f_range[0])&(spcg_f<=f_range[1])],axis=1),1,2)
spcg_fam = np.nanmean(spcg_1D[np.in1d(data_spk_all['181030']['trial_info']['stim_names'],np.stack(repeat_files))],axis=0)
spcg_nov = np.nanmean(spcg_1D[np.logical_not(np.in1d(data_spk_all['181030']['trial_info']['stim_names'],np.stack(repeat_files)))],axis=0)
list_channels = [5,6,8,11,12,16,19,20,21,22,23,28,30,31]
plt.figure()
plt.plot(spcg_t,spcg_fam[:,list_channels],'-')
plt.plot(spcg_t,spcg_nov[:,list_channels],'--')
##
t_win_visual = [0.050,0.250]
t_win_tuning = [0.20,0.300]
spike_counts_grouped_fam,spike_counts_grouped_nov = [],[]
psth_fam,psth_nov = [],[]
for date in dates:
    print(date)
    data_spk_i = data_spk_all[date]

    if signal_type == 'spk':
        threhold_cohen_d = 0.5
        mean_data = data_spk_i['data'].mean(axis=0)
        ts = data_spk_i['ts']
        ts_baseline = np.logical_and(ts >= -0.200, ts < 0.000)
        ts_visual = np.logical_and(ts >= t_win_visual[0], ts < t_win_visual[1])
        mean_baseline = mean_data[ts_baseline].mean(axis=0)
        std_baseline = mean_data[ts_baseline].std(axis=0)
        mean_visual = mean_data[ts_visual].mean(axis=0)
        std_visual = mean_data[ts_visual].std(axis=0)
        cohen_d_visual = (mean_visual - mean_baseline) \
                         / np.sqrt((std_baseline ** 2 + std_visual ** 2) / 2)
        is_visual = cohen_d_visual > threhold_cohen_d

        data_spk_i['data'] = pna.normalize_across_signals(data_spk_i['data'], axis_signal=-1, axis_t=-2, method='soft',
                                                          t_window=t_win_visual, ts=data_spk_i['ts'], thresh_soft=0)
        ts_tuning = np.logical_and(ts >= t_win_tuning[0], ts < t_win_tuning[1])
        spike_counts_all = np.mean(data_spk_i['data'][:,ts_tuning],axis=1)[:,is_visual]
        stim_names = np.unique(data_spk_i['trial_info']['stim_names'])
        fam_set = stim_names
        spike_counts_grouped = data_spk_i['trial_info'].groupby('stim_names').apply(lambda x: spike_counts_all[x.index,:].mean(axis=0))
        spike_counts_grouped_fam.append(np.array(spike_counts_grouped[np.in1d(stim_names,np.stack(repeat_files))].tolist()))
        spike_counts_grouped_nov.append(np.array(spike_counts_grouped[np.logical_not(np.in1d(stim_names,np.stack(repeat_files)))].tolist()))

        smoothed_data_fam = np.mean(pna.SmoothTrace(data_spk_i['data'][np.in1d(data_spk_i['trial_info']['stim_names'], np.stack(repeat_files))],ts=data_spk_i['ts'], sk_std=0.01), axis=0)
        smoothed_data_nov = np.mean(pna.SmoothTrace(data_spk_i['data'][np.logical_not(np.in1d(data_spk_i['trial_info']['stim_names'], np.stack(repeat_files)))], ts=data_spk_i['ts'], sk_std=0.01), axis=0)
        psth_fam.append(np.mean(smoothed_data_fam[:, is_visual], axis=1))
        psth_nov.append(np.mean(smoothed_data_nov[:, is_visual], axis=1))

    elif signal_type=='lfp':
        psth_fam.append(np.nanmean(data_spk_i['data'][np.in1d(data_spk_i['trial_info']['stim_names'],np.stack(repeat_files))],axis=0))
        psth_nov.append(np.nanmean(data_spk_i['data'][np.logical_not(np.in1d(data_spk_i['trial_info']['stim_names'],np.stack(repeat_files)))],axis=0))

## PSTH
plt.figure()
name_colormap = 'hot'
cycle_color = plt.cm.get_cmap(name_colormap)(np.linspace(0, 1, len(psth_nov)))
for i in range(len(psth_nov)):
    plt.subplot(3,4,i+1)
    plt.plot(data_spk_i['ts'],psth_nov[i],'--')
    plt.plot(data_spk_i['ts'],psth_fam[i],'-')
plt.figure()
if signal_type == 'spk':
    plt.subplot(1,2,1)
    for i in range(len(psth_fam)):
        plt.plot(data_spk_i['ts'],psth_fam[i],'-',c=cycle_color[i])
    plt.subplot(1,2,2)
    for i in range(len(psth_nov)):
        plt.plot(data_spk_i['ts'],psth_nov[i],'--',c=cycle_color[i])

elif signal_type == 'lfp':
    for j in range(32):
        plt.subplot(4,8,j+1)
        if j in [5,6,8,11,12,16,20,21,22,23,26,28,30,31]:
            for i in range(len(psth_nov)):
                plt.plot(data_spk_i['ts'],psth_nov[i][:,j],c=cycle_color[i])


## Rank tuning
plt.figure()
name_colormap = 'hot'
cycle_color = plt.cm.get_cmap(name_colormap)(np.linspace(0, 1, len(spike_counts_grouped_fam)))
for [i,spike_count] in enumerate(spike_counts_grouped_fam):
    rank_tuning = np.sort(spike_count,axis=0)
    plt.plot(np.arange(0,rank_tuning.shape[0]),rank_tuning.mean(axis=1),c=cycle_color[i])
for [i,spike_count] in enumerate(spike_counts_grouped_nov):
    rank_tuning = np.sort(spike_count,axis=0)
    # plt.plot(np.arange(0,rank_tuning.shape[0])*spike_counts_grouped_fam[0].shape[0]/(rank_tuning.shape[0]),rank_tuning.mean(axis=1),'--',c=cycle_color[i])
    random_selected = np.random.randint(0,rank_tuning.shape[0],113)
    rank_selected = np.sort(rank_tuning[random_selected],axis=0)
    plt.plot(np.arange(0,rank_selected.shape[0]),rank_selected.mean(axis=1),'--',c=cycle_color[i])