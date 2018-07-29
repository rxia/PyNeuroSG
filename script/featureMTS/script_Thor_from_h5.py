import store_hdf5
import os     # for getting file paths
import neo    # for reading neural data (TDT format)
import dg2df  # for reading behavioral data
import pandas as pd
import re     # use regular expression to find file names
import numpy as np
import scipy as sp
import math
import time
import mne
from mne.connectivity import spectral_connectivity
import copy
import misc_tools
import signal_align
import matplotlib as mpl
import matplotlib.pyplot as plt
import df_ana
import PyNeuroAna as pna
import PyNeuroPlot as pnp
import data_load_DLSH
import GM32_layout
mpl.style.use('ggplot')

dates_good = ['180715','180718','180721','180723','180727']
dates_bad = ['180713','180714','180717','180725','180726']
dates = dates_good
window_RF = [0.85, 1.2] # Time window to calculate tuning properties


def load_data_from_h5(dir_data_save='/shared/homes/rxia/data', name='', block_type='', signal_type='', dates=[]):
    hdf_file_path = '{}/all_data_{}_{}.hdf5'.format(dir_data_save, name, block_type)
    dict_data = dict()
    if len(dates) == 0:
        dates = list(store_hdf5.ShowH5(hdf_file_path).keys())
    for date in dates:
        print('loading {} {} {}'.format(date,block_type,signal_type))
        dict_data[date] = store_hdf5.LoadFromH5(hdf_file_path, h5_groups=[date, block_type, signal_type])
    return(dict_data)

dir_data_save = '/shared/homes/rxia/data'
data_lfp = load_data_from_h5(dir_data_save,'thor','featureMTS','lfp',dates)
data_spk = load_data_from_h5(dir_data_save,'thor','featureMTS','spk',dates)

##
con_o_o,con_o_c,con_c_c = [],[],[]
for day in dates:
    data_neuro_spk = data_spk[day]
    data_neuro_LFP = data_lfp[day]
    fs = list(data_neuro_LFP['signal_info']['sampling_rate'])[0]
    channels = np.unique(data_neuro_spk['signal_info']['channel_index'])
    data_neuro_LFP = signal_align.select_signal(data_neuro_LFP, indx=channels - 1)
    info = mne.create_info(ch_names=list(data_neuro_LFP['signal_info']['name'].astype('str')), sfreq=fs)
    data_mne = mne.EpochsArray(data_neuro_LFP['data'].transpose([0, 2, 1]), info, tmin=data_neuro_LFP['ts'][0])
    num_signals = len(data_neuro_spk['signal_info'])
    window_offset = [data_neuro_spk['ts'][0],data_neuro_spk['ts'][-1]]

    # Define tuning dimensions, 0 = orientation 1 = color
    data_neuro_spk['trial_info']['tuning0'] = data_neuro_spk['trial_info']['ProbeOrientation'] / 180
    data_neuro_spk['trial_info']['tuning1'] = data_neuro_spk['trial_info']['ProbeColor'] / np.pi / 2
    # Sort the aligned data by tuning dimensions
    data_neuro_spk = signal_align.neuro_sort( data_neuro_spk['trial_info'], ['tuning0', 'tuning1'], [], data_neuro_spk)
    tuning_all = []
    selectivity_type = np.zeros(num_signals)  # 0 = orientation, 1 = color
    plt.figure()
    for i_signal in range(num_signals):  # for each channel
        index_chan = data_neuro_spk['signal_info']['channel_index'][i_signal]
        tuning = pnp.RfPlot(data_neuro_spk, sk_std=0.02, indx_sgnl=i_signal, t_focus=window_RF, tlim=window_RF)
        tuning_all.append(tuning)
        if np.mean(np.std(tuning, axis=0)) < np.mean(np.std(tuning, axis=1)):
            selectivity_type[i_signal] = 1
        signal_name = data_neuro_spk['signal_info']['name'][i_signal]
    plt.close()

    fmin = 3
    fmax = 50
    tmin_b, tmax_b = -0.5 - window_offset[0], -0.2 - window_offset[0]
    tmin_d, tmax_d = 0.4 - window_offset[0], 0.7 - window_offset[0]
    tmin_v, tmax_v = 0.9 - window_offset[0], 1.2 - window_offset[0]
    con_methods = ['coh', 'ppc', 'pli', 'wpli2_debiased']
    groupby_condition = ['FeatureType','MatchNot']
    grpby = df_ana.DfGroupby(data_neuro_LFP['trial_info'], groupby=groupby_condition)
    grpby_idx = [grpby['idx'][key] for key in grpby['idx'].keys()]
    con = np.zeros((3,len(grpby['idx'].keys()),len(con_methods),len(channels),len(channels),15))
    for (i,condition) in enumerate(grpby['idx'].keys()):
        tmp_b, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            data_mne.get_data()[grpby_idx[i], :, :], method=con_methods, mode='multitaper', sfreq=fs, fmin=fmin,
            fmax=fmax, tmin=tmin_b, tmax=tmax_b, faverage=False, mt_adaptive=True, n_jobs=1)
        tmp_d, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            data_mne.get_data()[grpby_idx[i], :, :], method=con_methods, mode='multitaper', sfreq=fs, fmin=fmin,
            fmax=fmax, tmin=tmin_d, tmax=tmax_d, faverage=False, mt_adaptive=True, n_jobs=1)
        tmp_v, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            data_mne.get_data()[grpby_idx[i], :, :], method=con_methods, mode='multitaper', sfreq=fs, fmin=fmin,
            fmax=fmax, tmin=tmin_v, tmax=tmax_v, faverage=False, mt_adaptive=True, n_jobs=1)
        for (j,method) in enumerate(con_methods):
            con[0,i,j] = np.array(tmp_b[j])
            con[1,i,j] = np.array(tmp_d[j])
            con[2,i,j] = np.array(tmp_v[j])
    for i in np.arange(len(channels)):
        for j in np.arange(len(channels)):
            if i > j:
                if selectivity_type[i]<0.5 and selectivity_type[j]<0.5:
                    con_o_o.append(con[:,:,:,i,j,:])
                elif selectivity_type[i]>0.5 and selectivity_type[j]<0.5:
                    con_o_c.append(con[:,:,:,i,j,:])
                elif selectivity_type[i]<0.5 and selectivity_type[j]>0.5:
                    con_o_c.append(con[:,:,:,i,j,:])
                else:
                    con_c_c.append(con[:,:,:,i,j,:])
##
con_o_o,con_o_c,con_c_c = np.stack(con_o_o),np.stack(con_o_c),np.stack(con_c_c)
con_same_prefer_match = np.mean(np.concatenate((con_o_o[:,:,1,:,:],con_c_c[:,:,3,:,:])),axis=0)
con_same_nonprefer_match = np.mean(np.concatenate((con_o_o[:,:,3,:,:],con_c_c[:,:,1,:,:])),axis=0)
con_diff_match = np.mean(con_o_c[:,:,[1,3],:,:],axis=(0,2))
con_same_prefer_nonmatch = np.mean(np.concatenate((con_o_o[:,:,0,:,:],con_c_c[:,:,2,:,:])),axis=0)
con_same_nonprefer_nonmatch = np.mean(np.concatenate((con_o_o[:,:,2,:,:],con_c_c[:,:,0,:,:])),axis=0)
con_diff_nonmatch = np.mean(con_o_c[:,:,[0,2],:,:],axis=(0,2))
plt.figure(figsize=(10, 10))
for m in range(len(con_methods)):
    for t in range(3):
        plt.subplot(4,3,m*3+t+1)
        lines = {'S A M':con_same_prefer_match[t, m, :],
                 'S U M':con_same_nonprefer_match[t, m, :],
                 'D M':con_diff_match[t, m, :],
                 'S A No':con_same_prefer_nonmatch[t, m, :],
                 'S U N':con_same_nonprefer_nonmatch[t, m, :],
                 'D N':con_diff_nonmatch[t, m, :]}
        for l in lines.keys():
            plt.plot(freqs,lines[l],label=l)
        _ = plt.legend()
