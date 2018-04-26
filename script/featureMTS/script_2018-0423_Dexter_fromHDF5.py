import os
import sys
import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import time                 # time code execution
import pickle
import warnings
import h5py

import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import quantities as pq
import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc

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

data_spk_att = load_data_from_h5(dir_data_save,'featureMTS','spk')
# data_spk_rt = load_data_from_h5(dir_data_save,'spot','spk')
data_spk_tuning = load_data_from_h5(dir_data_save,'image','spk')

days = data_spk_att.keys()
firing_rate_criteria = 3
for i in days:
    firing_rate = np.mean(data_spk_att[i]['data'],axis=(0,1))
    data_spk_att[i] = signal_align.select_signal(data_spk_att[i],indx=firing_rate>=firing_rate_criteria)
    selected_chan = np.in1d(data_spk_tuning[i]['signal_id'], data_spk_att[i]['signal_id'])
    data_spk_tuning[i] = signal_align.select_signal(data_spk_tuning[i], indx=selected_chan)
##
""" PSTH """

def plot_mean_errorbar(to_plot=[], ts=[], error_type='se', labels=[]):
    mean = np.mean(to_plot, axis=2)
    if error_type=='se':
        error = np.std(to_plot, axis=2)/np.sqrt(to_plot.shape[2])
    elif error_type=='std':
        error = np.std(to_plot, axis=2)
    for i in range(mean.shape[0]):
        if labels==[]:
            plt.plot(ts, mean[i, :])
        else:
            plt.plot(ts, mean[i, :], label=labels[i])
        plt.fill_between(ts, mean[i, :] - 2 * error[i, :], mean[i, :] + 2 * error[i, :], alpha=0.5)
    plt.legend()


def plot_all_channels(to_plot=[], ts=[], time_window=[], titles=[], subplots=()):
    n_channel = to_plot.shape[2]
    if subplots==():
        ncols = np.ceil(np.sqrt(n_channel))
        nrows = np.ceil(n_channel/ncols)
    else:
        ncols, nrows = subplots
    if len(time_window)==2:
        to_plot = to_plot[:, (ts > time_window[0]) & (ts < time_window[1]),:]
        ts = ts[(ts > time_window[0]) & (ts < time_window[1])]
    for i in range(n_channel):
        plt.subplot(nrows,ncols,i+1)
        for j in range(to_plot[:, :, i].shape[0]):
            plt.plot(ts, to_plot[j, :, i])
        plt.legend()
        if titles!=[]:
            plt.title(titles[i])


def get_psth(data={},to_sort=[],limit=None,time_window=[0.04,0.14],select_firing=0,normalize='soft',smooth_std=0.005,date_remove=[]):
    psth_list,signalID,conditions = [],[],[]
    for i in data.keys():
        if i not in date_remove:
            if limit==None:
                data_sorted = signal_align.neuro_sort(data[i]['trial_info'], grpby=to_sort,neuro=data[i])
            else:
                data_sorted = signal_align.neuro_sort(data[i]['trial_info'], grpby=to_sort,neuro=data[i], fltr=limit[i])
            signalID.append(data[i]['signal_id'])
            conditions.append(data_sorted['cdtn'])
            psth_i = pna.GroupAve(data_sorted)
            psth_list.append(psth_i)
            ts = data[i]['ts']
    signalID = np.concatenate(signalID,axis=0) if len(signalID)>1 else signalID[0]
    psth_all = np.concatenate(psth_list,axis=2) if len(psth_list)>1 else psth_list[0][:,:,None]
    mean_response = np.mean(psth_all[:,(ts>time_window[0]) & (ts<time_window[1]),:], axis=(0, 1))
    psth_all_selected = psth_all[:,:,mean_response>=select_firing]
    if normalize=='soft':
        psth_all_normalized = psth_all_selected / (mean_response + 5)
    elif normalize=='regular':
        psth_all_normalized = psth_all_selected / mean_response
    elif normalize=='none':
        psth_all_normalized = psth_all_selected
    psth_all_smooth = pna.SmoothTrace(psth_all_normalized, sk_std=smooth_std, ts=ts, axis=1)
    return(psth_all_smooth,signalID,ts,conditions)


# plt.figure()
# psth_tuning,signalID,ts_tuning,orientation = get_psth(data_spk_tuning,to_sort=['description'],time_window=[0.04,0.14],normalize='none',smooth_std=0.005)
# plot_all_channels(psth_tuning, ts_tuning)

to_sort = ['ProbeOrientation']
limit_att = {i:(data_spk_att[i]['trial_info']['FeatureType']==0) & (data_spk_att[i]['trial_info']['MatchNot']==1) for i in data_spk_att.keys()}
limit_tuning = {i:(data_spk_tuning[i]['trial_info']['stim_durations']==200) for i in data_spk_tuning.keys()}
psth_tuning, signalID_tuning = [], []
for i in days:
    for j in range(len(data_spk_att[i]['signal_id'])):
        data_spk_att_i_j = {i:signal_align.select_signal(data_spk_att[i], indx=j)}
        data_spk_tuning_i_j = {i:signal_align.select_signal(data_spk_tuning[i], indx=j)}
        psth_tuning_i_j, _, ts_tuning, orientation_tuning = get_psth(data_spk_tuning_i_j, to_sort=['description'],
                                                                 time_window=[0.04, 0.14], normalize='none',limit=limit_tuning)
        psth_att_i_j, _, ts, orientation_att = get_psth(data_spk_att_i_j, to_sort=to_sort, normalize='none', limit=limit_att, time_window=[0.84, 1.04])

        psth_tuning.append(psth_tuning_i_j[np.in1d(orientation_tuning,orientation_att),:])
    signalID_tuning.append(data_spk_att[i]['signal_id'])
psth_tuning = np.concatenate(psth_tuning,axis=2)
signalID_tuning = np.concatenate(signalID_tuning,axis=0)

plt.figure()
plot_all_channels(psth_tuning, ts_tuning, [0,0.6], titles = signalID_tuning, subplots=(10,7))

plt.figure()
# psth_att,signalID,ts,_ = get_psth(data_spk_att,to_sort=to_sort,limit=limit,time_window=[0.84,1.04],date_remove=['171027','171030'])
psth_att,signalID,ts,_ = get_psth(data_spk_att,to_sort=to_sort, normalize='none',limit=limit_att,time_window=[0.84,1.04])
plot_all_channels(psth_att, ts, [0.8,1.4], titles = signalID, subplots=(10,7))

plt.figure()
psth_order,signalID_tuning,ts_tuning,_ = get_psth(data_spk_tuning,to_sort=['order'], normalize='none',limit=limit_tuning)
plot_all_channels(psth_order, ts_tuning, [0,0.6], titles = signalID_tuning, subplots=(10,7))

# labels = ['{} = {}'.format(to_sort[0], conditions[0][i]) for i in range(psth.shape[0])]
# plt.figure()
# plot_mean_errorbar(psth, ts, 'se', labels)
# plt.title('n = {}'.format(signalID.shape[0]))