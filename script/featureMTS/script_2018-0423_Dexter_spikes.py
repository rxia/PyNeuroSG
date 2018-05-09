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

data_spk_att_orig = copy.deepcopy(data_spk_att)
data_spk_tuning_orig = copy.deepcopy(data_spk_tuning)

##
data_spk_att = copy.deepcopy(data_spk_att_orig)
data_spk_tuning = copy.deepcopy(data_spk_tuning_orig)

days = data_spk_att.keys()
firing_rate_criteria = 4
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


def plot_all_channels(to_plot=[], ts=[], time_window=False, titles=[], h_ax=False, subplots=(),linestyle='-',colors=False):
    if to_plot.ndim==3:
        n_channel = to_plot.shape[2]
        if subplots==():
            ncols = np.ceil(np.sqrt(n_channel))
            nrows = np.ceil(n_channel/ncols)
        else:
            ncols, nrows = subplots
    elif to_plot.ndim==4:
        n_channel = to_plot.shape[3]*to_plot.shape[2]
        ncols = to_plot.shape[3]
        nrows = to_plot.shape[2]
        to_plot = np.reshape(to_plot,[to_plot.shape[0],to_plot.shape[1],to_plot.shape[2]*to_plot.shape[3]])
    if time_window:
        to_plot = to_plot[:, (ts > time_window[0]) & (ts < time_window[1]),:]
        ts = ts[(ts > time_window[0]) & (ts < time_window[1])]
    for i in range(n_channel):
        if h_ax:
            plt.axes(h_ax[i])
        else:
            plt.subplot(nrows, ncols, i + 1)
        for j in range(to_plot[:, :, i].shape[0]):
            if colors:
                plt.plot(ts, to_plot[j, :, i], linestyle=linestyle, color=colors[j])
            else:
                plt.plot(ts, to_plot[j, :, i], linestyle=linestyle)
        plt.legend()
        if titles!=[]:
            plt.title(titles[i])


def get_psth(data,to_sort,limit=False,select_firing=0,time_window=False,normalize='none',time_window_normalize=[0.04,0.14],smooth_std=0.005,date_remove=[]):
    psth_list,signalID,conditions = [],[],[]
    for i in data.keys():
        if i not in date_remove:
            if limit:
                data_sorted = signal_align.neuro_sort(data[i]['trial_info'], grpby=to_sort,neuro=data[i], fltr=limit[i])
            else:
                data_sorted = signal_align.neuro_sort(data[i]['trial_info'], grpby=to_sort,neuro=data[i])
            signalID.append(data[i]['signal_id'])
            conditions.append(data_sorted['cdtn'])
            psth_i = pna.GroupAve(data_sorted)
            psth_list.append(psth_i)
            ts = data[i]['ts']
    signalID = np.concatenate(signalID,axis=0) if len(signalID)>1 else signalID[0]
    psth_all = np.concatenate(psth_list,axis=2) if len(psth_list)>1 else psth_list[0][:,:,None]
    mean_response = np.mean(psth_all[:,(ts>time_window_normalize[0]) & (ts<time_window_normalize[1]),:], axis=(0, 1))
    psth_mean_selected = psth_all[:,:,mean_response>=select_firing]
    if normalize=='soft':
        psth_mean_normalized = psth_mean_selected / (mean_response + 5)
    elif normalize=='regular':
        psth_mean_normalized = psth_mean_selected / mean_response
    elif normalize=='none':
        psth_mean_normalized = psth_mean_selected
    if smooth_std:
        psth_mean_smooth = pna.SmoothTrace(psth_mean_normalized, sk_std=smooth_std, ts=ts, axis=1)
    else:
        psth_mean_smooth = psth_mean_normalized
    if time_window:
        psth_mean_smooth = psth_mean_smooth[:,(ts>time_window[0])&(ts<time_window[1])]
        ts = ts[(ts>time_window[0])&(ts<time_window[1])]
    return(psth_mean_smooth,signalID,ts,conditions)


def get_tuning(psth,conditions=False,time_window=False,ts=False):
    if time_window:
        psth_mean = np.mean(psth[:,(ts>time_window[0])&(ts<time_window[1]),:],axis=1)
    else:
        psth_mean = np.mean(psth,axis=1)
    if conditions:
        conditions_transposed = zip(*conditions)
        # conditions_unique = [np.unique(condition_i) for condition_i in conditions_transposed]
        # conditions_size = [len(condition_i_unique) for condition_i_unique in conditions_unique]
        # psth_mean = np.zeros(shape=conditions_size) * np.nan

        n_dims,axis = [],[]
        for i in conditions_transposed:
            n_dims.append(len(np.unique(i)))
            axis.append(np.unique(i))
        psth_mean = np.reshape(psth_mean,n_dims+[psth_mean.shape[-1]])
        return psth_mean, axis
    else:
        return psth_mean


def get_grouped_data(data,to_sort,limit=False,date_remove=[]):
    grouped_list, signalID, conditions = [], [], []
    for i in data.keys():
        if i not in date_remove:
            if limit:
                data_sorted = signal_align.neuro_sort(data[i]['trial_info'], grpby=to_sort, neuro=data[i],
                                                      fltr=limit[i])
            else:
                data_sorted = signal_align.neuro_sort(data[i]['trial_info'], grpby=to_sort, neuro=data[i])
            signalID.append(data[i]['signal_id'])
            conditions.append(data_sorted['cdtn'])
            grouped_i = pna.GroupWithoutAve(data_sorted)
            grouped_list.append(grouped_i)
            ts = data[i]['ts']
    return(grouped_list,signalID,ts,conditions)


def get_correlogram(data_all,ts_all,time_win=False,smooth_std=False,downsample_win=0):
    mean_corr_list = []
    for data_select in data_all:
        if time_win:
            data = data_select[:,(ts_all>time_win[0])&(ts_all<time_win[1]),:]
            ts = ts_all[(ts_all>time_win[0])&(ts_all<time_win[1])]
        else:
            ts = ts_all
        if downsample_win:
            downsample_kernel = np.ones(downsample_win)
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    data[i,:,j] = np.convolve(data[i,:,j],downsample_kernel,'same')
            data = data[:,0::downsample_win,:]
        corr = np.zeros([data.shape[0],data.shape[1]*2-1,data.shape[2],data.shape[2]])
        for i in range(data.shape[2]):
            for j in range(data.shape[2]):
                for k in range(data.shape[0]):
                    signal_i = data[k,:,i]/(np.std(data[k,:,i])+0.0000001)
                    signal_j = data[k,:,j]/(np.std(data[k,:,j])+0.0000001)
                    corr[k,:,i,j] = sp.signal.correlate(signal_i,signal_j)/len(data[k,:,i])
        mean_corr = np.mean(corr, axis=0)
        ts = np.linspace(ts.min()-ts.max(),ts.max()-ts.min(),mean_corr.shape[0])
        if smooth_std:
            mean_corr = pna.SmoothTrace(mean_corr, sk_std=smooth_std, ts=ts, axis=0)
        mean_corr_list.append(mean_corr)
    mean_corr_list = np.stack(mean_corr_list,axis=0)
    return mean_corr_list,ts


to_sort = ['ProbeOrientation']
subplots = (8,8)

# limit_tuning = {i:(data_spk_tuning[i]['trial_info']['stim_durations']==200) for i in data_spk_tuning.keys()}
# psth_tuning, signalID_tuning = [], []
# for i in days:
#     for j in range(len(data_spk_att[i]['signal_id'])):
#         data_spk_att_i_j = {i:signal_align.select_signal(data_spk_att[i], indx=j)}
#         data_spk_tuning_i_j = {i:signal_align.select_signal(data_spk_tuning[i], indx=j)}
#         psth_tuning_i_j, _, ts_tuning, orientation_tuning = get_psth(data_spk_tuning_i_j, to_sort=['description'],
#                                                                  normalize='none',limit=limit_tuning)
#         psth_att_i_j, _, ts, orientation_att = get_psth(data_spk_att_i_j, to_sort=to_sort, normalize='none')
#
#         psth_tuning.append(psth_tuning_i_j[np.in1d(orientation_tuning,orientation_att),:])
#     signalID_tuning.append(data_spk_att[i]['signal_id'])
# psth_tuning = np.concatenate(psth_tuning,axis=2)
# signalID_tuning = np.concatenate(signalID_tuning,axis=0)

# plt.figure()
# plot_all_channels(psth_tuning, ts_tuning, [0,0.6], titles = signalID_tuning, subplots=subplots)
# plt.suptitle('PSTH: tuning mapping task')

plt.figure()
limit_match = {i:(data_spk_att[i]['trial_info']['FeatureType']==0) & (data_spk_att[i]['trial_info']['MatchNot']==1) for i in data_spk_att.keys()}
psth_match,signalID,ts,_ = get_psth(data_spk_att,to_sort=['ProbeOrientation'], normalize='none',limit=limit_match,time_window=[0.84,1.04])
plot_all_channels(psth_match, ts, [0.8,1.4], titles = signalID, subplots=subplots)
plt.suptitle('PSTH: attention task - match')

plt.figure()
limit_nonmatch = {i:(data_spk_att[i]['trial_info']['FeatureType']==0) & (data_spk_att[i]['trial_info']['MatchNot']==0) for i in data_spk_att.keys()}
psth_nonmatch,signalID,ts,_ = get_psth(data_spk_att,to_sort=['ProbeOrientation'], normalize='none',limit=limit_nonmatch)
plot_all_channels(psth_nonmatch, ts, [0.8,1.4], titles = signalID, subplots=subplots)
plt.suptitle('PSTH: Attention task - nonmatch')

# plt.figure()
# tuning_match = get_tuning(psth_match,[0.84,1.24],ts)
# plot_all_channels(tuning_match, titles = signalID, subplots=subplots)
# tuning_nonmatch = get_tuning(psth_nonmatch,[0.84,1.24],ts)
# plot_all_channels(tuning_nonmatch, titles = signalID, subplots=subplots)
# tuning_passive = get_tuning(psth_tuning,[0.04,0.34],ts_tuning)
# plot_all_channels(tuning_passive, titles = signalID, subplots=subplots)
# plt.suptitle('Tuning: attention task - match VS nonmatch VS passive')


# plt.figure()
# psth_order,signalID_tuning,ts_tuning,_ = get_psth(data_spk_tuning,to_sort=['order'], normalize='none',limit=limit_tuning)
# plot_all_channels(psth_order, ts_tuning, [-0.2,0.4], titles = signalID_tuning, subplots=(11,7))

# labels = ['{} = {}'.format(to_sort[0], conditions[0][i]) for i in range(psth.shape[0])]
# plt.figure()
# plot_mean_errorbar(psth, ts, 'se', labels)
# plt.title('n = {}'.format(signalID.shape[0]))

##
limit_match = {i:data_spk_att[i]['trial_info']['MatchNot']==1 for i in data_spk_att.keys()}
limit_nonmatch = {i:data_spk_att[i]['trial_info']['MatchNot']==0 for i in data_spk_att.keys()}
limit_orientation = {i:data_spk_att[i]['trial_info']['FeatureType']==0 for i in data_spk_att.keys()}
limit_color = {i:data_spk_att[i]['trial_info']['FeatureType']==1 for i in data_spk_att.keys()}
to_sort=['FeatureType']
limit1,limit2 = False,False
limit1_tuning,limit2_tuning = limit_orientation,limit_color
time_win_compute_corr = [0.4,0.8]
time_win_diag = [0.8,1.3]
time_win_tuning = [1.04,1.34]
diag_plot = 'tuning'

grouped_1,signalID,ts,_ = get_grouped_data(data_spk_att,to_sort=to_sort, limit=limit1)
grouped_2,signalID,ts,_ = get_grouped_data(data_spk_att,to_sort=to_sort, limit=limit2)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:len(grouped_1[0])]
corr_all, tuning_all = [], []
for i in range(7):
    limit1_i = {list(days)[i]:limit1[list(days)[i]]} if limit1 else False
    if diag_plot == 'psth':
        psth_1_i, _, ts_i, _ = get_psth({list(days)[i]:data_spk_att[list(days)[i]]}, to_sort=to_sort, normalize='none',
                                           limit=limit1_i, time_window=time_win_diag)
    elif diag_plot == 'tuning':
        psth_1_i, _, ts_i, conditions = get_psth({list(days)[i]:data_spk_att[list(days)[i]]}, to_sort=['ProbeOrientation','ProbeColor'],
                                               limit=limit1_tuning, time_window=time_win_tuning)
        tuning_1_i,axis = get_tuning(psth_1_i,conditions[0])
        psth_2_i, _, ts_i, conditions = get_psth({list(days)[i]:data_spk_att[list(days)[i]]}, to_sort=['ProbeOrientation','ProbeColor'],
                                               limit=limit2_tuning, time_window=time_win_tuning)
        tuning_2_i,axis = get_tuning(psth_2_i,conditions[0])
    corr_1,ts_corr = get_correlogram(grouped_1[i],ts_all=ts,time_win=time_win_compute_corr,smooth_std = 0,downsample_win=15)
    corr_all.append(corr_1)
    h_fig, h_ax = plt.subplots(corr_1.shape[2], corr_1.shape[3], figsize=[12,9])
    h_ax_1d = [y for x in h_ax for y in x]
    plot_all_channels(corr_1, ts=ts_corr, time_window=[-0.1, 0.1], h_ax=h_ax_1d, colors=colors)
    for j in range(corr_1.shape[2]):
        # corr_1[:,:,j,j] = None
        h_ax[j, j].set_facecolor([1., 1., 0.3])
        plt.axes(h_ax[j,j])
        plt.cla()
        if diag_plot == 'psth':
            for k in range(psth_1_i.shape[0]):
                plt.plot(ts_i,psth_1_i[k,:,j],color=colors[k])
        elif diag_plot == 'tuning':
            # plt.pcolormesh(tuning_1_i[:,:,j])
            # plt.yticks(np.arange(len(axis[0])) + 0.5, axis[0])
            # plt.axis('square')
            plt.plot(np.mean(tuning_1_i[:,:,j],axis=1),'r-')
            plt.plot(np.mean(tuning_1_i[:,:,j],axis=0),'b--')
            plt.plot(np.mean(tuning_2_i[:,:,j],axis=1),'r--')
            plt.plot(np.mean(tuning_2_i[:,:,j],axis=0),'b-')
    tuning_all.append(np.concatenate([tuning_2_i[None,:,:,:],tuning_2_i[None,:,:,:]],axis=0))


##
tuning_1,tuning_2,corr_cropped,orient_group1,orient_group2,color_group1,color_group2,orient_selectivity1,orient_selectivity2 = [],[],[],[],[],[],[],[],[]
for d in range(len(tuning_all)):
    for i in range(tuning_all[d].shape[3]):
        for j in range(tuning_all[d].shape[3]):
            tuning_1.append(tuning_all[d][:,:,:,i])
            tuning_2.append(tuning_all[d][:,:,:,j])
            corr_cropped.append(corr_all[d][:,(ts_corr>-0.05)*(ts_corr<0.05),i,j][:,:,None])
            if ((np.std(np.mean(tuning_all[d][:,:,:,i],axis=(0,1)))>1) or (np.std(np.mean(tuning_all[d][:,:,:,j],axis=(0,2)))>1)) and (np.std(np.mean(tuning_all[d][:,:,:,j],axis=(0,1)))>1):
                if np.std(np.mean(tuning_all[d][:,:,:,i],axis=(0,1)))<np.std(np.mean(tuning_all[d][:,:,:,i],axis=(0,2))):
                    orient_group1.append(1)
                    color_group1.append(0)
                else:
                    color_group1.append(1)
                    orient_group1.append(0)
                if np.std(np.mean(tuning_all[d][:,:,:,j],axis=(0,1)))<np.std(np.mean(tuning_all[d][:,:,:,j],axis=(0,2))):
                    orient_group2.append(1)
                    color_group2.append(0)
                else:
                    color_group2.append(1)
                    orient_group2.append(0)
                orient_selectivity1.append(np.std(np.mean(tuning_all[d][:,:,:,i],axis=(0,1))))
                orient_selectivity2.append(np.std(np.mean(tuning_all[d][:,:,:,j],axis=(0,1))))
            else:
                orient_group1.append(0)
                orient_group2.append(0)
                color_group1.append(0)
                orient_selectivity1.append(0)
                orient_selectivity2.append(0)
            color_group2.append(0)
orient_group1 = np.array(orient_group1)
orient_group2 = np.array(orient_group2)
corr_cropped = np.concatenate(corr_cropped,axis=2)
corr_diff = np.mean(corr_cropped[1,:,:]-corr_cropped[0,:,:],axis=0)
corr_mean = np.mean(corr_cropped,axis=(0,1))
orient_selectivity1 = np.array(orient_selectivity1)
orient_selectivity2 = np.array(orient_selectivity2)

plt.hist(corr_diff[(orient_group1==1)&(orient_group2==1)],histtype='stepfilled',fc=(1, 0, 0, 0.5),normed=True)
plt.hist(corr_diff[(orient_group1==0)&(orient_group2==0)],histtype='stepfilled',fc=(0, 0, 1, 0.5),normed=True)
# plt.hist(corr_diff[(orient_group1==0)&(orient_group2==1)],histtype='stepfilled',fc=(0, 0, 0, 0.3),normed=True)

plt.plot(orient_selectivity1*orient_selectivity2,corr_mean,'o')