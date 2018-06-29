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


class DataNeural(dict):
    def __init__(self, data_dict):
        for (key, value) in data_dict.items():
            self[key] = value
#temp0=DataNeural(temp)

signal_info_detail = pd.read_pickle('/shared/homes/sguan/Coding_Projects/support_data/spike_wf_info_Dante.pkl')
def set_signal_id(signal_info):
    signal_info['signal_id'] = signal_info['date'].apply(lambda x: '{:0>6}'.format(x)).str.cat(
        [signal_info['channel_index'].apply(lambda x: '{:0>2}'.format(x)),
        signal_info['sort_code'].apply(lambda x: '{:0>1}'.format(x))],
        sep='_'
        )
    return signal_info
signal_info_detail = set_signal_id(signal_info_detail)


dir_data_save = '/shared/homes/sguan/Coding_Projects/support_data'
block_type = 'srv_mask'
signal_type = 'spk'
hdf_file_path = '{}/all_data_dante_{}.hdf5'.format(dir_data_save, block_type)
hf = h5py.File(hdf_file_path, 'r')

date = '161023'
data_df = pd.read_json(hf[date]['trial_info_json'][()])
#data_df = pd.read_hdf(hdf_file_path, '{}/trial_info'.format(date))
data_neural = dict([])
data_neural['data'] = hf[date][signal_type]['data'][:]
data_neural['ts'] = hf[date][signal_type]['ts'][:]
data_neural['signal_id'] = hf[date][signal_type]['signal_id'][:]
data_neural['trial_info'] = data_df

list_date = hf.keys()
list_psth = []
list_signal = []
for date in list_date:
    print(date)
    data_df = pd.read_json(hf[date]['trial_info_json'][()])
    #data_df = pd.read_hdf(hdf_file_path, '{}/trial_info'.format(date))
    data_neural = dict([])
    data_neural['data'] = hf[date][signal_type]['data'][:]
    data_neural['ts'] = hf[date][signal_type]['ts'][:]
    data_neural['signal_id'] = hf[date][signal_type]['signal_id'][:]
    data_neural['trial_info'] = data_df
    data_neural = signal_align.neuro_sort(data_df, grpby=['stim_familiarized','mask_opacity_int'], neuro=data_neural)
    psth = pna.GroupAve(data_neural)
    list_psth.append(psth)
    list_signal.append(data_neural['signal_id'])

ts = data_neural['ts']
signal_all = np.concatenate(list_signal)
psth_all = np.concatenate(list_psth, axis=2)
signal_full = pd.DataFrame({'signal_id': signal_all}).merge(signal_info_detail, how='inner', on=['signal_id'], copy=False)

psth_all_smooth = pna.SmoothTrace(psth_all, sk_std=0.005, ts=ts, axis=1)
psth_plot = np.mean(psth_all_smooth[:, :, (signal_full['wf_type']=='BS') * (signal_full['area']=='TEm')], axis=2)

list_color = ['r','r','r','b','b','b']
list_ls = ['-', '-', '-', '--', '--', '--']
list_alpha = [1.0, 0.7, 0.4, 1.0, 0.7, 0.4]

for i in range(6):
    plt.plot(ts, psth_plot[i, :], color=list_color[i], ls = list_ls[i], alpha = list_alpha[i])

psth_all_smooth = pna.SmoothTrace(psth_all, sk_std=0.020, ts=ts, axis=1)
psth_plot = psth_all_smooth[2][:, (signal_full['wf_type']=='BS') * (signal_full['area']=='TEm')]
list_color = pnp.gen_distinct_colors(psth_plot.shape[1])
for i in range(psth_plot.shape[1]):
    plt.plot(ts, psth_plot[:,i], color=list_color[i])
plt.title('TEm_BS_by_cells')
plt.savefig('./temp_figs/TEm_BS_by_cells')



# save fig for every neuron
list_date = hf.keys()
list_psth = []
list_signal = []
for date in list_date:
    print(date)
    data_df = pd.read_hdf(hdf_file_path, '{}/trial_info'.format(date))
    data_neural = dict([])
    data_neural['data'] = hf[date][signal_type]['data'][:]
    data_neural['ts'] = hf[date][signal_type]['ts'][:]
    data_neural['signal_id'] = hf[date][signal_type]['signal_id'][:]
    data_neural['trial_info'] = data_df
    data_neural = signal_align.neuro_sort(data_df, grpby=['stim_sname'], neuro=data_neural)
    sname_unq = np.unique(data_df['stim_sname'])
    signal_full = pd.DataFrame({'signal_id': data_neural['signal_id']}).merge(signal_info_detail, how='inner', on=['signal_id'],
                                                                copy=False)
    for j in range(len(signal_full)):
        h_fig, h_ax = plt.subplots(5, 4, sharex='all', sharey='all', figsize=[10,8])
        plt.suptitle('{}_{}_{}'.format(signal_full['signal_id'][j], signal_full['area'][j], signal_full['wf_type'][j]))
        h_ax = h_ax.ravel()
        for p in range(20):
            try:
                sname = sname_unq[p]
                plt.axes(h_ax[p])
                pnp.PsthPlot(data_neural['data'][:,:,j], ts=ts, cdtn=data_df['mask_opacity_int'], limit=data_df['stim_sname']==sname, sk_std=0.020)
                plt.title(sname)
            except:
                pass
        plt.savefig('./temp_figs/{}_{}_{}.png'.format(signal_full['signal_id'][j], signal_full['area'][j], signal_full['wf_type'][j]))
        plt.close('all')
    # list_psth.append(psth)
    # list_signal.append(data_neural['signal_id'])
