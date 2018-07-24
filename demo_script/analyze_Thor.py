""" script to generate the final result figures """

import importlib
import os
import warnings
import numpy as np
import scipy as sp
import pandas as pd
import h5py
import matplotlib as mlp
import matplotlib.pyplot as plt

import PyNeuroData as pnd
import PyNeuroAna  as pna
import PyNeuroPlot as pnp
import store_hdf5
import df_ana
import misc_tools


# set path to data
path_to_hdf5_data = '../support_data/data_neuro_Thor_U16_all.hdf5'
path_to_fig = './temp_figs'


##
"""========== get PSTH for every day =========="""

# shwo data file stucture
store_hdf5.ShowH5(path_to_hdf5_data)

tree_struct_hdf5 = store_hdf5.ShowH5(path_to_hdf5_data, yn_print=False)
list_psth_group_mean = []
list_psth_group_std = []
list_signal = []
sk_std = 0.010

""" get every day's data """
for datecode in tree_struct_hdf5.keys():
    print(datecode)
    try:
        # get data
        data_neuro = store_hdf5.LoadFromH5(path_to_hdf5_data, [datecode, 'srv_mask', 'spk'])
        df_ana.GroupDataNeuro(data_neuro, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                              groupby=['stim_familiarized', 'mask_opacity_int'])

        # smooth trace over time
        data_neuro['data'] = pna.SmoothTrace(data_neuro['data'], ts=data_neuro['ts'], sk_std=sk_std)

        # get mean and std
        psth_group_mean = pna.GroupStat(data_neuro)
        psth_group_std = pna.GroupStat(data_neuro, statfun='std')

        # get signal info
        signal_info = data_neuro['signal_info']
        signal_info['date'] = datecode

        list_psth_group_mean.append(psth_group_mean)
        list_psth_group_std.append(psth_group_std)
        list_signal.append(signal_info)
    except:
        print('date {} can not be processed'.format(datecode))

##
""" concatenate together """
data_group_mean = np.concatenate(list_psth_group_mean, axis=2)
data_group_std = np.concatenate(list_psth_group_std, axis=2)
signal_all = pd.concat(list_signal).reset_index()
ts = data_neuro['ts']
cdtn = data_neuro['cdtn']

def set_signal_id(signal_info):
    signal_info['signal_id'] = signal_info['date'].apply(lambda x: '{}'.format(x)).str.cat(
        [signal_info['channel_index'].apply(lambda x: '{:0>2}'.format(x)),
        signal_info['sort_code'].apply(lambda x: '{:0>1}'.format(x))],
        sep='_'
        )
    return signal_info
signal_all = set_signal_id(signal_all)


##
""" plot overall data """
colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9, style='continuous', cm='rainbow'),
                    pnp.gen_distinct_colors(3, luminance=0.7, style='continuous', cm='rainbow')])
linestyles = ['--', '--', '--', '-', '-', '-']
cdtn_name = ['nov, 00', 'nov, 50', 'nov, 70', 'fam, 00', 'fam, 50', 'fam, 70']

plot_highlight = dict()
plot_highlight['all'] = {'trace': [1,1,1,1,1,1], 'compare': [0,0,0,1,0]}
plot_highlight['nov'] = {'trace': [1,1,1,0,0,0], 'compare': [0,0,0,1,0]}
plot_highlight['fam'] = {'trace': [0,0,0,1,1,1], 'compare': [0,0,0,0,1]}
plot_highlight['00']  = {'trace': [1,0,0,1,0,0], 'compare': [1,0,0,0,0]}
plot_highlight['50']  = {'trace': [0,1,0,0,1,0], 'compare': [0,1,0,0,0]}
plot_highlight['70']  = {'trace': [0,0,1,0,0,1], 'compare': [0,0,1,0,0]}

h_fig, h_axes = plt.subplots(2,3, figsize=(12,8))
h_axes = np.ravel(h_axes)
for i, highlight in enumerate(plot_highlight):
    plt.axes(h_axes[i])
    plt.title(highlight)
    for c in range(len(cdtn)):
        plt.plot(ts, data_group_mean[c].mean(axis=1),
                 color=colors[c], linestyle=linestyles[c],
                 alpha=plot_highlight[highlight]['trace'][c])
    if i==0:
        plt.legend(cdtn_name)

##
""" save every signal's plot to disk """
if False:
    colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9, style='continuous', cm='rainbow'),
                        pnp.gen_distinct_colors(3, luminance=0.7, style='continuous', cm='rainbow')])
    linestyles = ['--', '--', '--', '-', '-', '-']
    cdtn_name = ['nov, 00', 'nov, 50', 'nov, 70', 'fam, 00', 'fam, 50', 'fam, 70']

    plot_highlight = dict()
    plot_highlight['all'] = {'trace': [1, 1, 1, 1, 1, 1], 'compare': [0, 0, 0, 1, 0]}
    plot_highlight['nov'] = {'trace': [1, 1, 1, 0, 0, 0], 'compare': [0, 0, 0, 1, 0]}
    plot_highlight['fam'] = {'trace': [0, 0, 0, 1, 1, 1], 'compare': [0, 0, 0, 0, 1]}
    plot_highlight['00'] = {'trace': [1, 0, 0, 1, 0, 0], 'compare': [1, 0, 0, 0, 0]}
    plot_highlight['50'] = {'trace': [0, 1, 0, 0, 1, 0], 'compare': [0, 1, 0, 0, 0]}
    plot_highlight['70'] = {'trace': [0, 0, 1, 0, 0, 1], 'compare': [0, 0, 1, 0, 0]}

    for i_signal, signal_id_cur in enumerate(signal_all['signal_id']):
        h_fig, h_axes = plt.subplots(2, 3, figsize=(12, 8))
        h_axes = np.ravel(h_axes)
        plt.suptitle(signal_id_cur)
        for i, highlight in enumerate(plot_highlight):
            plt.axes(h_axes[i])
            plt.title(highlight)
            for c in range(len(cdtn)):
                plt.plot(ts, data_group_mean[c, :, i_signal],
                         color=colors[c], linestyle=linestyles[c],
                         alpha=plot_highlight[highlight]['trace'][c])
            if i == 0:
                plt.legend(cdtn_name)
        plt.savefig(os.path.join(path_to_fig, 'final_psth_svr_mask_spk_thor', 'psth_{}.png'.format(signal_id_cur)))
        plt.savefig(os.path.join(path_to_fig, 'final_psth_svr_mask_spk_thor', 'psth_{}.pdf'.format(signal_id_cur)))
        plt.close('all')


##
"""========== CSD analysis =========="""

list_lfp_profile = []
list_signal = []
""" get every day's ERP """
for datecode in tree_struct_hdf5.keys():
    print(datecode)
    try:
        # get data
        data_neuro = store_hdf5.LoadFromH5(path_to_hdf5_data, [datecode, 'srv_mask', 'lfp'])

        # lfp profiel: [number_channels, num_ts]
        lfp_profile = np.mean(data_neuro['data'][data_neuro['trial_info']['mask_opacity_int']==0], axis=0).transpose()

        signal_info = data_neuro['signal_info']
        signal_info['date'] = datecode

        list_lfp_profile.append(lfp_profile)
        list_signal.append(signal_info)

    except:
        print('date {} can not be processed'.format(datecode))

##
""" analyze CSD """
chan_bad_all = dict()

