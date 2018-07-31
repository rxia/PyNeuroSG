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

signal_all_spk = signal_all

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

yn_keep_signal = np.ones(data_group_mean.shape[2]).astype('bool')

for i, highlight in enumerate(plot_highlight):
    plt.axes(h_axes[i])
    plt.title(highlight)
    for c in range(len(cdtn_name)):
        plt.plot(ts, data_group_mean[c][:, yn_keep_signal].mean(axis=1),
                 color=colors[c], linestyle=linestyles[c],
                 alpha=plot_highlight[highlight]['trace'][c])
    if i==0:
        plt.legend(cdtn_name)


##
""" test whether a neuron is visual or not """

psth_mean_00_noise = data_group_mean[[0,3]].mean(axis=0)
psth_var_00_noise = data_group_mean[[0,3]].mean(axis=0)

psth_mean_70_noise = data_group_mean[[2,5]].mean(axis=0)
psth_var_70_noise = data_group_mean[[2,5]].mean(axis=0)

ts_baseline = np.logical_and(ts>=-0.050, ts<0.050)
ts_visual = np.logical_and(ts>=0.050, ts<0.350)

mean_baseline = psth_mean_00_noise[ts_baseline].mean(axis=0)
std_baseline = psth_mean_00_noise[ts_baseline].mean(axis=0)
mean_visual_00_noise = psth_mean_00_noise[ts_visual].mean(axis=0)
std_visual_00_noise = psth_mean_00_noise[ts_visual].mean(axis=0)
mean_visual_70_noise = psth_mean_70_noise[ts_visual].mean(axis=0)
std_visual_70_noise = psth_var_70_noise[ts_visual].mean(axis=0)

threhold_cohen_d = 0.4

cohen_d_visual = (mean_visual_00_noise - mean_baseline) \
                 / np.sqrt((std_baseline**2 + std_visual_00_noise**2)/2)
keep_visual = cohen_d_visual > threhold_cohen_d

cohen_d_noise_effect = (mean_visual_00_noise - mean_visual_70_noise) \
                       / np.sqrt((std_visual_00_noise**2 + std_visual_70_noise**2)/2)
keep_noise_effect = cohen_d_noise_effect > 0.2

plt.subplot(1,2,1)
plt.scatter(mean_baseline, mean_visual_00_noise, c=cohen_d_visual, alpha=0.7)
plt.scatter(mean_baseline[~keep_visual], mean_visual_00_noise[~keep_visual], c='k')
plt.plot([0, 50], [0, 50])
plt.axis('equal')
plt.xlabel('baseline')
plt.ylabel('visual')
plt.title('visual/all, {}/{}'.format(np.sum(keep_visual), len(keep_visual)))


plt.subplot(1,2,2)
plt.scatter(mean_visual_00_noise, mean_visual_70_noise, c=cohen_d_noise_effect, alpha=0.7)
plt.scatter(mean_visual_00_noise[~(keep_noise_effect & keep_visual)],
            mean_visual_70_noise[~(keep_noise_effect & keep_visual)], c='k', alpha=0.7, )
plt.plot([0, 50], [0, 50])
plt.axis('equal')
plt.xlabel('no noise')
plt.ylabel('70 noise')
plt.title('visual/all, {}/{}'.format(np.sum(keep_visual & keep_noise_effect), len(keep_visual)))

keep_final = (keep_visual & keep_noise_effect)

if True:
    path_to_U_probe_manual_selection = './script/Thor_signal_manual_select.csv'
    keep_final_manual = pd.read_csv(path_to_U_probe_manual_selection)
    keep_final_manual = keep_final_manual['good_noise_effect'] == '1'

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
""" read file of U-probe location """
path_to_U_probe_loc_Thor = './script/Thor_U16_location.csv'

# granular layer location is zero-indexed
loc_U16 = pd.read_csv(path_to_U_probe_loc_Thor)
loc_U16['date'] = loc_U16['date'].astype('str')

signal_extend = pd.merge(signal_all, loc_U16, 'left', on='date')

signal_extend['depth'] = 0
signal_extend['depth'] = (signal_extend['channel_index']-1-signal_extend['granular']) * (signal_extend['area']=='TEd') \
                         +(16-signal_extend['channel_index']-signal_extend['granular']) * (signal_extend['area']=='TEm')


##
""" ========== distribution of neural latency ========== """
time_focus = np.logical_and(ts>=0.05, ts<0.250)
indx_cdtn = 3

latency_method = 'mean'

mean_latency = np.sum(data_group_mean[:, time_focus, :] * ts[None, time_focus, None], axis=1)\
       / np.sum(data_group_mean[:, time_focus, :], axis=1)
var_latency = np.sum(data_group_mean[:, time_focus, :] * ts[None, time_focus, None]**2, axis=1) \
               / np.sum(data_group_mean[:, time_focus, :], axis=1) \
              - mean_latency **2
std_latency = np.sqrt(var_latency)

cond_compare = [3, 5]
effect_size = (mean_latency[cond_compare[1], :] - mean_latency[cond_compare[0], :]) \
              / np.sqrt((var_latency[cond_compare[0]] + var_latency[cond_compare[1]])/2)
threhold_cohen_d = 0.4

keep_noise_delay = effect_size > threhold_cohen_d

plt.scatter(mean_latency[3, :], mean_latency[5, :])
plt.scatter(mean_latency[3, ~keep_noise_delay], mean_latency[5, ~keep_noise_delay], c='k')
pnp.plot_diagonal_line(mean_latency[3, :])
plt.xlabel('fam, no noise')
plt.ylabel('fam, 70 noise')
plt.suptitle('mean latency, valid = {}/{}'.format(np.sum(keep_noise_delay), len(keep_noise_delay)))


signal_extend['delay_effect'] = effect_size

df_ana.DfPlot(signal_extend, 'delay_effect', x='depth', p='area', plot_type='box',
              limit=(signal_extend['depth']==signal_extend['depth']) & keep_visual==True)


##
""" ========== plot overall data ========== """
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

h_fig, h_axes = plt.subplots(2, 3, figsize=(12,8))
h_axes = np.ravel(h_axes)

mode_keep_signal = ''
# mode_keep_signal = 'noise_delay'
if mode_keep_signal == 'manual':
    yn_keep_signal = keep_final_manual
elif mode_keep_signal == 'noise_delay':
    yn_keep_signal = keep_noise_delay
else:
    yn_keep_signal = np.ones(data_group_mean.shape[2]).astype('bool')

for i, highlight in enumerate(plot_highlight):
    plt.axes(h_axes[i])
    plt.title(highlight)
    for c in range(len(cdtn_name)):
        plt.plot(ts, data_group_mean[c][:, yn_keep_signal].mean(axis=1),
                 color=colors[c], linestyle=linestyles[c],
                 alpha=plot_highlight[highlight]['trace'][c])
    if i==0:
        plt.legend(cdtn_name)



##
"""========== analysis of tuning vs dynamics =========="""

list_psth_group_mean = []
t_window = [0.050, 0.300]
# list_psth_group_std = []
list_signal = []
list_cdtn = []
sk_std = 0.010

t_for_fr = (ts>=0.050) & (ts<0.350)

for datecode in tree_struct_hdf5.keys():
    print(datecode)
    try:
        # get data
        data_neuro = store_hdf5.LoadFromH5(path_to_hdf5_data, [datecode, 'srv_mask', 'spk'])
        df_ana.GroupDataNeuro(data_neuro, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                              groupby=['stim_familiarized', 'mask_opacity_int', 'stim_sname'])

        # smooth trace over time
        data_neuro['data'] = pna.SmoothTrace(data_neuro['data'], ts=data_neuro['ts'], sk_std=sk_std)
        ts = data_neuro['ts']

        # get mean and std
        psth_group_mean = pna.GroupStat(data_neuro)

        # sort stims by rank preference
        # cdtn = pd.DataFrame(data_neuro['cdtn'], columns=['stim_familiarized', 'mask_opacity_int', 'stim_sname'])

        psth_group_reshape = np.reshape(psth_group_mean, [2, 3, 10, data_neuro['data'].shape[1], data_neuro['data'].shape[2]])

        psth_rank = psth_group_reshape * 0
        for i_signal in range(data_neuro['data'].shape[2]):
            image_rank_nov = np.argsort(np.mean(psth_group_reshape[0, 0, :, :, i_signal][:, t_for_fr], axis=1))[::-1]
            image_rank_fam = np.argsort(np.mean(psth_group_reshape[1, 0, :, :, i_signal][:, t_for_fr], axis=1))[::-1]
            psth_rank[0, :, :, :, i_signal] = psth_group_reshape[0, :, :, :, i_signal][:, image_rank_nov, :]
            psth_rank[1, :, :, :, i_signal] = psth_group_reshape[1, :, :, :, i_signal][:, image_rank_fam, :]

        # get signal info
        signal_info = data_neuro['signal_info']
        signal_info['date'] = datecode

        list_psth_group_mean.append(psth_rank)
        list_signal.append(signal_info)
    except:
        print('date {} can not be processed'.format(datecode))


##

# mode_keep_signal = ''
mode_keep_signal = 'noise_delay'
if mode_keep_signal == 'manual':
    yn_keep_signal = keep_final_manual
elif mode_keep_signal == 'noise_delay':
    yn_keep_signal = keep_noise_delay
else:
    yn_keep_signal = np.ones(data_group_mean.shape[2]).astype('bool')

psth_rank_all = np.concatenate(list_psth_group_mean, axis=-1)

psth_rank_norm = pna.normalize_across_signals(psth_rank_all, ts=ts, t_window=[0.050, 0.300])


list_c = pnp.gen_distinct_colors(10, style='continuous')
psth_rank_mean = np.mean(psth_rank_norm[:, :, :, :, yn_keep_signal], axis=4)


h_fig, h_axes = plt.subplots(2, 3, sharex='all', sharey='all')
for i_r in range(2):
    for i_c in range(3):
        plt.axes(h_axes[i_r, i_c])
        for i in range(10):
            plt.plot(ts, psth_rank_mean[i_r, i_c, i, :], c=list_c[i])
plt.xlim([-0.05, 0.5])


h_fig, h_axes = plt.subplots(2, 3, sharex='all', sharey='all')
for i_r in range(2):
    for i_c in range(3):
        plt.axes(h_axes[i_r, i_c])
        for i in range(10):
            displacement = 10*np.mean(psth_rank_mean[i_r, i_c, i, :][(ts>=t_window[0]) & (ts<t_window[1])])
            plt.plot(ts, psth_rank_mean[i_r, i_c, i, :] + displacement, c=list_c[i])
plt.xlim([-0.05, 0.5])
plt.ylim([4, 10])


##
""" plot overall data with only prefered image """

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

h_fig, h_axes = plt.subplots(2, 3, figsize=(12,8))
h_axes = np.ravel(h_axes)

# mode_keep_signal = ''
mode_keep_signal = 'noise_delay'
if mode_keep_signal == 'manual':
    yn_keep_signal = keep_final_manual
elif mode_keep_signal == 'noise_delay':
    yn_keep_signal = keep_noise_delay
else:
    yn_keep_signal = np.ones(data_group_mean.shape[2]).astype('bool')

for i, highlight in enumerate(plot_highlight):
    plt.axes(h_axes[i])
    plt.title(highlight)
    for c in range(len(cdtn_name)):
        plt.plot(ts, data_group_mean[c][:, yn_keep_signal].mean(axis=1),
                 color=colors[c], linestyle=linestyles[c],
                 alpha=plot_highlight[highlight]['trace'][c])
    if i==0:
        plt.legend(cdtn_name)



##
""" ========== plot tuning curve by condition ========== """
t_window = [0.050, 0.150]
t_for_fr = (ts >= 0.050) & (ts < 0.300)

mode_keep_signal = 'noise_delay'
# mode_keep_signal = 'noise_delay'
if mode_keep_signal == 'manual':
    yn_keep_signal = keep_final_manual
elif mode_keep_signal == 'noise_delay':
    yn_keep_signal = keep_noise_delay
else:
    yn_keep_signal = np.ones(data_group_mean.shape[2]).astype('bool')

tuning_curve = np.mean(psth_rank_norm[:, :, :, t_for_fr, :], axis=3)
tuning_curve_mean = np.mean(tuning_curve[:, :, :, yn_keep_signal], axis=-1)
tuning_curve_std  = np.std(tuning_curve[:, :, :, yn_keep_signal], axis=-1)/np.sqrt(np.sum(yn_keep_signal))

colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9, style='continuous', cm='rainbow'),
                    pnp.gen_distinct_colors(3, luminance=0.7, style='continuous', cm='rainbow')])
linestyles = ['--', '--', '--', '-', '-', '-']
cdtn_name = ['nov, 00', 'nov, 50', 'nov, 70', 'fam, 00', 'fam, 50', 'fam, 70']

h_fig, h_axes = plt.subplots(1, 3, sharex='all', sharey='all')

plt.axes(h_axes[0])
for i_r in range(2):
    for i_c in range(3):
        i = i_r * 3 + i_c
        plt.fill_between(range(10),
                         tuning_curve_mean[i_r, i_c, :] - tuning_curve_std[i_r, i_c, :],
                         tuning_curve_mean[i_r, i_c, :] + tuning_curve_std[i_r, i_c, :],
                         color=colors[i], linestyle=linestyles[i], alpha=0.2)
        plt.plot(tuning_curve_mean[i_r, i_c, :], c=colors[i], linestyle=linestyles[i], label=cdtn_name[i])
plt.legend()

plt.axes(h_axes[1])
i_r = 0
for i_c in range(3):
    i = i_r * 3 + i_c
    plt.fill_between(range(10),
                     tuning_curve_mean[i_r, i_c, :] - tuning_curve_std[i_r, i_c, :],
                     tuning_curve_mean[i_r, i_c, :] + tuning_curve_std[i_r, i_c, :],
                     color=colors[i], linestyle=linestyles[i], alpha=0.2)
    plt.plot(tuning_curve_mean[i_r, i_c, :], c=colors[i], linestyle=linestyles[i], label=cdtn_name[i])

plt.axes(h_axes[2])
i_r = 1
for i_c in range(3):
    i = i_r * 3 + i_c
    plt.fill_between(range(10),
                     tuning_curve_mean[i_r, i_c, :] - tuning_curve_std[i_r, i_c, :],
                     tuning_curve_mean[i_r, i_c, :] + tuning_curve_std[i_r, i_c, :],
                     color=colors[i], linestyle=linestyles[i], alpha=0.2)
    plt.plot(tuning_curve_mean[i_r, i_c, :], c=colors[i], linestyle=linestyles[i], label=cdtn_name[i])


# h_fig, h_axes = plt.subplots(2, 3, sharex='all', sharey='all')
# for i_r in range(2):
#     for i_c in range(3):
#         plt.axes(h_axes[i_r, i_c])
#         plt.plot(tuning_curve_mean[i_r, i_c, :])



##
""" ========== cross-trial var vs mean ========== """
list_psth_group_mean = []
list_psth_group_std  = []
window_size = 0.100
# list_psth_group_std = []
list_signal = []
list_cdtn = []
sk_std = 0.030

t_for_fr = (ts>=0.050) & (ts<0.350)

for datecode in tree_struct_hdf5.keys():
    print(datecode)
    try:
        # get data
        data_neuro = store_hdf5.LoadFromH5(path_to_hdf5_data, [datecode, 'srv_mask', 'spk'])
        df_ana.GroupDataNeuro(data_neuro, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                              groupby=['stim_familiarized', 'mask_opacity_int', 'stim_sname'])

        # smooth trace over time
        ts = data_neuro['ts']
        data_neuro['data'] = pna.SpikeCountInWindow(data_neuro['data'], window_size, ts=ts)


        # get mean and std
        psth_group_mean = pna.GroupStat(data_neuro)
        psth_group_std  = pna.GroupStat(data_neuro, statfun='std')

        psth_group_reshape = np.reshape(psth_group_mean, [2, 3, 10, data_neuro['data'].shape[1], data_neuro['data'].shape[2]])
        psth_group_reshape_std = np.reshape(psth_group_std,
                                        [2, 3, 10, data_neuro['data'].shape[1], data_neuro['data'].shape[2]])

        psth_rank = psth_group_reshape * 0
        psth_rank_std = psth_group_reshape * 0
        for i_signal in range(data_neuro['data'].shape[2]):
            image_rank_nov = np.argsort(np.mean(psth_group_reshape[0, 0, :, :, i_signal][:, t_for_fr], axis=1))[::-1]
            image_rank_fam = np.argsort(np.mean(psth_group_reshape[1, 0, :, :, i_signal][:, t_for_fr], axis=1))[::-1]

            psth_rank[0, :, :, :, i_signal] = psth_group_reshape[0, :, :, :, i_signal][:, image_rank_nov, :]
            psth_rank[1, :, :, :, i_signal] = psth_group_reshape[1, :, :, :, i_signal][:, image_rank_fam, :]

            psth_rank_std[0, :, :, :, i_signal] = psth_group_reshape_std[0, :, :, :, i_signal][:, image_rank_nov, :]
            psth_rank_std[1, :, :, :, i_signal] = psth_group_reshape_std[1, :, :, :, i_signal][:, image_rank_fam, :]

        # get signal info
        signal_info = data_neuro['signal_info']
        signal_info['date'] = datecode

        list_psth_group_mean.append(psth_rank)
        list_psth_group_std.append(psth_rank_std)
        list_signal.append(signal_info)
    except:
        print('date {} can not be processed'.format(datecode))


##
mode_keep_signal = ''
# mode_keep_signal = 'noise_delay'
if mode_keep_signal == 'manual':
    yn_keep_signal = keep_final_manual
elif mode_keep_signal == 'noise_delay':
    yn_keep_signal = keep_noise_delay
else:
    yn_keep_signal = np.ones(data_group_mean.shape[2]).astype('bool')

psth_rank_all = np.concatenate(list_psth_group_mean, axis=-1)[:, :, :, :, yn_keep_signal]
psth_rank_all_std = np.concatenate(list_psth_group_std, axis=-1)[:, :, :, :, yn_keep_signal]


sampling_rate = 1/np.mean(np.diff(ts))

t_interest = np.flatnonzero(ts>0.100)[0]

h_fig, h_axes = plt.subplots(2, 3, sharex='all', sharey='all')
for i_r in range(2):
    for i_c in range(3):
        plt.axes(h_axes[i_r, i_c])
        x = psth_rank_all[i_r, i_c, :2, t_interest, :].ravel()
        y = psth_rank_all_std[i_r, i_c, :2, t_interest, :].ravel()**2
        res_regress = sp.stats.linregress(x, y)
        k, b = res_regress[:2]

        plt.plot(x, y, 'o')
        plt.plot([0, 10], [b, 10*k+b])
        plt.title('{:.2f}'.format(k))





##
"""========== CSD analysis =========="""

dict_lfp_profile = dict()
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

        dict_lfp_profile[datecode] = lfp_profile
    except:
        print('date {} can not be processed'.format(datecode))

##
yn_plot_spike_quality = True
yn_plot_granular_loc = True


""" analyze CSD """
chan_bad_all = dict()
chan_bad_all['180224'] = [2, 8]
chan_bad_all['180317'] = [0, 1, 2, 12]
chan_bad_all['180325'] = [2, 12]
chan_bad_all['180407'] = [2, 12, 13]
chan_bad_all['180411'] = [2, 8, 12, 13]
chan_bad_all['180413'] = [2, 8, 12, 13]
chan_bad_all['180418'] = [2, 8, 12, 13]
chan_bad_all['180420'] = [2, 8, 12, 13]
chan_bad_all['180424'] = [2, 8, 12, 13]
chan_bad_all['180501'] = []
chan_bad_all['180515'] = [13]
chan_bad_all['180520'] = [7, 13]
chan_bad_all['180523'] = [2, 7]
chan_bad_all['180530'] = [2, 5, 7, 13]
chan_bad_all['180603'] = [2, 5, 7, 13]
chan_bad_all['180606'] = [2, 5, 7, 13]
chan_bad_all['180610'] = [2, 5, 7, 13]
chan_bad_all['180614'] = [2, 5, 7, 13]
chan_bad_all['180617'] = [2, 5, 7, 13]
chan_bad_all['180620'] = [2, 5, 7, 13]
chan_bad_all['180622'] = [2, 5, 7, 13]
chan_bad_all['180624'] = [2, 5, 7, 13]


plt.ioff()

if yn_plot_spike_quality:
    # get spike info for channels
    signal_all_best_spk = signal_all_spk.groupby(['date', 'channel_index']).last()\
        .reset_index()[['date', 'channel_index', 'sort_code']]

    N_chan_total = 16

    def get_spike_quality_for_channel(datecode):
        """ get the signal quality, i.e, largest sortcode for every channel for a day """
        signal_quality = pd.merge(pd.DataFrame({'channel_index': range(N_chan_total)}),
                                   signal_all_best_spk[signal_all_best_spk['date']==datecode], 'left', on='channel_index')
        signal_quality = np.nan_to_num(np.array(signal_quality['sort_code']), 0).astype('int')
        return signal_quality

if yn_plot_granular_loc:
    path_to_U_probe_loc_Thor = './script/Thor_U16_location.csv'
    loc_U16 = pd.read_csv(path_to_U_probe_loc_Thor)
    loc_U16['date'] = loc_U16['date'].astype('str')
    def get_granular_loc(datecode):
        return loc_U16['granular'][loc_U16['date']==datecode]


""" plot CSD for every day """
for datecode in dict_lfp_profile:
    print(datecode)

    lfp = dict_lfp_profile[datecode]
    if datecode in chan_bad_all:
        chan_bad = chan_bad_all[datecode]
    else:
        chan_bad = []
    lambda_dev = np.ones(16)
    lambda_dev[chan_bad]=0

    _, h_axes = plt.subplots(2,3, figsize=[12,8], sharex='all', sharey='all')
    plt.axes(h_axes[0,0])
    pnp.ErpPlot_singlePanel(lfp, ts)
    plt.plot([0]*len(chan_bad), chan_bad, 'ok')
    plt.title('LFP original')
    plt.xlabel('time (s)')
    plt.ylabel('chan')
    plt.xticks(np.arange(-0.1,0.51,0.1))


    lfp_na = lfp
    lfp_sm = pna.lfp_cross_chan_smooth(lfp, method='der', lambda_dev=lambda_dev, lambda_der=5, sigma_t=0.5)
    lfp_nr = lfp_sm / np.std(lfp_sm, axis=1, keepdims=True)

    csd_na = pna.cal_1dCSD(lfp_na, axis_ch=0, tf_edge=True)
    csd_sm = pna.cal_1dCSD(lfp_sm, axis_ch=0, tf_edge=True)
    csd_nr = pna.cal_1dCSD(lfp_nr, axis_ch=0, tf_edge=True)

    plt.axes(h_axes[0, 1])
    pnp.ErpPlot_singlePanel(lfp_sm, ts)
    plt.title('LFP smoothed')

    if yn_plot_spike_quality:
        spike_quanlity = get_spike_quality_for_channel(datecode)
        plt.scatter(np.repeat(ts[0],N_chan_total), np.arange(N_chan_total),
                    c=spike_quanlity, vmin=-3, vmax=3, cmap='Spectral', edgecolors='k', s=100)
    if yn_plot_granular_loc:
        plt.plot(ts[0], get_granular_loc(datecode), 'k+')

    plt.axes(h_axes[0, 2])
    pnp.ErpPlot_singlePanel(lfp_nr, ts)
    plt.title('LFP normalized')

    plt.axes(h_axes[1, 0])
    pnp.ErpPlot_singlePanel(csd_na, ts, tf_inverse_color=True)
    plt.title('CSD native')
    plt.xlabel('time (s)')
    plt.ylabel('chan')

    plt.axes(h_axes[1, 1])
    pnp.ErpPlot_singlePanel(csd_sm, ts, tf_inverse_color=True)
    plt.title('CSD smoothed')

    if yn_plot_spike_quality:
        spike_quanlity = get_spike_quality_for_channel(datecode)
        plt.scatter(np.repeat(ts[0],N_chan_total), np.arange(N_chan_total),
                    c=spike_quanlity, vmin=-3, vmax=3, cmap='Spectral', edgecolors='k', s=100)
    if yn_plot_granular_loc:
        plt.plot(ts[0], get_granular_loc(datecode), 'k+')

    plt.axes(h_axes[1, 2])
    pnp.ErpPlot_singlePanel(csd_nr, ts, tf_inverse_color=True)
    plt.title('CSD normalized')

    plt.suptitle(datecode)

    plt.savefig(os.path.join(path_to_fig, 'final_csd_thor', 'csd_{}.png'.format(datecode)))
    plt.savefig(os.path.join(path_to_fig, 'final_csd_thor', 'csd_{}.pdf'.format(datecode)))
    plt.close('all')










