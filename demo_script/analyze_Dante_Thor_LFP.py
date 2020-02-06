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
import pickle

import PyNeuroData as pnd
import PyNeuroAna  as pna
import PyNeuroPlot as pnp
import store_hdf5
import df_ana
import misc_tools


# set path to data
path_to_hdf5_data_Dante = '../support_data/data_neuro_Dante_GM32_U16_all.hdf5'
path_to_hdf5_data_Thor = '../support_data/data_neuro_Thor_U16_all.hdf5'
path_to_fig = './temp_figs/final_Dante_Thor'


path_to_U_probe_loc = './script/U16_location_Dante_Thor.csv'

##
""" get U-probe location """
# granular layer location is zero-indexed
loc_U16 = pd.read_csv(path_to_U_probe_loc)
loc_U16['date'] = loc_U16['date'].astype('str')

def sinal_extend_loc_info(signal_all):
    signal_extend = pd.merge(signal_all, loc_U16, 'left', on='date')

    signal_extend['channel_index_U16'] = signal_extend['channel_index'] - 32*(signal_extend['animal']=='Dante')
    signal_extend['area'] = np.where((signal_extend['animal']=='Dante') & (signal_extend['channel_index']<=32),
                                     ['V4']*len(signal_extend), signal_extend['area'])

    signal_extend['depth'] = 0
    signal_extend['depth'] = (signal_extend['channel_index_U16']-1-signal_extend['granular']) * (signal_extend['area']=='TEd') \
                             +(16-signal_extend['channel_index_U16']-signal_extend['granular']) * (signal_extend['area']=='TEm')
    signal_extend['depth'] = np.where(signal_extend['area']=='V4',
                                     np.zeros(len(signal_extend))*np.nan, signal_extend['depth'])

    return signal_extend


def load_data_of_day(datecode):
    if datecode <= '180000':
        animal = 'Dante'
        path_to_hdf5_data = path_to_hdf5_data_Dante
    else:
        animal = 'Thor'
        path_to_hdf5_data = path_to_hdf5_data_Thor
    data_neuro = store_hdf5.LoadFromH5(path_to_hdf5_data, [datecode, 'srv_mask', 'lfp'])
    return data_neuro


##
"""========== get PSTH for every day =========="""

# shwo data file stucture

tree_struct_hdf5_Dante = store_hdf5.ShowH5(path_to_hdf5_data_Dante, yn_print=False)
tree_struct_hdf5_Thor  = store_hdf5.ShowH5(path_to_hdf5_data_Thor , yn_print=False)
all_dates = list(tree_struct_hdf5_Dante.keys()) + list(tree_struct_hdf5_Thor.keys())
list_psth_group_mean = []
list_psth_group_std = []
list_signal = []
sk_std = 0.010

""" get every day's data """
for datecode in all_dates:
    print(datecode)
    try:
        # get data
        data_neuro = load_data_of_day(datecode)
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

signal_all_lfp = signal_all

signal_all_lfp = sinal_extend_loc_info(signal_all_lfp)

# data_group_mean_norm = pna.normalize_across_signals(data_group_mean, ts=ts, t_window=[0.050, 0.300])
data_group_mean_norm = data_group_mean / data_group_mean.std(axis=1)[np.array([0, 3])].mean(axis=0)[None, None, :]


##
""" count neuorns """
temp = (signal_all_lfp['valid']==1) & (signal_all_lfp['animal']=='Dante') & (signal_all_lfp['area']=='V4')
print(np.sum(temp & (signal_all_lfp['sort_code']==1)))

temp = (signal_all_lfp['valid']==1) & (signal_all_lfp['animal']=='Dante') & (signal_all_lfp['area']!='V4')
print(np.sum(temp))
print(np.sum(temp & (signal_all_lfp['sort_code']==1)))

temp = (signal_all_lfp['valid']==1) & (signal_all_lfp['animal']=='Thor') & (signal_all_lfp['area']!='V4')
print(np.sum(temp))
print(np.sum(temp & (signal_all_lfp['sort_code']>1)))


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

# yn_keep_signal = np.ones(data_group_mean.shape[2]).astype('bool')
yn_keep_signal = signal_all_lfp['area'] == 'V4'
h_fig, h_axes = plt.subplots(2,3, figsize=(12,8), sharex='all', sharey='all')
h_axes = np.ravel(h_axes)
for i, highlight in enumerate(plot_highlight):
    plt.axes(h_axes[i])
    plt.title(highlight)
    for c in range(len(cdtn_name)):
        plt.plot(ts, data_group_mean_norm[c][:, yn_keep_signal].mean(axis=1),
                 color=colors[c], linestyle=linestyles[c],
                 alpha=plot_highlight[highlight]['trace'][c])
    if i==0:
        plt.legend(cdtn_name)
plt.xlim([-0.050, 0.400])
plt.savefig(os.path.join(path_to_fig, 'LFP_V4_before_select.png'))
plt.savefig(os.path.join(path_to_fig, 'LFP_V4_before_select.pdf'))


yn_keep_signal = (signal_all_lfp['area'] != 'V4') & (signal_all_lfp['valid']==1)
h_fig, h_axes = plt.subplots(2,3, figsize=(12,8), sharex='all', sharey='all')
h_axes = np.ravel(h_axes)
for i, highlight in enumerate(plot_highlight):
    plt.axes(h_axes[i])
    plt.title(highlight)
    for c in range(len(cdtn_name)):
        plt.plot(ts, data_group_mean_norm[c][:, yn_keep_signal].mean(axis=1),
                 color=colors[c], linestyle=linestyles[c],
                 alpha=plot_highlight[highlight]['trace'][c])
    if i==0:
        plt.legend(cdtn_name)
plt.xlim([-0.050, 0.400])
plt.savefig(os.path.join(path_to_fig, 'LFP_IT_before_select.png'))
plt.savefig(os.path.join(path_to_fig, 'LFP_IT_before_select.pdf'))


##
""" ========== signal_filter_function ========== """

def signal_filter(keep_mode=None, signal_all_lfp=signal_all_lfp):


    # keep_mode = 'V4'
    # keep_mode = 'IT'
    # keep_mode = 'IT_Dante'
    # keep_mode = 'IT_Thor'
    # keep_mode = 'IT_Gr'
    # keep_mode = 'IT_sGr'
    # keep_mode = 'IT_iGr'

    if keep_mode=='V4':
        yn_keep_signal = (signal_all_lfp['area'] == 'V4') \
                         & (signal_all_lfp['valid']==1)
    elif keep_mode=='IT':
        yn_keep_signal = ((signal_all_lfp['area'] == 'TEd') | (signal_all_lfp['area'] == 'TEm'))  \
                         & (signal_all_lfp['valid']==1) \
                         & (np.abs(signal_all_lfp['depth']) < 7)
    elif keep_mode=='IT_Dante':
        yn_keep_signal = ((signal_all_lfp['area'] == 'TEd') | (signal_all_lfp['area'] == 'TEm'))  \
                         & (signal_all_lfp['valid']==1) \
                         & (signal_all_lfp['animal']=='Dante') \
                         & (np.abs(signal_all_lfp['depth']) < 7)
    elif keep_mode=='IT_Thor':
        yn_keep_signal = ((signal_all_lfp['area'] == 'TEd') | (signal_all_lfp['area'] == 'TEm'))  \
                         & (signal_all_lfp['valid']==1) \
                         & (signal_all_lfp['animal']=='Thor') \
                         & (np.abs(signal_all_lfp['depth']) < 7)
    elif keep_mode=='IT_Gr':
        yn_keep_signal = ((signal_all_lfp['area'] == 'TEd') | (signal_all_lfp['area'] == 'TEm'))  \
                         & (signal_all_lfp['valid']==1) \
                         & (signal_all_lfp['depth']>=-2) \
                         & (np.abs(signal_all_lfp['depth']) < 7)
    elif keep_mode=='IT_sGr':
        yn_keep_signal = ((signal_all_lfp['area'] == 'TEd') | (signal_all_lfp['area'] == 'TEm'))  \
                         & (signal_all_lfp['valid']==1) \
                         & (signal_all_lfp['depth']>3) \
                         & (np.abs(signal_all_lfp['depth']) < 7)
    elif keep_mode=='IT_iGr':
        yn_keep_signal = ((signal_all_lfp['area'] == 'TEd') | (signal_all_lfp['area'] == 'TEm'))  \
                         & (signal_all_lfp['valid']==1) \
                         & (signal_all_lfp['depth']<-3) \
                         & (np.abs(signal_all_lfp['depth']) < 7)
    else:
        raise Exception('keep mode not correct')

    return yn_keep_signal


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


keep_mode = 'V4'
# keep_mode = 'IT'
# keep_mode = 'IT_Dante'
# keep_mode = 'IT_Thor'
# keep_mode = 'IT_Gr'
# keep_mode = 'IT_sGr'
# keep_mode = 'IT_iGr'


yn_keep_signal = signal_filter(keep_mode)

h_fig, h_axes = plt.subplots(2, 3, figsize=(12,8), sharex='all', sharey='all')
h_axes = np.ravel(h_axes)
for i, highlight in enumerate(plot_highlight):
    plt.axes(h_axes[i])
    plt.title(highlight)
    for c in range(len(cdtn_name)):
        trace_N = np.sum(yn_keep_signal)
        trace_mean = data_group_mean_norm[c][:, yn_keep_signal].mean(axis=1)
        trace_se = data_group_mean_norm[c][:, yn_keep_signal].std(axis=1)/np.sqrt(trace_N)
        plt.fill_between(ts, trace_mean-2*trace_se, trace_mean+2*trace_se,
                 color=colors[c], linestyle=linestyles[c],
                 alpha=plot_highlight[highlight]['trace'][c]*0.25)
        plt.plot(ts, trace_mean,
                 color=colors[c], linestyle=linestyles[c],
                 alpha=plot_highlight[highlight]['trace'][c])

        # plt.plot(ts, 10**3*data_group_mean_norm[c][:, yn_keep_signal].mean(axis=1),
        #          color=colors[c], linestyle=linestyles[c],
        #          alpha=plot_highlight[highlight]['trace'][c])
    if i==0:
        plt.legend(cdtn_name)
plt.xlim([-0.050, 0.450])
# plt.ylim([-2.5, 2.5])
plt.savefig(os.path.join(path_to_fig, 'LFP_{}.png'.format(keep_mode)))
plt.savefig(os.path.join(path_to_fig, 'LFP_{}.pdf'.format(keep_mode)))


##
""" ========== PSTH by depth: not finished ========== """


# plot in one panel
range_l = range(-8,9,2)
def plot_psth_by_depth(keep_mode):
    scale_depth = 3.0
    [h_fig, h_ax]=plt.subplots(nrows=1, ncols=6, sharex='all', sharey='all', figsize=[10,8])
    for l in range_l:
        neuron_keep = signal_filter(keep_mode)
        # neuron_keep = (signal_info['channel_index'] > 32) * (signal_info['area']==area) * (np.abs(signal_info['depth']-l)<=laminar_range_ave)
        psth = None
        if np.sum(neuron_keep)>0:
            for j in range(6):
                plt.axes(h_ax[j])
                if j==0:
                    plt.text(ts[0], l+0.2, '{}N'.format(np.sum(neuron_keep)))
                    plt.title('all')
                    alphas = [1, 1, 1, 1, 1, 1]
                elif j==1:
                    plt.title('nov')
                    alphas = [1, 1, 1, 0, 0, 0]
                elif j == 2:
                    plt.title('fam')
                    alphas = [0, 0, 0, 1, 1, 1]
                elif j==3:
                    plt.title('noise 0%')
                    alphas = [1, 0, 0, 1, 0, 0]
                elif j==4:
                    plt.title('noise 50%')
                    alphas = [0, 1, 0, 0, 1, 0]
                elif j==5:
                    plt.title('noise 70%')
                    alphas = [0, 0, 1, 0, 0, 1]
                for i in range(psth.shape[0]):
                    plt.plot(ts, psth[i, :]/scale_depth+l, color=colors[i], linestyle=linestyles[i], alpha=alphas[i])
                # plt.ylabel(ylabel)
    plt.xlim([ts[0], ts[-1]])
    plt.xticks([-0.1,0,0.1,0.2,0.3,0.4,0.5], ['','0','','0.2','','0.4',''])
    plt.yticks(range_l)
    h_fig.text(0.5, 0.04, 'time from stim onset (s)', ha='center')
    h_fig.text(0.04, 0.5, 'laminar position (0.1 mm), [+ for SG, - for IG]', va='center', rotation='vertical')
    plt.suptitle('{} by depth'.format(area))
    # plt.savefig('./temp_figs/PSTH_{}_by_depth_{}_{}.pdf'.format(area, block_type, signal_type))
    # plt.savefig('./temp_figs/PSTH_{}_by_depth_{}_{}.png'.format(area, block_type, signal_type))

plot_psth_by_depth('IT')


