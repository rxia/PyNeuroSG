""" script to load all dataset and get the six conditions, designed to run on Pogo """

import os
import sys
import numpy as np
import scipy as sp
import warnings
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import time                 # time code execution
import cPickle as pickle


import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import quantities as pq
import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc

import data_load_DLSH       # package specific for DLSH lab data

from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge


from GM32_layout import layout_GM32



""" ----- load psth data ----- """

block_type = 'srv_mask'
# block_type = 'matchnot'
signal_type='spk'
# signal_type='lfp'
[list_data_groupave, list_ts, list_signal_info, list_cdtn, list_date] = pickle.load(open('/shared/homes/sguan/Coding_Projects/support_data/GroupAve_{}_{}'.format(block_type, signal_type)))


def GetDataCat( list_data_groupave, list_ts, list_signal_info, list_cdtn, list_date ):
    # add date to signal_info
    list_signal_info_date = []
    for signal_info, date in zip(list_signal_info, list_date):
        signal_info_date = pd.DataFrame(signal_info)
        signal_info_date['date'] = date
        list_signal_info_date.append(signal_info_date)
    return [np.dstack(list_data_groupave), list_ts[0], pd.concat(list_signal_info_date, ignore_index=True), list_cdtn[0]]

[data_groupave, ts, signal_info, cdtn] = GetDataCat( list_data_groupave, list_ts, list_signal_info, list_cdtn, list_date )

if signal_type == 'lfp':
    data_groupave = data_groupave*10**6
    sk_std = 0.002
    ylabel = 'uV'
    laminar_range_ave = 1
else:
    sk_std = 0.007
    # ylabel = 'spk/sec'
    ylabel = 'normalized firing rate'
    laminar_range_ave = 1


""" ----- load signal info data (area, depth, neuron type) ------ """
signal_info_detail = pd.read_pickle('/shared/homes/sguan/Coding_Projects/support_data/spike_wf_info_Dante.pkl')

def set_signal_id(signal_info):
    signal_info['signal_id'] = signal_info['date'].apply(lambda x: '{:0>6}'.format(x)).str.cat(
        [signal_info['channel_index'].apply(lambda x: '{:0>2}'.format(x)),
        signal_info['sort_code'].apply(lambda x: '{:0>1}'.format(x))],
        sep='_'
        )
    return signal_info

signal_info = set_signal_id(signal_info)
signal_info_detail = set_signal_id(signal_info_detail)
signal_info_full = signal_info.merge(signal_info_detail, how='inner', on=['signal_id','date','channel_index','sort_code'], copy=False)
if len(signal_info_full) != len(signal_info):
    warnings.warn('DataFrame signal_info length changed')




""" plot PSTH """
def cal_psth_std(neuron_keep):
    data_smooth = pna.SmoothTrace(data_groupave[:,:, neuron_keep], ts=ts, sk_std=sk_std, axis=1)
    if signal_type == 'spk':
        data_norm = data_smooth / (np.mean(data_smooth, axis=(0,1)) + 5)   # soft normalize
    else:
        data_norm = (data_smooth- np.mean(data_smooth, axis=(0,1), keepdims=True)) / np.std(data_smooth, axis=(0,1), keepdims=True)
    psth_mean = np.mean(data_norm, axis=2)
    psth_std = np.std(data_norm, axis=2)
    return psth_mean, psth_std, data_norm
def t_test_score(data_norm):
    t_nf_00, p_nf_00 = sp.stats.ttest_1samp(data_norm[0, :, :] - data_norm[3, :, :], 0, axis=1)
    t_nf_50, p_nf_50 = sp.stats.ttest_1samp(data_norm[1, :, :] - data_norm[4, :, :], 0, axis=1)
    t_nf_70, p_nf_70 = sp.stats.ttest_1samp(data_norm[2, :, :] - data_norm[5, :, :], 0, axis=1)
    t_ns_no, p_ns_no = sp.stats.ttest_1samp(data_norm[0, :, :] - data_norm[2, :, :], 0, axis=1)
    t_ns_fa, p_ns_fa = sp.stats.ttest_1samp(data_norm[3, :, :] - data_norm[5, :, :], 0, axis=1)
    return np.vstack([t_nf_00, t_nf_50, t_nf_70, t_ns_no, t_ns_fa])
def d_test_score(data_norm):
    d_nf_00 = pna.cal_CohenD(data_norm[0, :, :], data_norm[3, :, :], axis=1, type_test='paired')
    d_nf_50 = pna.cal_CohenD(data_norm[1, :, :], data_norm[4, :, :], axis=1, type_test='paired')
    d_nf_70 = pna.cal_CohenD(data_norm[2, :, :], data_norm[5, :, :], axis=1, type_test='paired')
    d_ns_no = pna.cal_CohenD(data_norm[0, :, :], data_norm[2, :, :], axis=1, type_test='paired')
    d_ns_fa = pna.cal_CohenD(data_norm[3, :, :], data_norm[5, :, :], axis=1, type_test='paired')
    return np.vstack([d_nf_00, d_nf_50, d_nf_70, d_ns_no, d_ns_fa])

colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9), pnp.gen_distinct_colors(3, luminance=0.6)])
linestyles = ['--', '--', '--', '-', '-', '-']
colors_p     = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9), [0.4,0.4,0.4,1], [0,0,0,1]])
linestyles_p = ['-', '-', '-', '--', '--']

plot_highlight = ''
if plot_highlight == 'nov':
    alphas = [1,1,1,0,0,0]
    alphas_p = [0, 0, 0, 1, 0]
elif plot_highlight == 'fam':
    alphas = [0,0,0,1,1,1]
    alphas_p = [0, 0, 0, 0, 1]
elif plot_highlight == '00':
    alphas = [1,0,0,1,0,0]
    alphas_p = [1, 0, 0, 0, 0]
elif plot_highlight == '50':
    alphas = [0,1,0,0,1,0]
    alphas_p = [0, 1, 0, 0, 0]
elif plot_highlight == '70':
    alphas = [0,0,1,0,0,1]
    alphas_p = [0, 0, 1, 0, 0]
elif plot_highlight == '':
    alphas = [1, 1, 1, 1, 1, 1]
    alphas_p = [1, 1, 1, 1, 1]
else:
    alphas = [1, 1, 1, 1, 1, 1]
    alphas_p = [0.5, 0.5, 0.5, 0.5, 0.5]

t_thrhd = 4
def plot_psth_p(neuron_keep):
    N_neuron = np.sum(neuron_keep)
    [psth_mean, psth_std, data_norm] = cal_psth_std(neuron_keep)
    t_values = t_test_score(data_norm)
    d_values = d_test_score(data_norm)

    for i in range(psth_mean.shape[0]):
        plt.plot(ts, psth_mean[i, :], color=colors[i], linestyle=linestyles[i], alpha=alphas[i])
        plt.fill_between(ts, psth_mean[i, :] - psth_std[i, :] / np.sqrt(N_neuron),
                         psth_mean[i, :] + psth_std[i, :] / np.sqrt(N_neuron), color=colors[i], alpha=0.2*alphas[i])
    plt.xlabel('t (s)')
    plt.ylabel(ylabel)

    h_ax_top = pnp.add_sub_axes(h_axes=plt.gca(), size=0.4, loc='top')
    plt.axes(h_ax_top)
    thrhd_d_small = 0.2
    thrhd_d_medium = 0.5
    size_gap = 2
    for i, d in enumerate(d_values):
        plt.fill_between(ts, -i*size_gap + d, -i*size_gap, color=colors_p[i], alpha=0.1*alphas_p[i])
        plt.fill_between(ts, -i*size_gap + d, -i*size_gap, where=np.abs(d) >= thrhd_d_small, color=colors_p[i], alpha=0.3*alphas_p[i])
        plt.fill_between(ts, -i*size_gap + d, -i*size_gap, where=np.abs(d) >= thrhd_d_medium, color=colors_p[i], alpha=0.7*alphas_p[i])
    plt.ylim([-len(d_values)*size_gap,1])
    plt.ylabel('effect size')
    plt.yticks([])

[h_fig, h_ax]=plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all', figsize=[8,10])

plt.axes(h_ax[0,0])
neuron_keep = (signal_info_full['area']=='TEd') * (signal_info_full['wf_type']=='BS')
plot_psth_p(neuron_keep)
plt.title('TEd BS, N={}'.format(np.sum(neuron_keep)))

plt.axes(h_ax[1,0])
neuron_keep = (signal_info_full['area']=='TEd') * (signal_info_full['wf_type']=='NS')
plot_psth_p(neuron_keep)
plt.title('TEd NS, N={}'.format(np.sum(neuron_keep)))

plt.axes(h_ax[0,1])
neuron_keep = (signal_info_full['area']=='TEm') * (signal_info_full['wf_type']=='BS')
plot_psth_p(neuron_keep)
plt.title('TEm BS, N={}'.format(np.sum(neuron_keep)))

plt.axes(h_ax[1,1])
neuron_keep = (signal_info_full['area']=='TEm') * (signal_info_full['wf_type']=='NS')
plot_psth_p(neuron_keep)
plt.title('TEm NS, N={}'.format(np.sum(neuron_keep)))

plt.axes(h_ax[1,1])
plt.xticks(np.arange(-0.1,0.5+0.01,0.1))
h_legend = plt.legend(['nov, 0%', 'nov,50%', 'nov,70%', 'fam, 0%', 'fam,50%', 'fam,70%'], loc='upper left', bbox_to_anchor=(0.9, 1.05))
plt.setp(h_legend.texts, family='monospace')
plt.suptitle('PSTH by area neuron_type {} {}'.format(block_type, signal_type))
plt.savefig('./temp_figs/PSTH_by_area_neuron_type_{}_{}_{}.pdf'.format(block_type, signal_type, plot_highlight))
plt.savefig('./temp_figs/PSTH_by_area_neuron_type_{}_{}_{}.png'.format(block_type, signal_type, plot_highlight))



""" tuning curve """
[list_rank_tuning, list_rank_tuning_early, list_rank_tuning_late, list_ts, list_signal_info, list_cdtn] = pickle.load(open('/shared/homes/sguan/Coding_Projects/support_data/RankTuning_srv_mask'))
list_date = ['161015','161023','161026','161029','161118','161121','161125','161202','161206','161222','161228','170103','170106','170113','170117','170214','170221']


def GetDataCat( list_rank_tuning, list_ts, list_signal_info, list_cdtn ):
    list_signal_info_date=[]
    for signal_info, date in zip(list_signal_info, list_date):
        signal_info_date = pd.DataFrame(signal_info)
        signal_info_date['date'] = date
        list_signal_info_date.append(signal_info_date)
    return [np.dstack(list_rank_tuning), list_ts[0], pd.concat(list_signal_info_date, ignore_index=True), list_cdtn[0]]

tuning_time_window = 'full'
if tuning_time_window == 'full':
    [data_RankTuning, ts, signal_info, cdtn] = GetDataCat(list_rank_tuning, list_ts, list_signal_info, list_cdtn)
    tuning_time_window_str = '50-350ms'
elif tuning_time_window == 'early':
    [data_RankTuning, ts, signal_info, cdtn] = GetDataCat(list_rank_tuning_early, list_ts, list_signal_info, list_cdtn)
    tuning_time_window_str = '50-150ms'
elif tuning_time_window == 'late':
    [data_RankTuning, ts, signal_info, cdtn] = GetDataCat(list_rank_tuning_late, list_ts, list_signal_info, list_cdtn)
    tuning_time_window_str = '250-350ms'

signal_info = set_signal_id(signal_info)
signal_info_detail = pd.read_pickle('/shared/homes/sguan/Coding_Projects/support_data/spike_wf_info_Dante.pkl')
signal_info_detail = set_signal_id(signal_info_detail)
signal_info_full = signal_info.merge(signal_info_detail, how='inner', on=['signal_id','date','channel_index','sort_code'], copy=False)


colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9), pnp.gen_distinct_colors(3, luminance=0.6)])
linestyles = ['--', '--', '--', '-', '-', '-']
[h_fig, h_ax]=plt.subplots(nrows=2, ncols=3, sharex='all', sharey='all', figsize=[8,6])

plt.axes(h_ax[0,0])
neuron_keep = (signal_info_full['area']=='V4')
for i in range(data_RankTuning.shape[0]):
    plt.plot(np.mean(data_RankTuning[i, :, neuron_keep], axis=0), color=colors[i],
             linestyle=linestyles[i])
plt.title('V4, N={}'.format(np.sum(neuron_keep)))
plt.xlabel('image rank')
plt.ylabel('spk/sec')

plt.axes(h_ax[0,1])
neuron_keep = (signal_info_full['area']=='TEd') * (signal_info_full['wf_type']=='BS')
for i in range(data_RankTuning.shape[0]):
    plt.plot( np.mean(data_RankTuning[i,:, neuron_keep ], axis=0), color=colors[i], linestyle=linestyles[i])
plt.title('TEd, N={}'.format(np.sum(neuron_keep)))

plt.axes(h_ax[0,2])
neuron_keep = (signal_info_full['area']=='TEm') * (signal_info_full['wf_type']=='BS')
for i in range(data_RankTuning.shape[0]):
    plt.plot( np.mean(data_RankTuning[i,:, neuron_keep ], axis=0), color=colors[i], linestyle=linestyles[i])
plt.legend(cdtn)
plt.title('TEm, N={}'.format(np.sum(neuron_keep)))

plt.axes(h_ax[1,1])
neuron_keep = (signal_info_full['area']=='TEd') * (signal_info_full['wf_type']=='NS')
for i in range(data_RankTuning.shape[0]):
    plt.plot( np.mean(data_RankTuning[i,:, neuron_keep ], axis=0), color=colors[i], linestyle=linestyles[i])
plt.title('TEd, N={}'.format(np.sum(neuron_keep)))

plt.axes(h_ax[1,2])
neuron_keep = (signal_info_full['area']=='TEm') * (signal_info_full['wf_type']=='NS')
for i in range(data_RankTuning.shape[0]):
    plt.plot( np.mean(data_RankTuning[i,:, neuron_keep ], axis=0), color=colors[i], linestyle=linestyles[i])
plt.legend(cdtn)
plt.title('TEm, N={}'.format(np.sum(neuron_keep)))

plt.suptitle('population rank tuning by area and cell type during {}'.format(tuning_time_window_str))
plt.savefig('./temp_figs/RangkTuning_area_wf_srv_mask_{}.pdf'.format(tuning_time_window))
plt.savefig('./temp_figs/RangkTuning_area_wf_srv_mask_{}.png'.format(tuning_time_window))