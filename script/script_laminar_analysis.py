import os
import sys
import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import time                 # time code execution

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


plt.ion()

""" load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
tankname = '.*GM32.*U16.*180803.*'
block_type = 'srv_mask'
# block_type = 'matchnot'
[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('h_.*{}.*'.format(block_type), tankname, tf_interactive=True,
                                                               dir_tdt_tank='/shared/lab/projects/encounter/data/TDT',
                                                               dir_dg='/shared/lab/projects/analysis/ruobing/data_dg')


""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')


""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'


""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk     = data_load_DLSH.standardize_blk(blk)


""" ==================== """

""" glance LFP trace """
t_plot = [-0.4, 1.000]

data_neuro_LFP = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='analog.*',
                                               name_filter='LFPs.*', spike_bin_rate=1000,
                                               chan_filter=range(1, 48 + 1))
data_neuro_LFP = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro_LFP)

ts = data_neuro_LFP['ts']
fs = data_neuro_LFP['signal_info'][0]['sampling_rate']
signal_info = data_neuro_LFP['signal_info']
cdtn = data_neuro_LFP['cdtn']
erp_groupave = pna.GroupAve(data_neuro_LFP)


pnp.ErpPlot_singlePanel(np.mean(data_neuro_LFP['data'],axis=0).transpose()[32:,:], ts=ts)

""" spectratrum """
t_window = [-0.200, 0.000]
# t_window = [ 0.050, 0.350]
t_window = [ 0.150, 0.350]
# t_window = [ 0.350, 0.650]
[spct, spcg_f] = pna.ComputeWelchSpectrum(data_neuro_LFP['data']*10**6, fs=fs, t_ini=ts[0], t_window=t_window, t_bin=0.1, t_step=None,
                         t_axis=1, batchsize=100, f_lim=[00,150])


h_fig, h_axes = plt.subplots(2,1)
plt.axes(h_axes[0])
pnp.ErpPlot_singlePanel(np.mean(data_neuro_LFP['data'],axis=0).transpose()[32:,:], ts=ts)
plt.title('LFP')
plt.xlabel('time')
plt.axes(h_axes[1])
pnp.ErpPlot_singlePanel(np.log(np.mean(spct, axis=0)).transpose()[32:,:], ts=spcg_f, c_lim_style='basic', cmap='rainbow')
plt.title('spectrum')
plt.xlabel('freqency')

spct_groupave = pna.GroupAve(data_neuro_LFP, spct)

# U16 spectrum by condition
h_fig, h_axes = plt.subplots(4,4, sharex=True, sharey=True)
h_axes = np.ravel(h_axes)

spct_groupave = pna.GroupAve(data_neuro_LFP, spct)
colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9), pnp.gen_distinct_colors(3, luminance=0.6)])
linestyles = ['-', '-', '-', '--', '--', '--']
for j, ch in enumerate(range(33,48+1)) :
    plt.axes(h_axes[j])
    for i in range(len(cdtn)):
        plt.plot(spcg_f, np.log(spct_groupave[i, :, ch - 1]), color=colors[i], linestyle=linestyles[i])


# GM32 spectrum by condition
# h_fig, h_axes = plt.subplots(8,4, sharex=True, sharey=True)
h_fig, h_axes =  pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9), pnp.gen_distinct_colors(3, luminance=0.6)])
linestyles = ['-', '-', '-', '--', '--', '--']
for j, ch in enumerate(range(1,32+1)) :
    plt.axes(h_axes[j])
    for i in range(len(cdtn)):
        plt.plot(spcg_f, np.log(spct_groupave[i, :, ch - 1]), color=colors[i], linestyle=linestyles[i])



""" spectrum over time """
h_fig_GM32, h_axes_GM32 = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
h_fig_U16,  h_axes_U16  = plt.subplots(4,4, sharex=True, sharey=True)
h_axes_U16 = np.ravel(h_axes_U16)
h_axes = np.concatenate((h_axes_GM32, h_axes_U16))
h_fig_GM32.set_size_inches(14, 14)
h_fig_U16.set_size_inches(12, 12)
list_t_window = [[-0.200, 0.000], [ 0.150, 0.350], [ 0.400, 0.600]]
list_color = np.linspace(1.0, 0.2, len(list_t_window))
list_epoch_name = ['pre', 'stim', 'gap']
for i, t_window in enumerate(list_t_window):
    [spct, spcg_f] = pna.ComputeWelchSpectrum(data_neuro_LFP['data'] * 10 ** 6, fs=fs, t_ini=ts[0], t_window=t_window,
                                              t_bin=0.1, t_step=None, t_axis=1, batchsize=100, f_lim=[00, 120])

    spct_ave = np.mean(spct, axis=0)
    for j in range(48):
        if j<32:
            plt.figure(h_fig_GM32.number)
        else:
            plt.figure(h_fig_U16.number)
        plt.axes(h_axes[j])
        plt.plot(spcg_f, np.log(spct_ave[:, j]), color='k', alpha=list_color[i])
plt.figure(h_fig_GM32.number)
plt.legend(list_epoch_name)
plt.xticks(range(0,120,20))
plt.xlabel('frequency')
plt.suptitle('LFP power spectrum, V4')
plt.savefig('./temp_figs/spectrum_V4_{}.pdf'.format(tankname))
plt.figure(h_fig_U16.number)
plt.legend(list_epoch_name)
plt.xticks(range(0,120,20))
plt.xlabel('frequency')
plt.suptitle('LFP power spectrum, IT')
plt.savefig('./temp_figs/spectrum_IT_{}.pdf'.format(tankname))


""" coherence over time """
h_fig_GM32, h_axes_GM32 = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
h_fig_U16,  h_axes_U16  = plt.subplots(4,4, sharex=True, sharey=True)
h_axes_U16 = np.ravel(h_axes_U16)
h_axes = np.concatenate((h_axes_GM32, h_axes_U16))
h_fig_GM32.set_size_inches(14, 14)
h_fig_U16.set_size_inches(12, 12)
ch0=21
ch1=45
list_t_window = [[-0.200, 0.000], [ 0.150, 0.350], [ 0.400, 0.600]]
list_color = np.linspace(1.0, 0.2, len(list_t_window))
list_epoch_name = ['pre', 'stim', 'gap']

for ch1 in range(48):
    if ch1 < 32:
        plt.figure(h_fig_GM32.number)
    else:
        plt.figure(h_fig_U16.number)
    plt.axes(h_axes[ch1])
    for i, t_window in enumerate(list_t_window):
        data0 = data_neuro_LFP['data'][:, :, ch0]*10**6
        data1 = data_neuro_LFP['data'][:, :, ch1]*10**6
        [cohe, spcg_f] = pna.ComputeWelchCoherence(data0, data1, fs=fs, t_ini=ts[0], t_window=t_window,
                                                  t_bin=0.1, t_step=None, t_axis=1, batchsize=100, f_lim=[00, 120])

        plt.plot(spcg_f, cohe, color='k', alpha=list_color[i])
plt.figure(h_fig_GM32.number)
plt.legend(list_epoch_name)
plt.xticks(range(0,120,20))
plt.xlabel('frequency')
plt.suptitle('coherence, ch{}-V4'.format(ch0))
plt.savefig('./temp_figs/coherence_{}-V4_{}.pdf'.format(ch0,tankname))
plt.figure(h_fig_U16.number)
plt.legend(list_epoch_name)
plt.xlabel('frequency')
plt.xticks(range(0,150,20))
plt.suptitle('LFP power spectrum, ch{}-IT'.format(ch0))
plt.savefig('./temp_figs/coherence_{}-IT_{}.pdf'.format(ch0,tankname))


""" coherence over condition """
for ch0 in range(48):
    h_fig_GM32, h_axes_GM32 = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
    h_fig_U16,  h_axes_U16  = plt.subplots(4,4, sharex=True, sharey=True)
    h_axes_U16 = np.ravel(h_axes_U16)
    h_axes = np.concatenate((h_axes_GM32, h_axes_U16))
    h_fig_GM32.set_size_inches(14, 14)
    h_fig_U16.set_size_inches(12, 12)

    list_t_window = [ 0.150, 0.350]
    list_color = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9), pnp.gen_distinct_colors(3, luminance=0.6)])
    list_linestyle = ['-', '-', '-', '--', '--', '--']

    for ch1 in range(48):
        if ch1==ch0:
            continue
        if ch1 < 32:
            plt.figure(h_fig_GM32.number)
        else:
            plt.figure(h_fig_U16.number)
        plt.axes(h_axes[ch1])
        for i, cd in enumerate(cdtn):
            cd_indx = data_neuro_LFP['cdtn_indx'][cd]
            data0 = data_neuro_LFP['data'][cd_indx, :, ch0]*10**6
            data1 = data_neuro_LFP['data'][cd_indx, :, ch1]*10**6
            [cohe, spcg_f] = pna.ComputeWelchCoherence(data0, data1, fs=fs, t_ini=ts[0], t_window=t_window,
                                                      t_bin=0.1, t_step=None, t_axis=1, batchsize=100, f_lim=[00, 120])

            plt.plot(spcg_f, cohe, color=list_color[i], linestyle=list_linestyle[i])
    plt.figure(h_fig_GM32.number)
    plt.legend(cdtn)
    plt.xticks(range(0,120,20))
    plt.xlabel('frequency')
    plt.suptitle('coherence, ch{}-V4'.format(ch0))
    plt.savefig('./temp_figs/coherence_cdtn_{}-V4_{}.pdf'.format(ch0,tankname))
    plt.figure(h_fig_U16.number)
    plt.legend(cdtn)
    plt.xlabel('frequency')
    plt.xticks(range(0,150,20))
    plt.suptitle('coherence, ch{}-IT'.format(ch0))
    plt.savefig('./temp_figs/coherence_cdtn_{}-IT_{}.pdf'.format(ch0,tankname))
    plt.close('all')
