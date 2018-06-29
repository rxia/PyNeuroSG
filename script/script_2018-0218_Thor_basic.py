""" script to get the basic tuning curve """

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
import df_ana
import misc_tools           # in this package: misc

import data_load_DLSH       # package specific for DLSH lab data

from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge


dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT'
dir_dg = '/shared/lab/projects/analysis/shaobo/data_dg'

[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(
    '.*MT', '.*Thor.*U16.*180224.*',
    dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg,
    tf_interactive=True)

""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'


""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk     = data_load_DLSH.standardize_blk(blk)


""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

""" sort and plot """
window_offset = [-0.05, 0.500]
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset,
                                         type_filter='spiketrains.*', name_filter='.*Code[1-9]$',
                                         spike_bin_rate=1000, chan_filter=range(1,32+1))

# srv
grpby = 'stim_sname'
df_group = df_ana.DfGroupby(data_df, groupby=grpby, tf_aggregate=False)
for i_signal in range(len(data_neuro['signal_info'])):
    signal_name = data_neuro['signal_info'][i_signal]['name']
    h_fig, h_axes = pnp.CreateSubplotFromGroupby(df_group['order'], tf_title=False)
    plt.gcf().set_size_inches([10, 8])
    tf_legend = True
    for cdtn in df_group['idx']:
        plt.axes(h_axes[cdtn])
        pnp.PsthPlot(data_neuro['data'][:, :, i_signal], ts=data_neuro['ts'],
                     cdtn=data_df['mask_opacity_int'], limit=df_group['idx'][cdtn], sk_std=0.005,
                     tf_legend=tf_legend)
        tf_legend = False
        plt.title(cdtn, fontsize='small')
    plt.suptitle('{} {}, by {}'.format(filename_common, signal_name, grpby))
    plt.savefig('./temp_figs/{}_{}.png'.format(filename_common, signal_name))
    plt.close()


# MT mapping
num_signals = len(data_neuro['signal_info'])
h_fig, h_axes = pnp.SubplotsAutoRowCol(num_signals, figsize=(10, 8))
for i_signal in range(num_signals):
    plt.axes(h_axes[i_signal])
    signal_name = data_neuro['signal_info'][i_signal]['name']
    pnp.PsthPlot(data_neuro['data'][:, :, i_signal], ts=data_neuro['ts'],
                 cdtn=data_df['rotationDirections'], limit=None, sk_std=0.015,
                 tf_legend=tf_legend, color_style='continuous')
    plt.title(signal_name)
plt.savefig('./temp_figs/direction_tuning_{}.png'.format(filename_common))


# RF spot mapping
window_offset = [0.0, 0.2]
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset,
                                         type_filter='spiketrains.*', name_filter='.*Code[1-9]$',
                                         spike_bin_rate=1000, chan_filter=range(1,32+1))
data_neuro=signal_align.neuro_sort(data_df, ['stim_pos_x','stim_pos_y'], [], data_neuro)
num_signals = len(data_neuro['signal_info'])
h_fig, h_axes = pnp.SubplotsAutoRowCol(num_signals, figsize=[8,8])
for i_signal in range(num_signals):
    plt.axes(h_axes[i_signal])
    pnp.RfPlot(data_neuro, sk_std=0.01, indx_sgnl=i_signal)
    plt.axis('equal')
    plt.axis('square')
plt.savefig('./temp_figs/RF_mapping_{}.png'.format(filename_common))


""" spike waveforms """
pnp.SpkWfPlot(blk.segments[0], sortcode_min=1)
plt.savefig('{}/{}_spk_waveform.png'.format(dir_temp_fig, filename_common))


""" ERP plot """
t_plot = [-0.100, 0.500]

data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*', name_filter='LFPs.*')
ERP = np.mean(data_neuro['data'], axis=0).transpose()
pnp.ErpPlot(ERP, data_neuro['ts'])
plt.savefig('{}/{}_ERP_all.png'.format(dir_temp_fig, filename_common))



""" ERP by recording depth from diffent files """

[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(
    '.*srv', '.*Thor.*S4.*180227.*',
    dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg,
    tf_interactive=True)

file_depth = {2: 10100, 3: 11400, 4: 12300,
              5: 13900, 6: 14400,
              7: 16300, 8: 16800, 9: 17000, 10: 17600, 13: 18150, 14: 18400}

# signal = 'spk'
signal = 'LFP'

data_df['depth'] = [file_depth[file] for file in data_df['file']]


filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'

data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk     = data_load_DLSH.standardize_blk(blk)

ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

window_offset = [-0.05, 0.500]
if signal == 'LFP':
    data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset,
                                             type_filter='ana.*', name_filter='LFP.*',
                                             spike_bin_rate=1000, chan_filter=[1])
elif signal == 'spk':
    data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset,
                                             type_filter='spiketrains*', name_filter='.*Code[1-9]$',
                                             spike_bin_rate=1000, chan_filter=[1])

# combine different sortcodes
data_neuro['data'][:,:,0] = np.sum(data_neuro['data'], axis=2)

pnp.PsthPlot(data_neuro['data'][:, :, 0], ts=data_neuro['ts'],
             cdtn=data_df['depth'], limit=None, sk_std=0.010,
             tf_legend=True, color_style='continuous')
plt.title('response by depth, {}'.format(filename_common))

# plot traces by depth
data_neuro = signal_align.neuro_sort(data_df, grpby='depth', neuro=data_neuro)
ERPs_by_depth = pna.GroupAve(data_neuro)

colors = pnp.gen_distinct_colors(len(ERPs_by_depth), style='continuous')
if signal == 'LFP':
    ERPs_by_depth = ERPs_by_depth*10**6
    scale = 100  # 10 uV/mm in axis
    scale_unit = 'mV/mm'
elif signal == 'spk':
    ERPs_by_depth = pna.SmoothTrace(ERPs_by_depth, sk_std=0.005, ts=data_neuro['ts'])
    scale = 100  # 10 uV/mm in axis
    scale_unit = 'spk/sec /mm'

plt.figure(figsize=[6,8])
for i, cdtn in enumerate(data_neuro['cdtn']):
    plt.plot(data_neuro['ts'],  ERPs_by_depth[i,:,0]/scale - cdtn/1000.0, color=colors[i])
    plt.plot(data_neuro['ts'][0], -cdtn/1000.0, 'o', color=colors[i])
plt.xlabel('time form stimulus onset, in sec')
plt.ylabel('depth in mm, {} {}'.format(scale, scale_unit))
plt.title('ERP by depth, {}'.format(filename_common))
plt.savefig('./temp_figs/ERP_by_depth_{}.png'.format(filename_common))


""" below is legacy code """


""" PSTH plot by condition """
# align
window_offset = [-0.100, 0.6]
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000, chan_filter=range(1,32+1))



window_offset = [-0.100, 0.6]
""" GM32 spike by condition  """
# align
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000, chan_filter=range(1,32+1))
# group
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro)
# plot
pnp.NeuroPlot(data_neuro, sk_std=0.010, tf_legend=True, tf_seperate_window=False)
plt.suptitle('spk_GM32    {}'.format(filename_common), fontsize=20)
plt.savefig('{}/{}_spk.png'.format(dir_temp_fig, filename_common))



"""" psth plot """
pnp.PsthPlotCdtn(data_neuro, data_df, i_signal=0, grpby=['stim_sname'], sk_std=0.01)





pnp.PsthPlot(data_neuro['data'][:,:,0], ts=data_neuro['ts'],
                 cdtn=data_df['mask_opacity_int'], sk_std=0.005)

import df_ana; import importlib; importlib.reload(df_ana); importlib.reload(misc_tools); df_group = df_ana.DfGroupby(data_df, 'stim_categories', limit=range(100,200), tf_aggregate=True, tf_linearize=False)
importlib.reload(pnp); pnp.CreateSubplotFromGroupby(df_group['order'])

