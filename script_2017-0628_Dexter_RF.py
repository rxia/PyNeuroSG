""" Import modules """
# ----- standard modules -----
import os
import sys
sys.path.append('/shared/homes/sguan/Coding_Projects/PyNeuroSG')
import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import time                 # time code execution
# ----- modules used to read neuro data -----
import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import quantities as pq
# ----- modules of the project PyNeuroSG -----
import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc
# ----- modules for the data location and organization in Sheinberg lab -----
import data_load_DLSH       # package specific for DLSH lab data
from GM32_layout import layout_GM32

# dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank'
# dir_dg='/shared/homes/sguan/neuro_data/stim_dg'
dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT'
dir_dg='/shared/lab/projects/analysis/ruobing/data_dg'
keyword_tank = '.*Dexter.*170629.*'
keyword_block= 'x.*spotopto.*'
name_file =  keyword_block
tankname =   keyword_tank
# signal_type = 'spk'
signal_type = 'LFP'
tf_plot_movie = False

""" load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(name_file, tankname, tf_interactive=True,
                                                           dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg)

""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'

""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk = data_load_DLSH.standardize_blk(blk)

""" waveform plot """
# pnp.SpkWfPlot(blk.segments[0])

""" data align """
if signal_type == 'spk':
    data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.020, 0.200], type_filter='spiketrains.*',
                                               name_filter='.*Code[1-9]$', spike_bin_rate=100)
elif signal_type == 'LFP':
    data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.1, 0.5], type_filter='ana.*', name_filter='LFPs.*')

# ERP plot
# ERP = np.mean(data_neuro['data'][data_df['stimulate_type'] == 0,:,:], axis=0).transpose()
# pnp.ErpPlot(ERP[0:32, :], data_neuro['ts'], array_layout=layout_GM32)

# plot 1 channel's data
pnp.PsthPlot(data_neuro['data'][:,:,0], ts=data_neuro['ts'], cdtn=data_df['stimulate_type'])

# group by x,y cooridnate
data_neuro = signal_align.neuro_sort(data_df, ['stim_pos_x', 'stim_pos_y'], data_df['stimulate_type']==0, data_neuro)

# calculate power spectrum
if signal_type == 'LFP':
    [spcg, spcg_t, spcg_f] = pna.ComputeSpectrogram(data_neuro['data'], fs=data_neuro['signal_info']['sampling_rate'][0],
                           t_ini=data_neuro['ts'][0], t_bin=0.200, t_step=None, t_axis=1, batchsize=500,
                           f_lim=[5,200])
# plot
def plot_RF(t_window_plot=[0.0,0.15], tf_spcg=False, f_lim=[0,500], data_neuro=data_neuro):
    h_fig, h_axes = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
    h_fig.set_size_inches([10, 9], forward=True)
    plt.tight_layout()
    if tf_spcg:
        data_neuro_spcg = {}
        data_neuro_spcg['data'] = np.mean(spcg[:, np.logical_and(spcg_f >= f_lim[0], spcg_f < f_lim[1]), :, :],
                                          axis=1).transpose([0, 2, 1])
        data_neuro_spcg['ts'] = spcg_t
        data_neuro_spcg['signal_info'] = data_neuro['signal_info']
        data_neuro_spcg['fltr'] = data_neuro['fltr']
        data_neuro_spcg['grpby'] = data_neuro['grpby']
        data_neuro_spcg['cdtn'] = data_neuro['cdtn']
        data_neuro_spcg['cdtn_indx'] = data_neuro['cdtn_indx']
    for i in sorted(range(len(data_neuro['signal_info'])), reverse=True):
        ch = data_neuro['signal_info'][i]['channel_index']
        if ch <= 32:
            plt.axes(h_axes[ch - 1])
            if tf_spcg:
                pnp.RfPlot(data_neuro_spcg, indx_sgnl=i, t_focus=t_window_plot,
                           psth_overlay=False, tf_scr_ctr=True)
            else:
                pnp.RfPlot(data_neuro, indx_sgnl=i, t_focus=t_window_plot,
                   psth_overlay=False, tf_scr_ctr=True)


data_neuro = signal_align.neuro_sort(data_df, ['stim_pos_x', 'stim_pos_y'], data_df['stimulate_type']==0, data_neuro)
f_lim=[30,55]
plot_RF(t_window_plot=[0.0,0.200], tf_spcg=True, f_lim=f_lim, data_neuro=data_neuro)
plt.suptitle('{}_{}.png'.format(filename_common, f_lim))

plt.savefig('RF_mapping_{}_LFP_gamma.png'.format(filename_common))






""" ===== tuning curve ===== """

keyword_block= 'x.*imageopto.*'
name_file =  keyword_block
tankname =   keyword_tank
# signal_type = 'spk'
signal_type = 'LFP'
tf_plot_movie = True
""" load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(name_file, tankname, tf_interactive=True,
                                                           dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg)

""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')
""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'
""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk = data_load_DLSH.standardize_blk(blk)
""" waveform plot """
# pnp.SpkWfPlot(blk.segments[0])

data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.020, 0.200], type_filter='spiketrains.*',
                                           name_filter='.*Code[1-9]$', spike_bin_rate=1000)
# data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.1, 0.5], type_filter='ana.*', name_filter='LFPs.*')
data_neuro = signal_align.neuro_sort(data_df, ['orient'], [], data_neuro)

pnp.PsthPlot(data_neuro['data'][:,:,0], ts=data_neuro['ts'], cdtn=data_df['orient'], sk_std=0.005, color_style='continuous')

[spcg, spcg_t, spcg_f] = pna.ComputeSpectrogram(data_neuro['data'], fs=data_neuro['signal_info']['sampling_rate'][0],
                           t_ini=data_neuro['ts'][0], t_bin=0.200, t_step=None, t_axis=1, batchsize=500,
                           f_lim=[5,200])


def plot_tuning(t_window_plot=[0.0,0.15], tf_spcg=False, f_lim=[0,500], data_df=data_df, plot_type='psth'):
    h_fig, h_axes = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
    h_fig.set_size_inches([10, 9], forward=True)
    plt.tight_layout()
    spcg_power = np.mean(spcg[:, np.logical_and(spcg_f >= f_lim[0], spcg_f < f_lim[1]), :, :], axis=1).transpose([0, 2, 1])
    for i in range(len(data_neuro['signal_info'])):
        ch = data_neuro['signal_info'][i]['channel_index']
        if ch <= 32:
            plt.axes(h_axes[ch - 1])
            if plot_type == 'psth':
                if tf_spcg:
                    pnp.PsthPlot(spcg_power[:,:,ch-1], ts=spcg_t, cdtn=data_df['orient'], color_style='continuous', subpanel='')
                else:
                    pnp.PsthPlot(data_neuro['data'][:, :, i], ts=data_neuro['ts'], cdtn=data_df['orient'], color_style='continuous', subpanel='')
            elif plot_type == 'tuning_curve':
                if tf_spcg:
                    tuning_x, tuning_y = pna.TuningCurve(spcg_power[:, :, ch-1], label=data_df['orient'], ts=spcg_t,
                                                     t_window=[0, 0.200])
                else:
                    tuning_x, tuning_y = pna.TuningCurve(data_neuro['data'][:, :, i], label=data_df['orient'], ts=data_neuro['ts'],
                                                         t_window=[0, 0.200])
                plt.plot(tuning_x, tuning_y, 'o-')

plot_tuning(t_window_plot=[0.0,0.15], tf_spcg=True, f_lim=[30,55], plot_type='tuning_curve')
plot_tuning(t_window_plot=[0.05,0.30], tf_spcg=False, f_lim=[30,55], plot_type='tuning_curve')





""" ===== gratings ===== """

signal_type = 'spk'

keyword_tank = '.*Dexter.*170703.*'
keyword_block= 'x.*texture.*'
# keyword_block= 'x.*imageopto.*'

""" load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(keyword_block, keyword_tank, tf_interactive=True,
                                                           dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg)

""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'

""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk = data_load_DLSH.standardize_blk(blk)

""" waveform plot """
# pnp.SpkWfPlot(blk.segments[0])

""" data align """
if signal_type == 'spk':
    data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.2, 0.600], type_filter='spiketrains.*',
                                               name_filter='.*Code[1-9]$', spike_bin_rate=1000)
elif signal_type == 'LFP':
    data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.1, 0.5], type_filter='ana.*', name_filter='LFPs.*')

data_df['orient_float'] = np.array(data_df['stim_names'].str.extract('fftnoise_(\d\d\d)_.*'), dtype=float)/180*np.pi
data_df['hue_float']    = data_df['hue']*1.0
data_df['orient'] = data_df['orient_float']

t_focus = [0.05, 0.300]
data_neuro = signal_align.neuro_sort(tlbl=data_df, grpby=['hue_float', 'orient_float'], fltr=[], neuro=data_neuro)

pnp.RfPlot(data_neuro, indx_sgnl=2, t_focus=t_focus, psth_overlay=True, tf_scr_ctr=False)

h_fig, h_axes = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
h_fig.set_size_inches([10, 9], forward=True)
plt.tight_layout()
for i in sorted(range(len(data_neuro['signal_info'])), reverse=True):
    ch = data_neuro['signal_info'][i]['channel_index']
    if ch <= 32:
        plt.axes(h_axes[ch - 1])
        pnp.RfPlot(data_neuro, indx_sgnl=i, t_focus=t_focus,
                   psth_overlay=False, tf_scr_ctr=True)

