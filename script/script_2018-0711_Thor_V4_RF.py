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
import PyNeuroData as pnd   # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc
# ----- modules for the data location and organization in Sheinberg lab -----
import data_load_DLSH       # package specific for DLSH lab data
from GM32_layout import layout_GM32
dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT'
dir_dg='/shared/lab/projects/analysis/ruobing/data_dg'

f_lim = [30,55]
stimulate_type = 0
signal_type = ['LFP', 'spcg']
keyword_tank = '.*Thor.*180710.*'
task_type = 'spot'
if task_type=='spot':
    keyword_block= 'h.*V4_spot.*'
    group_by = ['stim_pos_x', 'stim_pos_y']
    t_range = [-0.1, 0.3]
elif task_type=='colorxgrating':
    keyword_block = 'h.*texture.*'
    group_by = ['orient', 'hue']
    t_range = [-0.2, 0.5]
elif task_type=='fft':
    keyword_block = 'h.*imageopto.*'
    group_by = ['orient']
    t_range = [-0.2, 0.5]
elif task_type=='MTS':
    keyword_block = 'h.*detection.*'
    group_by = ['SampleOrientation']
    t_range = [-0.6, 1.5]

# if 'task_type2' not in locals() or task_type2 != task_type:
#     task_type2 = task_type
#     load_files = True
# elif task_type2 == task_type:
#     load_files = False
load_files = True

""" load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
if load_files:
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(keyword_block, keyword_tank, tf_interactive=False,
                                                               dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg)
    if task_type=='colorxgrating':
        data_df['orient'] = np.array(data_df['stim_names'].str.extract('fftnoise_(\d\d\d)_.*'),
                                           dtype=float) / 180 * np.pi
    if task_type=='MTS':
        data_df['stimulate_type'] = data_df['StimulateType']

    filename_common = misc_tools.str_common(name_tdt_blocks)

    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    # Get StimOn time stamps in neo time frame
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

    # some settings for saving figures
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'




""" data align """

data_neuro_spk = pnd.blk_align_to_evt(blk, ts_StimOn, t_range, type_filter='spiketrains.*',
                                           name_filter='.*Code[1-9]$', spike_bin_rate=1000)

data_neuro_LFP = pnd.blk_align_to_evt(blk, ts_StimOn, t_range, type_filter='ana.*', name_filter='LFPs.*')

# calculate power spectrum
if 'spcg' in signal_type:
    [spcg, spcg_t, spcg_f] = pna.ComputeSpectrogram(data_neuro_LFP['data'], fs=data_neuro_LFP['signal_info']['sampling_rate'][0],
                           t_ini=data_neuro_LFP['ts'][0], t_bin=0.200, t_step=None, t_axis=1, batchsize=100,
                           f_lim=[5,60])



""""""

#""" temp plots """
# # plot 1 channel's data
# pnp.PsthPlot(data_neuro['data'][:,:,0], ts=data_neuro['ts'], cdtn=data_df['stimulate_type'])

# # waveform plot
# pnp.SpkWfPlot(blk.segments[0])


""" group by x,y cooridnate """

if 'stimulate_type' in data_df.keys():
    trial_filter = np.in1d(data_df['stimulate_type'],stimulate_type)
elif 'StimulateType' in data_df.keys():
    trial_filter = np.in1d(data_df['StimulateType'],stimulate_type)
else:
    trial_filter = []

data_neuro_spk = pnd.neuro_sort(data_df, group_by, trial_filter, data_neuro_spk)
data_neuro_LFP = pnd.neuro_sort(data_df, group_by, trial_filter, data_neuro_LFP)
if 'spcg' in signal_type:
    data_neuro_spcg = {}
    data_neuro_spcg['data'] = np.mean(spcg[:, np.logical_and(spcg_f >= f_lim[0], spcg_f < f_lim[1]), :, :],
                                      axis=1).transpose([0, 2, 1])
    data_neuro_spcg['ts'] = spcg_t
    data_neuro_spcg['signal_info'] = data_neuro_LFP['signal_info']
    data_neuro_spcg['fltr'] = data_neuro_LFP['fltr']
    data_neuro_spcg['grpby'] = data_neuro_LFP['grpby']
    data_neuro_spcg['cdtn'] = data_neuro_LFP['cdtn']
    data_neuro_spcg['cdtn_indx'] = data_neuro_LFP['cdtn_indx']

if signal_type == 'spk':
    data_neuro = data_neuro_spk
elif signal_type == 'LFP':
    data_neuro = data_neuro_LFP
elif  'spcg' in signal_type:
    data_neuro = data_neuro_spcg


""" plot RF """

def plot_RF(t_window_plot=t_range, signal_type=signal_type,data_neuro = data_neuro,label = data_df[group_by[0]]):
    h_fig, h_axes = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
    h_fig.set_size_inches([10, 9], forward=True)
    plt.tight_layout()

    for i in sorted(range(len(data_neuro['signal_info'])), reverse=True):
        ch = data_neuro['signal_info'][i]['channel_index']
        if ch <= 32:
            plt.axes(h_axes[ch - 1])
            if len(group_by) == 2:
                pnp.RfPlot(data_neuro, indx_sgnl=i, t_focus=t_window_plot,
                       psth_overlay=False, tf_scr_ctr=True)
            elif len(group_by)==1:
                tuning_x, tuning_y =pna.TuningCurve(data_neuro['data'][:, :, i], label,
                                ts=data_neuro['ts'], t_window=t_window_plot)
                plt.plot(tuning_x, tuning_y, 'o-')
    return h_fig, h_axes

if task_type != 'MTS':

    plot_RF(t_window_plot=[0.05,0.25], signal_type=signal_type,data_neuro = data_neuro)
    plt.suptitle('{}_{}_{}'.format(filename_common, signal_type, f_lim))
    # plt.savefig('{}_{}_{}.png'.format(filename_common, signal_type, f_lim))


""" plot_movie for LFP """
if signal_type == 'LFP':
    data_neuro = data_neuro_LFP
    t_window_dur = 0.05
    for t_window_start in np.arange(0, 0.2, 0.05):
        t_window_plot_cur = [t_window_start, t_window_start+t_window_dur]
        h_fig, h_axes = plot_RF(t_window_plot=t_window_plot_cur, signal_type=signal_type,data_neuro=data_neuro)
        # pnp.share_clim(h_axes)
        plt.suptitle('{}_{}_{}'.format(filename_common, signal_type, t_window_start))

""" plot_movie for LFP for individual channels """
t_window_dur = 0.05
t_window_starts = np.arange(0, 0.3, 0.05)
i_signal = 4
h_fig, h_axes = plt.subplots(1, len(t_window_starts), figsize=(12, 2))
plt.subplots_adjust(left=0.05, right=0.95, top=0.8)
for i, t_window_start in enumerate(t_window_starts):
    t_window_plot = [t_window_start, t_window_start+t_window_dur]
    plt.axes(h_axes[i])
    pnp.RfPlot(data_neuro, indx_sgnl=i_signal, t_focus=t_window_plot, psth_overlay=False, tf_scr_ctr=True)
    plt.title('{:.3f}-{:.3f} s'.format(t_window_start, t_window_start+t_window_dur))
clim = pnp.share_clim(h_axes)
clim = [-np.max(np.abs(clim)), np.max(np.abs(clim))]
clim = pnp.share_clim(h_axes, c_lim=clim, cmap='coolwarm')
plt.suptitle(data_neuro['signal_info']['name'][i])


""" PSTH for MTS """

def plot_PSTH(t_window_plot=t_range, signal_type=signal_type,data_neuro = data_neuro, spcg_result = None):

    h_fig, h_axes = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
    h_fig.set_size_inches([10, 9], forward=True)
    plt.tight_layout()

    for i in sorted(range(len(data_neuro['signal_info'])), reverse=True):
        ch = data_neuro['signal_info'][i]['channel_index']
        if ch <= 32:
            plt.axes(h_axes[ch - 1])
            if signal_type=='spcg':
                pnp.SpectrogramPlot(spcg[:, :, i, :], spcg_t=spcg_t, spcg_f=spcg_f, limit_trial=trial_filter,
                                    tf_phase=True, tf_mesh_t=False, tf_mesh_f=False,
                                    tf_log=True, time_baseline=None,
                                    t_lim=None, f_lim=[5, 100], c_lim=None, c_lim_style=None, name_cmap=None,
                                    rate_interp=None, tf_colorbar=False)
            elif signal_type=='spcg_diff':
                pnp.SpectrogramPlot(data_neuro['spcg_diff'][:, i, :], spcg_t=spcg_t, spcg_f=spcg_f,
                                    tf_phase=True, tf_mesh_t=False, tf_mesh_f=False,
                                    tf_log=False, time_baseline=None,
                                    t_lim=None, f_lim=[5, 100], c_lim=[-0.4,1], c_lim_style=None, name_cmap=None,
                                    rate_interp=None, tf_colorbar=False)
            elif signal_type=='cohg':
                pnp.SpectrogramPlot(spcg_result['cohg'][(3,ch-1)], spcg_t=spcg_result['spcg_t'], spcg_f=spcg_result['spcg_f'],
                                    tf_phase=True, tf_mesh_t=False, tf_mesh_f=False,
                                    tf_log=False, time_baseline=None,
                                    t_lim=None, f_lim=[5, 60], c_lim=[0,1], c_lim_style=None, name_cmap='viridis',
                                    rate_interp=None, tf_colorbar=False,quiver_scale=7, max_quiver=16)
            else:
                pnp.PsthPlot(data_neuro['data'][:, :, i], ts=data_neuro['ts'], cdtn=data_df[group_by[0]],
                             limit=trial_filter, sk_std=0.005, subpanel='')
    return h_fig, h_axes

#if task_type == 'MTS':
# plot_PSTH(t_window_plot=t_range, signal_type=signal_type, data_neuro=data_neuro)


""" Get difference between stimulate & non stimulate """

orient = 158
if task_type == 'spot':
    filter0 = np.logical_and( data_df['stimulate_type']==0, data_df['order']==0)
    filter1 = np.logical_and( data_df['stimulate_type']==2, data_df['order']==0)
elif task_type == 'MTS':
    filter0 = np.logical_and( data_df['stimulate_type']==0, np.in1d(data_df['SampleOrientation'],orient) )
    filter1 = np.logical_and( data_df['stimulate_type']==6, np.in1d(data_df['SampleOrientation'],orient))

spcg_diff = np.log(np.mean(spcg[filter1,:,:,:], axis=0)) - np.log(np.mean(spcg[filter0,:,:,:], axis=0))
data_neuro['spcg_diff'] = spcg_diff
h_fig, h_axes = plot_PSTH(t_window_plot=t_range, signal_type=signal_type, data_neuro=data_neuro)

orients = [0,23,45,68,90,113,135,158]
flim = [45, 55]
tlim = [0.4, 0.7]
t_index = np.logical_and(spcg_t>=tlim[0],spcg_t<=tlim[1])
f_index = np.logical_and(spcg_f>=flim[0],spcg_f<=flim[1])
spcg_values = {}
for orient in orients:
    filter0 = np.logical_and(data_df['stimulate_type'] == 0, np.in1d(data_df['SampleOrientation'], orient))
    filter1 = np.logical_and(data_df['stimulate_type'] == 6, np.in1d(data_df['SampleOrientation'], orient))
    spcg_diff = np.log(np.mean(spcg[filter1, :, :, :], axis=0)) - np.log(np.mean(spcg[filter0, :, :, :], axis=0))
    for ch in range(32):
        temp = np.log(spcg[:,f_index,ch,:])
        temp = np.mean(np.mean(temp[:,:,t_index],axis=2),axis=1)
        spcg_values[ch,orient] = np.mean(temp[filter1])-np.mean(temp[filter0])

h_fig, h_axes = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
h_fig.set_size_inches([10, 9], forward=True)
plt.tight_layout()

for ch in range(32):
    plt.axes(h_axes[ch])
    for orient in orients:
        plt.plot(orient,spcg_values[ch,orient],'o')


""" Coherence """
# calculate coherence (intermediate variable)
ch_list0 = [3]
ch_list1 = range(32)
spcg_multipair = pna.ComputeSpcgMultiPair(data_neuro_LFP['data'], ch_list0, ch_list1, fs=data_neuro_LFP['signal_info']['sampling_rate'][0],
                                          t_ini=data_neuro_LFP['ts'][0], t_bin=0.200, t_step=None, batchsize=100,f_lim=[5,200], tf_verbose=True)
flim = [35,55]
tlim = [0.4,0.7]
t_index = np.logical_and(spcg_multipair['spcg_t']>=tlim[0],spcg_multipair['spcg_t']<=tlim[1])
f_index = np.logical_and(spcg_multipair['spcg_f']>=flim[0],spcg_multipair['spcg_f']<=flim[1])
cohg_values = {}
orients = [0,23,45,68,90,113,135,158]
for orient in orients:
    filter_trial = np.logical_and( data_df['stimulate_type']==0, np.in1d(data_df['SampleOrientation'],orient))
    spcg_result = pna.ComputeCohgFromIntermediate(spcg_multipair, limit_trial=filter_trial, tf_phase=False)
    for ch in spcg_result['ch_list1']:
        cohg_values[ch,orient] = np.mean(np.mean(spcg[np.ix_(f_index,t_index)],axis=1))


h_fig, h_axes = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
h_fig.set_size_inches([10, 9], forward=True)
plt.tight_layout()

for ch in spcg_result['ch_list1']:
    plt.axes(h_axes[ch])
    for orient in orients:
        plt.plot(orient,cohg_values[ch,orient],'o')
plt.ylim([0,1])


orient = [0,23,45,68,90,113,135,158]
filter_trial = np.logical_and( data_df['stimulate_type']==6, np.in1d(data_df['SampleOrientation'],orient))
spcg_result = pna.ComputeCohgFromIntermediate(spcg_multipair, limit_trial=filter_trial, tf_phase=True)
plot_PSTH(t_window_plot=t_range, signal_type=signal_type, data_neuro=data_neuro,spcg_result=spcg_result)