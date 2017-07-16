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

dir_tdt_tank='/shared/lab/projects/encounter/data/TDT'
dir_dg='/shared/lab/projects/analysis/diana'

keyword_block = 'y_MTst.*'
keyword_tank = '.*Murphy-170705.*'

f_lim = [35,55]
signal_type = 'spcg'
task_type = 'MTstar'
if task_type=='spot':
    keyword_block= 'x.*spotopto.*'
    group_by = ['stim_pos_x', 'stim_pos_y']
    t_range = [-0.2, 0.6]
elif task_type == 'MTstar':
    keyword_block = 'y.*MTstar.*'
    group_by = ['stim_pos_x', 'stim_pos_y']
    t_range = [-0.2, 0.6]


""" load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
if load_files:
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(keyword_block, keyword_tank, tf_interactive=False,
                                                               dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg)
    if task_type=='colorxgrating':
        data_df['orient'] = np.array(data_df['stim_names'].str.extract('fftnoise_(\d\d\d)_.*'),
                                           dtype=float) / 180 * np.pi
    if task_type=='MTS':
        data_df['stimulate_type'] = data_df['StimulateType']

    # Get StimOn time stamps in neo time frame
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

    # some settings for saving figures
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'

    # make sure data field exists
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)


""" data align """
data_neuro_spk = signal_align.blk_align_to_evt(blk, ts_StimOn, t_range, type_filter='spiketrains.*',
                                           name_filter='.*Code[1-9]$', spike_bin_rate=1000)

data_neuro_LFP = signal_align.blk_align_to_evt(blk, ts_StimOn, t_range, type_filter='ana.*', name_filter='LFPs.*')

# calculate power spectrum
if 'spcg' in signal_type:
    [spcg, spcg_t, spcg_f] = pna.ComputeSpectrogram(data_neuro_LFP['data'], fs=data_neuro_LFP['signal_info']['sampling_rate'][0],
                           t_ini=data_neuro_LFP['ts'][0], t_bin=0.200, t_step=None, t_axis=1, batchsize=100,
                           f_lim=[5,200])

data_neuro_spk = signal_align.neuro_sort(data_df, group_by, [], data_neuro_spk)
data_neuro_LFP = signal_align.neuro_sort(data_df, group_by, [], data_neuro_LFP)

f_lim = [70,100]

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


def plot_RF(t_window_plot=t_range, signal_type=signal_type, data_neuro = data_neuro):
    h_fig, h_axes = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
    h_fig.set_size_inches([10, 9], forward=True)
    plt.tight_layout()

    for i in sorted(range(len(data_neuro['signal_info'])), reverse=True):
        ch = data_neuro['signal_info'][i]['channel_index']
        if ch <= 32:
            plt.axes(h_axes[ch - 1])
            pnp.RfPlot(data_neuro, indx_sgnl=i, t_focus=t_window_plot,
                   psth_overlay=False, tf_scr_ctr=True)

plot_RF(t_window_plot=[0.00,0.50], signal_type=signal_type,data_neuro = data_neuro)
plt.suptitle('{}_{}_{}'.format(filename_common, signal_type, f_lim))