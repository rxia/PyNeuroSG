""" script to test spectral analysis, 2017-0310 """


import os
import sys
import numpy as np
import scipy as sp
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


""" ========== load data ========== """
tankname = '.*GM32.*U16.*161125'
try:
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv_mask.*', tankname, tf_interactive=False,
                                                               dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank',
                                                               dir_dg='/shared/homes/sguan/neuro_data/stim_dg')
except:
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv_mask.*', tankname, tf_interactive=False)

""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'

""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk = data_load_DLSH.standardize_blk(blk)

t_plot = [-0.100, 0.500]

data_neuro_LFP = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='analog.*',
                                               name_filter='LFPs.*', spike_bin_rate=1000,
                                               chan_filter=range(1, 48 + 1))
data_neuro_LFP = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro_LFP)

ts = data_neuro_LFP['ts']
fs = data_neuro_LFP['signal_info'][0]['sampling_rate']
signal_info = data_neuro_LFP['signal_info']
cdtn = data_neuro_LFP['cdtn']
erp_groupave = pna.GroupAve(data_neuro_LFP)


""" spectratrum """
[spct, spcg_f] = pna.ComputeWelchSpectrum(data_neuro_LFP['data'], fs=fs, t_ini=ts[0], t_window=[0.050, 0.350], t_bin=0.1, t_step=None,
                         t_axis=1, batchsize=100, f_lim=[0,120])
spct_groupave = pna.GroupAve(data_neuro_LFP, spct)


for ch in range(1,48+1):
    h_fig, h_axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    plt.axes(h_axes[0])
    colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9), pnp.gen_distinct_colors(3, luminance=0.6)])
    linestyles = ['-', '-', '-', '--', '--', '--']
    for i in range(len(cdtn)):
        plt.plot( ts, erp_groupave[i,:,ch-1], color=colors[i], linestyle=linestyles[i])
    plt.legend(cdtn)

    plt.axes(h_axes[1])
    for i in range(len(cdtn)):
        plt.plot( spcg_f, np.log(spct_groupave[i,:,ch-1]), color=colors[i], linestyle=linestyles[i])
    plt.legend(cdtn)

    plt.suptitle('{}: chan {}'.format(filename_common, ch))
    plt.savefig('{}/ERP_spectrum_{}_ch{:0>2}.png'.format(dir_temp_fig, filename_common, ch))
    plt.close()


"""  """
