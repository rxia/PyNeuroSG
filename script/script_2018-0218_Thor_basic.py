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
import misc_tools           # in this package: misc

import data_load_DLSH       # package specific for DLSH lab data

from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge


dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT'
dir_dg = '/shared/lab/projects/analysis/shaobo/data_dg'

[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(
    '.*007', '.*Thor.*S4.*180214.*',
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


""" spike waveforms """
pnp.SpkWfPlot(blk.segments[0], sortcode_min=0)
plt.savefig('{}/{}_spk_waveform.png'.format(dir_temp_fig, filename_common))


""" ERP plot """
t_plot = [-0.100, 0.500]

data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*', name_filter='LFPs.*')
ERP = np.mean(data_neuro['data'], axis=0).transpose()
pnp.ErpPlot(ERP, data_neuro['ts'])
plt.savefig('{}/{}_ERP_all.png'.format(dir_temp_fig, filename_common))



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