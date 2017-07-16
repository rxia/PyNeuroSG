"""  script to test copulat model to capture the correlation """

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
import datetime
import pickle

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
dir_dg='/shared/lab/projects/analysis/shaobo/data_dg'


keyword_block = 'd_.*srv.*'
keyword_tank = '.*GM32.*U16.*161125.*'

[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(keyword=keyword_block, keyword_tank=keyword_tank,
                                                           tf_interactive=True, dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg)

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


data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.100, 0.600], type_filter='spiketrains.*',
                                           name_filter='.*Code[1-9]$', spike_bin_rate=1000)
data_neuro = signal_align.neuro_sort(data_df, ['stim_sname'], [], data_neuro)

pnp.PsthPlot(data_neuro['data'], ts=data_neuro['ts'], sk_std=0.010, color_style='continuous')

t_focus = [0.030, 0.400]

data_neuro = signal_align.select_signal(data_neuro, chan_filter=np.arange(33, 48+1), sortcode_filter=(2,3))

spk_count = pna.AveOverTime(data_neuro['data'], ts=data_neuro['ts'], t_range=[0.050, 0.250], tf_count=True)


N_tr, N_ch = spk_count.shape
spk_hist2d_list = []
for i in range(N_ch):
    for j in range(N_ch):
        spk_hist2d, _, _ = np.histogram2d(spk_count[:,i], spk_count[:,j], bins=range(20))
        spk_hist2d_list.append(spk_hist2d)

reload(pnp)
pnp.DataFastSubplot(spk_hist2d_list, data_type='mesh')
