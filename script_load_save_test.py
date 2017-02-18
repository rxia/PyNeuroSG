
""" script to load and save file, a test """

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

from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge


from GM32_layout import layout_GM32


""" load data """
dir_tdt_tank='/shared/lab/projects/encounter/data/TDT/'
list_name_tanks = os.listdir(dir_tdt_tank)
keyword_tank = '.*GM32.*U16'
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_name_tanks_0 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is None]
list_name_tanks_1 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is not None]
list_name_tanks = sorted(list_name_tanks_0) + sorted(list_name_tanks_1)


tankname = list_name_tanks[1]
[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv_mask.*', tankname, tf_interactive=False,
                                                           dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank',
                                                           dir_dg='/shared/homes/sguan/neuro_data/stim_dg')

""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'

""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk = data_load_DLSH.standardize_blk(blk)

t_plot = [-0.100, 0.500]

data_neuro_spk = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000, chan_filter=range(1,48+1))
data_neuro_spk = signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro_spk)
# plot GM32
pnp.DataNeuroSummaryPlot(signal_align.select_signal(data_neuro_spk, chan_filter=range(1,32+1)), sk_std=0.01, signal_type='auto', suptitle='spk_GM32  {}'.format(filename_common))
# plt.savefig('{}/{}_spk_GM32.png'.format(dir_temp_fig, filename_common))
# plot U16
pnp.DataNeuroSummaryPlot(signal_align.select_signal(data_neuro_spk, chan_filter=range(33,48+1)), sk_std=0.01, signal_type='auto', suptitle='spk_U16  {}'.format(filename_common))
# plt.savefig('{}/{}_spk_U16.png'.format(dir_temp_fig, filename_common))

""" save test """
pickle.dump(data_neuro, open( '/shared/homes/sguan/Coding_Projects/support_data/temp', "wb" ))