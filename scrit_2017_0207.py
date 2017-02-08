# ----- standard modules -----
import os
import sys
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


from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge


plt.ioff()

""" load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
name_file = 'd_.V4_spot.*'
[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(name_file, '.*GM32.*170207.*', tf_interactive=True,)

""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'

""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk     = data_load_DLSH.standardize_blk(blk)


if re.match('.*spot.*', filename_common) is not None:
    # align, RF plot
    data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.020, 0.200], type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=100)
    # group
    data_neuro=signal_align.neuro_sort(data_df, ['stim_pos_x','stim_pos_y'], [], data_neuro)
    # plot
    pnp.RfPlot(data_neuro, indx_sgnl=0, x_scale=0.2)
    for i in range(len(data_neuro['signal_info'])):
        pnp.RfPlot(data_neuro, indx_sgnl=i, x_scale=0.2, y_scale=100)
        try:
            plt.savefig('{}/{}_RF_{}.png'.format(dir_temp_fig, filename_common, data_neuro['signal_info'][i]['name']))
        except:
            plt.savefig('./temp_figs/RF_plot_' + misc_tools.get_time_string() + '.png')
        plt.close()
