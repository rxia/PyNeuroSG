""" script to test whether it is possbile to do CSD analysis from single electrode """

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





""" load data """

dir_tdt_tank='/shared/homes/sguan/Coding_Projects/support_data/Flynn_test_CSD_single_electrode/'
dir_dg = '/shared/homes/sguan/Coding_Projects/support_data/Flynn_test_CSD_single_electrode/'
name_tank = 'Flynn-170616.*'



block_type = 'srv_mask'
t_plot = [-0.100, 0.500]


def GetGroupAve(tankname, signal_type='spk'):

    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('f_.*{}.*'.format(block_type), name_tank, tf_interactive=False,
                                                           dir_tdt_tank= dir_tdt_tank,
                                                           dir_dg= dir_dg)

    " rename signal to standard name, for Wenhao's recording "
    for segment in blk.segments:
        for analogsignal in segment.analogsignals:
            if analogsignal.name[:4]=='Lfp1':
                analogsignal.name = 'LFPs' + analogsignal.name[4:]
        for spiketrain in segment.spiketrains:
            if spiketrain.name[:4]=='eNeu':
                analogsignal.name = 'spks' + analogsignal.name[4:]

    """ Get StimOn time stamps in neo time frame """
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'

    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    if signal_type=='spk':
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                                       name_filter='.*Code[1-9]$', spike_bin_rate=1000,
                                                       chan_filter=range(1, 48 + 1))
    else:
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*',
                                                       name_filter='LFPs.*',
                                                       chan_filter=range(1, 48 + 1))
    data_neuro = signal_align.neuro_sort(data_df, ['filename'], [], data_neuro)

    ts = data_neuro['ts']
    signal_info = data_neuro['signal_info']
    cdtn = data_neuro['cdtn']
    data_groupave = pna.GroupAve(data_neuro)

    return [data_groupave, ts, signal_info, cdtn]


[data_groupave, ts, signal_info, cdtn] = GetGroupAve('', signal_type='LFP')

ERPs = data_groupave[:,:,0]
ERPs = np.flipud(ERPs)
ERPs = pna.SmoothTrace(ERPs, sk_std=0.010, ts=ts, axis=1)

# pnp.ErpPlot(ERPs, ts)
pnp.ErpPlot_singlePanel(ERPs, ts)
plt.title('ERPs from single electrode {}'.format(name_tank))
plt.savefig('./temp_figs/ERPs_from_single_electrode_Flynn_170616.png')
