""" script to test whether the information carried by IT and V4 is congruent or independent """


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
from sklearn import svm
from sklearn import cross_validation


""" load data """
keyword_tank = '.*GM32.*U16.*161125.*'
block_type = 'srv'

[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*{}.*'.format(block_type), keyword_tank,
                                                           tf_interactive=False,
                                                           dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank',
                                                           dir_dg='/shared/homes/sguan/neuro_data/stim_dg')

ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'

""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk = data_load_DLSH.standardize_blk(blk)

t_plot = [-0.100, 0.500]
spike_bin_interval = 0.050
data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                           name_filter='.*Code[1-9]$', spike_bin_rate=1 / spike_bin_interval,
                                           chan_filter=range(1, 48 + 1))

data_neuro = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int', ''], [], data_neuro)

data_neuro['ts'] = data_neuro['ts'] + spike_bin_interval / 2
ts = data_neuro['ts']
signal_info = data_neuro['signal_info']
cdtn = data_neuro['cdtn']

""" decode """


def decode_activity(data_neuro, target_name='stim_names'):
    X_normalize = (data_neuro['data'] - np.mean(data_neuro['data'], axis=(0, 1), keepdims=True)) / np.std(
        data_neuro['data'], axis=(0, 1), keepdims=True)

    clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=1)
    [N_tr, N_ts, N_sg] = data_neuro['data'].shape
    N_cd = len(data_neuro['cdtn'])
    clf_score = np.zeros([N_cd, N_ts])
    clf_score_std = np.zeros([N_cd, N_ts])
    for i in range(N_cd):
        cdtn = data_neuro['cdtn'][i]
        indx = np.array(data_neuro['cdtn_indx'][cdtn])
        for t in range(N_ts):
            cfl_scores = cross_validation.cross_val_score(clf, X_normalize[indx, t, :],
                                                          data_df[target_name][indx].tolist(), cv=5)
            clf_score[i, t] = np.mean(cfl_scores)
            clf_score_std[i, t] = np.std(cfl_scores)
    return clf_score


decode_GM32_image = decode_activity(signal_align.select_signal(data_neuro, chan_filter=np.arange(1, 32 + 1)),
                                    'stim_names')
decode_GM32_noise = decode_activity(signal_align.select_signal(data_neuro, chan_filter=np.arange(1, 32 + 1)),
                                    'mask_orientation')
decode_U16_image = decode_activity(signal_align.select_signal(data_neuro, chan_filter=np.arange(33, 48 + 1)),
                                   'stim_names')
decode_U16_noise = decode_activity(signal_align.select_signal(data_neuro, chan_filter=np.arange(33, 48 + 1)),
                                   'mask_orientation')

decode_result = np.dstack([decode_GM32_image, decode_GM32_noise, decode_U16_image, decode_U16_noise])