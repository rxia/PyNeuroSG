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
import sklearn

import data_load_DLSH       # package specific for DLSH lab data

from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge
import sklearn
from sklearn import svm
from sklearn import cross_validation


keyword_tank = '.*GM32.*U16.*161222.*'
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

t_plot = [0.000, 0.400]
spike_bin_interval = 0.010
data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                           name_filter='.*Code[1-9]$', spike_bin_rate=1 / spike_bin_interval,
                                           chan_filter=range(1, 48 + 1))
data_neuro['data'] = pna.SmoothTrace(data_neuro['data'], sk_std=0.015, fs=1/spike_bin_interval)
grpby = ['stim_familiarized', 'stim_sname', 'mask_opacity_int']

data_neuro = signal_align.neuro_sort(data_df, grpby, [], data_neuro)
# data_neuro = signal_align.neuro_sort(data_df, ['', 'mask_opacity_int', ''], [], data_neuro)


data_neuro['ts'] = data_neuro['ts'] + spike_bin_interval / 2
ts = data_neuro['ts']
signal_info = data_neuro['signal_info']
cdtn = data_neuro['cdtn']


N, T, M = data_neuro['data'].shape


""" group average """
psth_grpave_raw = pna.GroupAve(data_neuro)[:, :, data_neuro['signal_info']['channel_index']>32]
psth_grpave = ( psth_grpave_raw - np.mean(psth_grpave_raw, axis=2, keepdims=True) ) \
              / ( np.std(psth_grpave_raw, axis=2, keepdims=True)+10 )

C = psth_grpave.shape[0]
def data_unpack_time(data, axis=1):
    N, T, M = data.shape
    return np.reshape(data, [N*T, M])

def data_pack_time(data, N=N):
    NT, M = data.shape
    return np.reshape(data, [N, NT//N, M])

psth_grpave_2D = data_unpack_time(psth_grpave)

K = 2
pca = sklearn.decomposition.PCA(n_components=K)

psth_ns_2D = pca.fit_transform(psth_grpave_2D)
psth_ns = data_pack_time(psth_ns_2D, N=C)


cdtn_stim = zip(*cdtn)[1]
cdtn_stim_unq = np.unique(cdtn_stim)
color_unq = pnp.gen_distinct_colors(len(cdtn_stim_unq))
color_dict = dict([(cdtn_stim_unq[i], color_unq[i]) for i in range(len(cdtn_stim_unq))])
color_cdtn = np.array([color_dict[s] for s in cdtn_stim])
# color_cdtn[:,3] = (100-np.array(zip(*cdtn)[2]))*0.01

h_fig, h_ax = plt.subplots(2,3, sharex=True, sharey=True)
for i, c in enumerate(cdtn):
    if c[0]==0:
        h_row = 0
    else:
        h_row = 1
    if c[2]==0:
        h_col = 0
    elif c[2]==50:
        h_col = 1
    else:
        h_col = 2
    plt.axes(h_ax[h_row, h_col])
    plt.plot(psth_ns[i,:,0], psth_ns[i,:,1], color=color_cdtn[i])

plt.plot(psth_ns[:,:t,0].transpose(), psth_ns[:,:t,1].transpose())
for t in range(psth_ns.shape[1]):
    plt.cla()
    plt.plot(psth_ns[:,:t,0].transpose(), psth_ns[:,:t,1].transpose())
    plt.show()
    plt.pause(0.1)
