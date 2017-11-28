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
import statsmodels.api as sm


import data_load_DLSH       # package specific for DLSH lab data

from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge
import sklearn
from sklearn import svm
from sklearn import cross_validation


keyword_tank = '.*GM32.*U16.*161228.*'
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

t_plot = [-0.600, 1.200]
spike_bin_interval = 0.010
sk_std = None
data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                           name_filter='.*Code[1-9]$', spike_bin_rate=1 / spike_bin_interval,
                                           chan_filter=range(1, 48 + 1))
data_neuro['data'] = pna.SmoothTrace(data_neuro['data'], sk_std=sk_std, fs=1/spike_bin_interval)
grpby = ['stim_familiarized', 'mask_opacity_int']

data_neuro = signal_align.neuro_sort(data_df, grpby, [], data_neuro)
# data_neuro = signal_align.neuro_sort(data_df, ['', 'mask_opacity_int', ''], [], data_neuro)

data = data_neuro['data']
data_neuro['ts'] = data_neuro['ts'] + spike_bin_interval / 2
ts = data_neuro['ts']
signal_info = data_neuro['signal_info']
cdtn = data_neuro['cdtn']
cdtn_indx = data_neuro['cdtn_indx']


N, T, M = data_neuro['data'].shape


""" group average """
def data_unpack_time(data, axis=1):
    N, T, M = data.shape
    return np.reshape(data, [N*T, M])

def data_pack_time(data, N=N):
    NT, M = data.shape
    return np.reshape(data, [N, NT//N, M])

psth_grpave_raw = pna.GroupAve(data_neuro)[:, :, data_neuro['signal_info']['channel_index']>32]
psth_grpave = ( psth_grpave_raw - np.mean(psth_grpave_raw, axis=2, keepdims=True) ) \
              / ( np.std(psth_grpave_raw, axis=2, keepdims=True)+10 )

C = psth_grpave.shape[0]

psth_grpave_2D = data_unpack_time(psth_grpave)

K = 3
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



""" PCA dimension """
data_3D = data_neuro['data']
data_2D = data_unpack_time(data_3D)
K = 2
pca = sklearn.decomposition.PCA()
pca.fit(data_2D)

h_fit, h_ax = plt.subplots(nrows=2, ncols=1, figsize=[4,6])
plt.axes(h_ax[0])
plt.plot(pca.explained_variance_ratio_)
plt.title('explained variance ratio')
plt.axes(h_ax[1])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('cumulative explained variance ratio')
plt.xlabel('num dimensions')


""" VAR code """
def flatten_x_for_regress(x, p=p):
    """ turn [p,M,M] to  """
    x = np.array(x)
    T, M = np.shape(x)
    x_flat = np.zeros([T-p, M])
    y_flat = x[p:]
    x_flat = np.concatenate([x[p-pp-1:T-1-pp] for pp in range(p)], axis=1)
    return x_flat, y_flat

def flatten_X_for_regress(X, p=p):
    N, T, M = X.shape
    X_for_reg, Y_for_reg = zip(*[flatten_x_for_regress(x, p=p) for x in X])
    X_for_reg = np.concatenate(X_for_reg, axis=0)
    Y_for_reg = np.concatenate(Y_for_reg, axis=0)
    return X_for_reg, Y_for_reg

def VAR_fit(X, p=p):
    N, T, M = X.shape
    X_for_reg, Y_for_reg = flatten_X_for_regress(X, p=p)   # reformat data for linear regression
    # lr = linear_model.LinearRegression()
    lr = linear_model.RidgeCV()
    lr.fit(X_for_reg, Y_for_reg)
    phi_hat = np.stack(np.split(lr.coef_, p, axis=1), axis=0)  # recover phi and c in the orignal format
    c_hat = lr.intercept_
    Y_hat_reg = lr.predict(X_for_reg)                          # get predicted values based on history term (VAR model)
    X_hat = X*np.nan
    X_hat[:,p:,:] = np.stack(np.split(Y_hat_reg, N, axis=0))   # re-format the predicted value in it's original structure as X
    return phi_hat, c_hat, X_hat

def VAR_esp(X, p=p, m_subset0=1):
    """ compute the noise residue that VAR fails to capture """
    X0 = X[:,:, :m_subset0]
    X1 = X[:,:, m_subset0:]
    phi_hat_full, c_hat_full, X_hat_full = VAR_fit(X, p=p)   # fit on full model
    phi_hat_0, c_hat_0, X_hat_0 = VAR_fit(X0, p=p)           # fit model only using ch subset 0
    phi_hat_1, c_hat_1, X_hat_1 = VAR_fit(X1, p=p)           # fit model only using ch subset 1
    eps_full = X_hat_full-X
    eps0 = X_hat_0 - X0
    eps1 = X_hat_1 - X1
    return eps_full, eps0, eps1

def VAR_GC(eps_full=None, eps0=None, eps1=None, X=None, p=p, m_subset0=1, n_tf=None):
    """ granger causality """
    if X is not None:
        eps_full, eps0, eps1 = VAR_esp(X, p=p, m_subset0=m_subset0)
    if n_tf is not None:
        eps_full = eps_full[n_tf, :, :]
        eps0 = eps0[n_tf, :, :]
        eps1 = eps1[n_tf, :, :]
    m_subset0 = eps0.shape[2]
    gc_1to0 = np.log( np.nanmean(eps0**2, axis=(0,2)) / np.nanmean(eps_full[:, :, :m_subset0]**2, axis=(0,2)))
    gc_0to1 = np.log( np.nanmean(eps1**2, axis=(0,2)) / np.nanmean(eps_full[:, :, m_subset0:]**2, axis=(0,2)))
    return gc_0to1, gc_1to0


""" use VAR to compute GC """
K0, K1 = 4, 4
p = 11

data0 = data[:, :, signal_info['channel_index']>39]
data1 = data[:, :, (signal_info['channel_index']<39) * (signal_info['channel_index']>32)]

if False:   # substract mean
    data0 = data0 - np.mean(data0, axis=0, keepdims=0)
    data1 = data1 - np.mean(data1, axis=0, keepdims=0)
N = data.shape[0]

pca_window = [0.020, 0.370]
ts_tf_pca = (ts>=pca_window[0]) * (ts<=pca_window[1])
if K>0:
    pca0 = sklearn.decomposition.PCA(n_components=K0)
    pca0.fit(np.mean(data0[:, ts_tf_pca, :], axis=1))
    X0 = data_pack_time(pca0.transform(data_unpack_time(data0)), N)
    pca1 = sklearn.decomposition.PCA(n_components=K1)
    pca1.fit(np.mean(data1[:, ts_tf_pca, :], axis=1))
    X1 = data_pack_time(pca1.transform(data_unpack_time(data1)), N)
else:
    X0 = data0
    X1 = data1
X = np.concatenate([X0, X1], axis=2)

eps_full, eps0, eps1 = VAR_esp(X, p=p, m_subset0=X0.shape[2])

smooth_size = 9
h_fig, h_ax = plt.subplots(2,3, sharex=True, sharey=True)
h_ax = h_ax.ravel()
for i, c_cur in enumerate(cdtn):
    gc_01, gc_10 = VAR_GC(eps_full, eps0, eps1, p=3, m_subset0=X0.shape[2], n_tf=cdtn_indx[c_cur])
    plt.axes(h_ax[i])
    plt.plot(ts, np.convolve(gc_01, np.ones(smooth_size)/smooth_size,  mode='same'))
    plt.plot(ts, np.convolve(gc_10, np.ones(smooth_size)/smooth_size,  mode='same'))
    plt.axvspan(0.0, 0.3, color='k', alpha=0.1)
    plt.axvspan(0.6, 0.9, color='k', alpha=0.1)
    plt.axhline(0, color='k')
plt.legend(['01','10'])
plt.xlim([-0.1,0.4])








""" ========== ========== ========== ========== """
""" process data for a day """

keyword_tank = '.*GM32.*U16.*161228.*'
block_type = 'matchnot'

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

t_plot = [-0.200, 0.600]
spike_bin_interval = 0.020
sk_std = None
data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                           name_filter='.*Code[1-9]$', spike_bin_rate=1 / spike_bin_interval,
                                           chan_filter=range(1, 48 + 1))
data_neuro['data'] = pna.SmoothTrace(data_neuro['data'], sk_std=sk_std, fs=1/spike_bin_interval)
grpby = ['stim_familiarized', 'mask_opacity_int']

data_neuro = signal_align.neuro_sort(data_df, grpby, [], data_neuro)
# data_neuro = signal_align.neuro_sort(data_df, ['', 'mask_opacity_int', ''], [], data_neuro)

data = data_neuro['data']
data_neuro['ts'] = data_neuro['ts'] + spike_bin_interval / 2
ts = data_neuro['ts']
signal_info = data_neuro['signal_info']
cdtn = data_neuro['cdtn']
cdtn_indx = data_neuro['cdtn_indx']


N, T, M = data_neuro['data'].shape



K0, K1 = 0, 0
p = 5

data = data_neuro['data']
# data = np.stack([pna.group_shuffle_data(data_neuro['data'][:,:,m], cdtn_indx.values(), axis=0) for m in range(M)], axis=2)

data0 = data[:, :, signal_info['channel_index']>39]
data1 = data[:, :, (signal_info['channel_index']<39) * (signal_info['channel_index']>32)]

if False:   # substract mean
    data0 = data0 - np.mean(data0, axis=0, keepdims=0)
    data1 = data1 - np.mean(data1, axis=0, keepdims=0)
N = data.shape[0]

pca_window = [0.020, 0.370]
ts_tf_pca = (ts>=pca_window[0]) * (ts<=pca_window[1])
if K0>0 or K1>0:
    pca0 = sklearn.decomposition.PCA(n_components=K0)
    pca0.fit(np.mean(data0[:, ts_tf_pca, :], axis=1))
    X0 = data_pack_time(pca0.transform(data_unpack_time(data0)), N)
    pca1 = sklearn.decomposition.PCA(n_components=K1)
    pca1.fit(np.mean(data1[:, ts_tf_pca, :], axis=1))
    X1 = data_pack_time(pca1.transform(data_unpack_time(data1)), N)
else:
    X0 = data0
    X1 = data1
X = np.concatenate([X0, X1], axis=2)

eps_full, eps0, eps1 = VAR_esp(X, p=p, m_subset0=X0.shape[2])

smooth_size = 9
h_fig, h_ax = plt.subplots(2,3, sharex=True, sharey=True)
h_ax = h_ax.ravel()
for i, c_cur in enumerate(cdtn):
    gc_01, gc_10 = VAR_GC(eps_full, eps0, eps1, p=3, m_subset0=X0.shape[2], n_tf=cdtn_indx[c_cur])
    plt.axes(h_ax[i])
    plt.plot(ts, np.convolve(gc_01, np.ones(smooth_size)/smooth_size,  mode='same'))
    plt.plot(ts, np.convolve(gc_10, np.ones(smooth_size)/smooth_size,  mode='same'))
    plt.axvspan(0.0, 0.3, color='k', alpha=0.1)
    plt.axvspan(0.6, 0.9, color='k', alpha=0.1)
    plt.axhline(0, color='k')
plt.legend(['01','10'])
# plt.xlim([-0.1,0.5])