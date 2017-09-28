
""" Import modules """
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



tankname = '.*GM32.*U16.*161125'
try:
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv.*.*', tankname, tf_interactive=False,
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


data_neuro_spk = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                               name_filter='.*Code[1-9]$', spike_bin_rate=1000,
                                               chan_filter=range(1, 32 + 1))
data_neuro_spk = signal_align.neuro_sort(data_df, ['stim_sname', 'mask_names', 'mask_opacity_int'],
                                         data_df['stim_familiarized']==0, data_neuro_spk)


""" use a time period to train lda model (supervised dimemsionality reduction method) """
ts = data_neuro_spk['ts']
Xs = pna.GroupAve(data_neuro_spk)
Xs = pna.SmoothTrace(Xs, ts=ts, sk_std=0.010)
X = np.mean(Xs[:,np.logical_and(ts>0.05, ts<=0.35),:], axis=1)
Y  = data_neuro_spk['cdtn']

y0, y1, y2 = zip(*Y)
y2 = np.array(y2)
y2_unique = np.unique(y2)

model0 = pna.DimRedLDA(X, y0, dim=1, return_model=True)   # use stim_sname to get the first dim
model1 = pna.DimRedLDA(X, y1, dim=1, return_model=True)   # use mask_names to get another dim

def GenColorForColm(cdtn, d=0):
    """ coloring scheme """
    item = zip(*cdtn)[d]
    dist_item = list(set(item))
    N = len(dist_item)
    dist_color = pnp.gen_distinct_colors(len(dist_item))
    item_indx = dict( zip(dist_item, range(N)) )
    return dist_color[ [item_indx[it] for it in item], :]

"""  evaluate at multiple time points and  plot """
t_dur=0.10
h_fig, h_axes = plt.subplots(3,2, sharex=True, sharey=True, figsize=[8,8])
for i_t, t in enumerate(np.arange(-0.05, 0.5, 0.025)) :
    X = np.mean(Xs[:,np.logical_and(ts>t-t_dur/2, ts<=t+t_dur/2),:], axis=1)
    temp0 = pna.DimRedLDA(lda=model0, X_test=X)
    temp1 = pna.DimRedLDA(lda=model1, X_test=X)
    # temp = pna.DimRedLDA(X, zip(*Y)[0], dim=2)
    # temp0= temp[:,0]
    # temp1= temp[:,1]

    for row in range(3):
        plt.axes(h_axes[row,0])
        plt.cla()
        plt.scatter(temp0[y2 == y2_unique[row]], temp1[y2 == y2_unique[row]],
                    c=GenColorForColm(Y, d=0)[y2 == y2_unique[row]])
        plt.title('noise level: {} by image'.format(y2_unique[row]))
        plt.axes(h_axes[row, 1])
        plt.cla()
        plt.scatter(temp0[y2 == y2_unique[row]], temp1[y2 == y2_unique[row]],
                    c=GenColorForColm(Y, d=1)[y2 == y2_unique[row]])
        plt.title('noise level: {} by noise'.format(y2_unique[row]))
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
    plt.suptitle('t={}'.format(t))
    plt.savefig('temp_figs/LDA_spike_trains_{:0>2d}.png'.format(i_t))
    plt.pause(1.0)
