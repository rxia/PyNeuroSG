
import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
pd.set_option('display.max_columns', None)
import dg2df
import neo
from neo.core import (Block, Segment, ChannelIndex, AnalogSignal, Unit)
import matplotlib as mpl
import matplotlib.pyplot as plt
import standardize_TDT_blk
from standardize_TDT_blk import select_obj_by_attr
import quantities as pq
from signal_align import signal_align_to_evt
import re
import PyNeuroPlot as pnp
import time
from scipy import signal
from PyNeuroPlot import center2edge


# read
# dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/GM32-161012-160949'
# name_tdt_blocks = ['d_V4_spot_101216002', 'd_V4_spot_101216003']
dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/GM32-161014-151720'
name_tdt_blocks = ['d_srv_mask_101416004']
blk = Block()
reader = neo.io.TdtIO(dirname=dir_tdt_tank)
for name_tdt_block in name_tdt_blocks:
    seg = reader.read_segment(blockname=name_tdt_block, sortname='PLX')
    blk.segments.append(seg)



# read dg file
dir_dg  = '/Volumes/Labfiles/projects/analysis/shaobo/data_dg'
# file_dgs = ['d_V4_spot_101216002.dg', 'd_V4_spot_101216003.dg']
file_dgs = ['d_srv_mask_101416004.dg']

data_dfs = []
for file_dg in file_dgs:
    path_dg = os.path.join(dir_dg, file_dg)
    data_df = dg2df.dg2df(path_dg)
    data_dfs.append(data_df)
data_df = pd.concat(data_dfs)
data_df = data_df.reset_index(range(len(data_df)))


# get ts_align
blk_StimOn = []
for i in range(len(data_dfs)):
    print i
    data_df_segment = data_dfs[i]
    id_Obsv = np.array(data_df_segment['obsid'])      # !!! needs to be modified if multiple dg files are read
    tos_StimOn = np.array(data_df_segment['stimon'])  # tos: time of offset
    ts_ObsvOn = select_obj_by_attr(blk.segments[i].events, attr='name', value='obsv')[0].times
    ts_StimOn = ts_ObsvOn[np.array(id_Obsv)] + tos_StimOn * pq.ms
    blk_StimOn.append(ts_StimOn)

# align
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.020, 0.120], type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000); print(time.time()-t)
# import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.0200, 0.0700], type_filter='ana.*', name_filter='LFPs 6$', spike_bin_rate=1000); print(time.time()-t)
# group
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.neuro_sort(data_df, ['stim_pos_y','stim_pos_x'], [], data_neuro); print(time.time()-t)
# plot
import PyNeuroPlot as pnp; reload(pnp); t=time.time(); pnp.NeuroPlot(data_neuro, sk_std=0.005,tf_legend=False, tf_seperate_window=True); print(time.time()-t)


# align
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 0.600], type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000); print(time.time()-t)
# import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 0.600], type_filter='ana.*', name_filter='LFPs.*', spike_bin_rate=1000); print(time.time()-t)
# group
data_df['mask_opacity_int'] = np.round(data_df['mask_opacity']*100).astype(int)
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro); print(time.time()-t)
# plot
import PyNeuroPlot as pnp; reload(pnp); t=time.time(); pnp.NeuroPlot(data_neuro, sk_std=0.005,tf_legend=False, tf_seperate_window=False); print(time.time()-t)


# ==========
# decoding using spkikes
from sklearn import svm
from sklearn import cross_validation
from sklearn.preprocessing import normalize


data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 0.500], type_filter='spiketrains.*', name_filter='.*Code[1-9]', spike_bin_rate=50)
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro); pnp.NeuroPlot(data_neuro)

# Y_train = np.array(data_df['stim_names'].tolist())
Y_train = np.array(data_df['mask_orientation'].tolist())

N_window_smooth = 7
data_neuro['data'] = sp.signal.convolve(data_neuro['data'], np.ones([1,N_window_smooth,1]), 'same')

X_normalize = ( data_neuro['data'] - np.mean( data_neuro['data'] , axis=(0,1), keepdims=True ) ) /np.std( data_neuro['data'] , axis=(0,1), keepdims=True )


# for C in [0.1,1,10,100]:
for C in [1]:
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=C)
    time_tic = time.time()
    [N_tr, N_ts, N_sg] = data_neuro['data'].shape
    N_cd = len(data_neuro['cdtn'])
    clf_score = np.zeros([N_cd, N_ts])
    clf_score_std = np.zeros([N_cd, N_ts])
    for i in range(N_cd):
        cdtn = data_neuro['cdtn'][i]
        indx = np.array(data_neuro['cdtn_indx'][cdtn])
        print(cdtn)
        for t in range(N_ts):
            cfl_scores = cross_validation.cross_val_score(clf, X_normalize[indx, t, :], Y_train[indx], cv=5)
            clf_score[i, t] = np.mean(cfl_scores)
            clf_score_std[i, t] = np.std(cfl_scores)
            # clf_score[i, t] = np.mean(cross_validation.cross_val_score(clf, normalize(data_neuro['data'][indx, t, :]), data_df['mask_names'][indx].tolist(), cv=5))
    print(time.time()-time_tic)

    fig = plt.figure(figsize=(16,9))
    fig.canvas.set_window_title('C={}'.format(C))
    for i in range(N_cd):
        plt.subplot(2,3,i+1)
        plt.fill_between(data_neuro['ts'], clf_score[i,:]-clf_score[i,:]/5, clf_score[i,:]+clf_score[i,:]/5, alpha=0.5)
        plt.plot(data_neuro['ts'],clf_score[i,:])
        plt.title(data_neuro['cdtn'][i])
        plt.ylim([0,1])
    plt.show()
    fig.canvas.manager.window.raise_()


# ==========
# spectroagram
data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 0.500], type_filter='ana.*', name_filter='LFPs.*', spike_bin_rate=50)
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro); pnp.NeuroPlot(data_neuro)
[spcg_f,spcg_t,spcg] = signal.spectrogram(data_neuro['data'], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
spcg_t = np.array(spcg_t) + np.array( data_neuro['ts'][0] )
plt.pcolormesh(spcg_t, spcg_f, np.mean(spcg,axis=0)[:,0,:])
plt.ylim(0,100)

spcg_cdtn = []
for i in range(len(data_neuro['cdtn'])):
    spcg_cdtn.append(np.mean(spcg[data_neuro['cdtn_indx'][data_neuro['cdtn'][i]],:,:,:],axis=0))
clim_max = [np.stack(spcg_cdtn,axis=-1)[:,j,:,:].max() for j in range(32)]
for j in range(32):
    fig = plt.figure(figsize=(16,9))
    fig.canvas.set_window_title('chan_{}'.format(j+1))
    fig.suptitle(data_neuro['grpby'])
    for i in range(len(data_neuro['cdtn'])):
        plt.subplot(2,3,i+1)
        plt.pcolormesh(center2edge(spcg_t), center2edge(spcg_f), spcg_cdtn[i][:,j,:], cmap=plt.get_cmap('inferno'))
        plt.clim(0, clim_max[j])
        if True:
            plt.pcolormesh(center2edge(spcg_t), center2edge(spcg_f), 10*np.log10(spcg_cdtn[i][:, j, :]/clim_max[j]), cmap=plt.get_cmap('inferno'))
            # plt.colorbar()
            plt.clim(-30, 0)
        plt.ylim(0, 80)
        plt.title(data_neuro['cdtn'][i])
        # if i==len(data_neuro['cdtn'])-1:
        #     plt.colorbar()
    plt.get_current_fig_manager().window.raise_()
    plt.show()


""" active task """
dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/GM32-161014-151720'
name_tdt_blocks = ['d_matchnot_101416005','d_matchnot_101416005']
blk = Block()
reader = neo.io.TdtIO(dirname=dir_tdt_tank)
for name_tdt_block in name_tdt_blocks:
    seg = reader.read_segment(blockname=name_tdt_block, sortname='PLX')
    blk.segments.append(seg)

# read dg file
dir_dg  = '/Volumes/Labfiles/projects/analysis/shaobo/data_dg'
# file_dgs = ['d_V4_spot_101216002.dg', 'd_V4_spot_101216003.dg']
file_dgs = ['d_matchnot_101416005.dg', 'd_matchnot_101416005.dg']

data_dfs = []
for file_dg in file_dgs:
    path_dg = os.path.join(dir_dg, file_dg)
    data_df = dg2df.dg2df(path_dg)
    data_dfs.append(data_df)
data_df = pd.concat(data_dfs)
data_df = data_df.reset_index(range(len(data_df)))


# get timestamps to align data with
# for prf task, we use stim on
id_Obsv = np.array(data_df['obsid'])      # !!! needs to be modified if multiple dg files are read
tos_StimOn = np.array(data_df['stimon'])  # tos: time of offset
ts_ObsvOn = select_obj_by_attr(blk.segments[0].events, attr='name', value='obsv')[0].times
ts_StimOn = ts_ObsvOn[np.array(id_Obsv)] + tos_StimOn * pq.ms


# align
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.200, 1.500], type_filter='spiketrains.*', name_filter='.*Code[1-9]', spike_bin_rate=1000); print(time.time()-t)
# group
data_df['mask_opacity_int'] = np.round(data_df['MaskOpacity']*100).astype(int)
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.neuro_sort(data_df, ['mask_opacity_int'], [], data_neuro); print(time.time()-t)
# plot
import PyNeuroPlot as pnp; reload(pnp); t=time.time(); pnp.NeuroPlot(data_neuro, sk_std=0.005, tf_seperate_window=True); print(time.time()-t)



