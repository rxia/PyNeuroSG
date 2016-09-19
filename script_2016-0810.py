# exec(open("./script_2016-0810.py").read())
# test script for analyzing the data collected on 2016-0810, Dante, U-probe with 16 channels

# import modules

import os
import sys
import numpy as np
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


# read dg file
dir_dg  = '/Users/Summit/Documents/neural_data/2016-0817_Dante_U16'
# dir_dg  = '/Users/Summit/Documents/neural_data/2016-0902_Dante_U16'
# dir_dg  = '/Users/Summit/Documents/neural_data/2016-0908_Dante_U16'
file_dg = 'd_srv_mask_081716002.dg'
path_dg = os.path.join(dir_dg, file_dg)

data_df = dg2df.dg2df(path_dg)

# read tdt file
# dir_tdt_tank  = '/Users/Summit/Documents/neural_data/2016-0908_Dante_U16/U16-160908-103844'
# name_tdt_block = 'd_srv_mask_090816003'
dir_tdt_tank  = '/Users/Summit/Documents/neural_data/2016-0817_Dante_U16/U16-160817-135857'
name_tdt_block = 'd_srv_mask_081716002'
# dir_tdt_tank  = '/Users/Summit/Documents/neural_data/2016-0902_Dante_U16/U16-160902-130744'
# name_tdt_block = 'd_srv_mask_090216002'
reader = neo.io.TdtIO(dirname=dir_tdt_tank)
seg = reader.read_segment(blockname=name_tdt_block, sortname='PLX')
blk = Block()
blk.segments.append(seg)
# standardize_TDT_blk.create_rcg(blk)

# get timestamps to align data with
# for prf task, we use stim on
id_Obsv = np.array(data_df['obsid'])      # !!! needs to be modified if multiple dg files are read
tos_StimOn = np.array(data_df['stimon'])  # tos: time of offset
ts_ObsvOn = select_obj_by_attr(blk.segments[0].events, attr='name', value='obsv')[0].times
ts_StimOn = ts_ObsvOn[np.array(id_Obsv)] + tos_StimOn * pq.ms
type_StimOn = data_df['mask_opacity']


# ERP plot
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.600], type_filter='ana.*', name_filter='LFPs.*'); print(time.time()-t)
reload(pnp); pnp.ErpPlot(np.mean(data_neuro['data'], axis=0).transpose(), data_neuro['ts'])

# spike waveforms:
fig, axes2d = plt.subplots(nrows=16, ncols=1, sharex=True, sharey=True, figsize=(2,16))
plt.tight_layout()
fig.subplots_adjust(hspace=0.05, wspace=0.05)
for i, axes in enumerate (axes2d):
    plt.sca(axes)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(i+1)
for i in range(len(blk.segments[0].spiketrains)):
    cur_chan = int(re.match('Chan(\d*) .*', blk.segments[0].spiketrains[i].name).group(1))
    cur_code = int(re.match('.*Code(\d*)', blk.segments[0].spiketrains[i].name).group(1))
    if cur_code>=1:
        plt.sca(axes2d[cur_chan-1])
        plt.plot(np.squeeze(np.mean(blk.segments[0].spiketrains[i].waveforms, axis=0)))
        # plt.title(blk.segments[0].spiketrains[i].name)




# sort by condition
for i in range(len(blk.segments[0].spiketrains)):
    plt.subplot(6,6,i+1); plt.plot(np.squeeze(np.mean(blk.segments[0].spiketrains[i].waveforms, axis=0)));
    plt.xticks([]); plt.yticks([]); plt.title(blk.segments[0].spiketrains[i].name)

# align
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.600], type_filter='spiketrains.*', name_filter='.*Code[1-9]', spike_bin_rate=1000); print(time.time()-t)
# group
data_df['mask_opacity_int'] = np.round(data_df['mask_opacity']*100).astype(int)
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro); print(time.time()-t)
# plot
import PyNeuroPlot as pnp; reload(pnp); t=time.time(); pnp.NeuroPlot(data_neuro, sk_std=0.005); print(time.time()-t)

# spikes
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.600], type_filter='spiketrains.*', name_filter='.*Code[1-9]', spike_bin_rate=1000); print(time.time()-t)
# LFPs
# import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.600], type_filter='ana.*', name_filter='LFPs.*'); print(time.time()-t)
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.neuro_sort(data_df, ['stim_names'], (data_df['mask_opacity_int']==0) & (data_df['stim_familiarized']==0), data_neuro); print(time.time()-t)
pnp.NeuroPlot(data_neuro, layout=[5,2], sk_std=0.005, tf_legend=True)
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.neuro_sort(data_df, ['stim_names'], (data_df['mask_opacity_int']==0) & (data_df['stim_familiarized']==1), data_neuro); print(time.time()-t)
pnp.NeuroPlot(data_neuro, layout=[5,2], sk_std=0.005, tf_legend=False)

reload(signal_align)
reload(pnp)
data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.500], type_filter='spiketrains.*', name_filter='.*Code[1-9]', spike_bin_rate=50)
# data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.600], name_filter='LFPs.*')
data_neuro=signal_align.neuro_sort(data_df, ['stim_categories', 'mask_opacity_int'], [], data_neuro); pnp.NeuroPlot(data_neuro)
# data_neuro=signal_align.neuro_sort(data_df, ['mask_orientation', 'mask_opacity_int'], [], data_neuro); pnp.NeuroPlot(data_neuro)


# ========== machine learning ==========
from sklearn.decomposition import PCA, FastICA, NMF
N_pc = 2
pca = NMF(n_components=N_pc)

def unfoldtime(X_3D):
    [N_tr, N_ts, N_sg] = X_3D.shape
    X = np.zeros([N_tr * N_ts, N_sg])
    for i in range(N_ts):
        X[i * N_tr:(i + 1) * N_tr, :] = X_3D[:, i, :]
    return X
def foldtime(X_2D, X_3D):
    [N_tr, N_ts, _] = X_3D.shape
    [_, N_dm] = X_2D.shape
    X_fold_3D = np.zeros([N_tr, N_ts, N_dm])
    for i in range(N_ts):
        X_fold_3D[:, i, :] = X_2D[i * N_tr:(i + 1) * N_tr, :]
    return X_fold_3D
def data_grpave(X, data_neuro):
    [N_tr, N_ts, N_sg] = X.shape
    X_grp = np.zeros([len(data_neuro['cdtn']), N_ts, N_sg])
    for i in range(len(data_neuro['cdtn'])):
        X_grp[i, :, :] = np.mean(X[data_neuro['cdtn_indx'][data_neuro['cdtn'][i]], :, :], axis=0)
    return X_grp
def pca_dynamics_plot(X_pca_grp, data_neuro):
    plt.figure(figsize=(12,9))
    plt.ion()
    hl = []
    [N_tr, N_ts, N_sg] = X_pca_grp.shape
    N_cdtn0 = len(np.unique([item[0] for item in data_neuro['cdtn']]))
    N_cdtn1 = len(np.unique([item[1] for item in data_neuro['cdtn']]))
    cycle_color = plt.cm.get_cmap('rainbow')(np.linspace(0, 1, N_cdtn0))
    cycle_linestyle = ['-','--','-.']
    for i in range(len(data_neuro['cdtn'])):
        hc, = plt.plot(X_pca_grp[i, :, 0], X_pca_grp[i, :, 1], lw=((i%N_cdtn1)+1)*3, c=cycle_color[i//N_cdtn1], ls=cycle_linestyle[0], alpha=1-(i%N_cdtn1)/(N_cdtn1-0.5))
        hl.append(hc)
    plt.legend(hl, data_neuro['cdtn'], loc='upper left', bbox_to_anchor=(0.9, 1))
    for j in range(N_ts):
        plt.clf
        for i in range(len(data_neuro['cdtn'])):
            hl[i].set_data(X_pca_grp[i,0:j,0],X_pca_grp[i,0:j,1])
        # plt.draw()
        plt.show()
        print(j)
        plt.title(data_neuro['ts'][j])
        plt.savefig('temp_img/ICA_dynamics_stim_{:0>3}.png'.format(j))
        plt.pause(0.05)


data_pcafit = signal_align.neuro_sort(data_df, ['stim_names'], data_df['mask_opacity_int']==70)
pca.fit(unfoldtime( data_grpave(data_neuro['data'][:,(data_neuro['ts']>0) * (data_neuro['ts']<0.4),:], data_pcafit) ))
# pca.fit(unfoldtime( data_neuro['data'] ))
X_pca = foldtime( pca.transform(unfoldtime(data_neuro['data'])), data_neuro['data'] )
# data_pcaplot = signal_align.neuro_sort(data_df, ['stim_categories','mask_names'], data_df['mask_opacity_int']==50, data_neuro)
data_pcaplot = signal_align.neuro_sort(data_df, ['mask_names','stim_categories'], data_df['mask_opacity_int']==50, data_neuro)
X_pca_grp = data_grpave(X_pca, data_pcaplot)
pca_dynamics_plot(X_pca_grp, data_pcaplot)


# ==========
# decoding using spkikes
from sklearn import svm
from sklearn import cross_validation
from sklearn.preprocessing import normalize


data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.500], type_filter='spiketrains.*', name_filter='.*Code[1-9]', spike_bin_rate=50)
# data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.600], name_filter='LFPs.*')
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro); pnp.NeuroPlot(data_neuro)

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
            cfl_scores = cross_validation.cross_val_score(clf, X_normalize[indx, t, :], data_df['stim_names'][indx].tolist(), cv=5)
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


# decoding using LFP power
data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.600], name_filter='LFPs.*')
[spcg_f,spcg_t,spcg] = signal.spectrogram(data_neuro['data'], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
spcg_t = np.array(spcg_t) + np.array( data_neuro['ts'][0] )
data_neuro['ts']   = spcg_t
LFP_power_band = 'alpha'
# LFP_power_band = 'beta'
# LFP_power_band = 'gamma'
if LFP_power_band is 'alpha':
    LFP_power_limit = [5,15]
elif LFP_power_band is 'beta':
        LFP_power_limit = [15, 30]
elif LFP_power_band is 'gamma':
    LFP_power_limit = [35, 50]
data_neuro['data'] = np.transpose(np.mean(spcg[:, np.logical_and(spcg_f >= LFP_power_limit[0], spcg_f <= LFP_power_limit[1]), :, :], axis=1), [0, 2, 1])
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro);

# temp, delete broken channels
# data_neuro['data'][:,:,14]=data_neuro['data'][:,:,14]*0
pnp.NeuroPlot(data_neuro)
plt.suptitle(LFP_power_band)

clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=1)
time_tic = time.time()
[N_tr, N_ts, N_sg] = data_neuro['data'].shape
N_cd = len(data_neuro['cdtn'])
clf_score = np.zeros([N_cd, N_ts])
X_normalize = ( data_neuro['data'] - np.mean( data_neuro['data'] , axis=(0,1), keepdims=True ) ) /np.std( data_neuro['data'] , axis=(0,1), keepdims=True )
for i in range(N_cd):
    cdtn = data_neuro['cdtn'][i]
    indx = np.array(data_neuro['cdtn_indx'][cdtn])
    print(cdtn)
    for t in range(N_ts):
        # clf_score[i, t] = np.mean(cross_validation.cross_val_score(clf, normalize(data_neuro['data'][indx, t, :]), data_df['stim_names'][indx].tolist(), cv=5))
        clf_score[i, t] = np.mean(cross_validation.cross_val_score(clf, X_normalize[indx, t, :], data_df['stim_names'][indx].tolist(), cv=5))
print(time.time() - time_tic)

fig = plt.figure(figsize=(16, 9))
fig.canvas.set_window_title('C={}'.format(C))
for i in range(N_cd):
    plt.subplot(2, 3, i + 1)
    plt.fill_between(data_neuro['ts'], clf_score[i, :] - clf_score[i, :] / 5, clf_score[i, :] + clf_score[i, :] / 5,
                     alpha=0.5)
    plt.plot(data_neuro['ts'], clf_score[i, :])
    plt.title(data_neuro['cdtn'][i])
    plt.ylim([0, 1])
plt.show()
fig.canvas.manager.window.raise_()



# ==========
# spectrogram
data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.500], type_filter='ana.*', name_filter='LFPs.*', spike_bin_rate=50)
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro); pnp.NeuroPlot(data_neuro)
[spcg_f,spcg_t,spcg] = signal.spectrogram(data_neuro['data'], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
spcg_t = np.array(spcg_t) + np.array( data_neuro['ts'][0] )
plt.pcolormesh(spcg_t, spcg_f, np.mean(spcg,axis=0)[:,0,:])
plt.ylim(0,100)

spcg_cdtn = []
for i in range(len(data_neuro['cdtn'])):
    spcg_cdtn.append(np.mean(spcg[data_neuro['cdtn_indx'][data_neuro['cdtn'][i]],:,:,:],axis=0))
clim_max = [np.stack(spcg_cdtn,axis=-1)[:,j,:,:].max() for j in range(16)]
for j in range(16):
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


# coherence

from scipy.signal import spectral
# ch_x=10
# ch_y=14
ch_x=4
ch_y=12
[spcg_f,spcg_t,spcg_xy] = spectral._spectral_helper(data_neuro['data'][:,:,ch_x], data_neuro['data'][:,:,ch_y], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
spcg_t = np.array(spcg_t) + np.array( data_neuro['ts'][0] )
data_neuro = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro )

if False:
    data_neuro = signal_align.neuro_sort(data_df, ['stim_names'], np.logical_and(data_df['mask_opacity_int']==0, data_df['stim_familiarized']==0), data_neuro )
    indx_used = data_neuro['cdtn_indx'][temp['cdtn'][0]]
    [spcg_f,spcg_t,spcg_xy] = spectral._spectral_helper(data_neuro['data'][indx_used,:,ch_x], data_neuro['data'][indx_used,:,ch_y], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)


[_,_,spcg_xx] = spectral._spectral_helper(data_neuro['data'][:,:,ch_x], data_neuro['data'][:,:,ch_x], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
[_,_,spcg_yy] = spectral._spectral_helper(data_neuro['data'][:,:,ch_y], data_neuro['data'][:,:,ch_y], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)


# by condition
plt.figure(figsize=(16,9))
spcg_cdtn = []
for i in range(len(data_neuro['cdtn'])):
    cdtn_indx = data_neuro['cdtn_indx'][data_neuro['cdtn'][i]]
    spcg_xy_ave = np.mean(spcg_xy[cdtn_indx,:],axis=0)
    spcg_xx_ave = np.mean(spcg_xx[cdtn_indx, :], axis=0)
    spcg_yy_ave = np.mean(spcg_yy[cdtn_indx, :], axis=0)
    spcg_coh= np.abs(np.abs(spcg_xy_ave)**2/(spcg_xx_ave * spcg_yy_ave))
    spcg_cdtn.append(spcg_coh)
    # clim_max = [np.stack(spcg_cdtn,axis=-1)[:,j,:].max() for j in range(16)]
    plt.subplot(2,3,i+1)
    plt.pcolormesh(center2edge(spcg_t) , center2edge(spcg_f) , spcg_cdtn[i], cmap=plt.get_cmap('viridis'))
    plt.ylim(0,80)
    plt.clim(0, 1)
    # plt.colorbar()
    plt.title(data_neuro['cdtn'][i])

# all channels, all conditions
fig, axes2d = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True, figsize=(16,10))
plt.tight_layout()
fig.subplots_adjust(hspace=0.05, wspace=0.05)
for ch_x in range(0,16,2):
    for ch_y in range(0,16,2):
        [spcg_f, spcg_t, spcg_xy] = spectral._spectral_helper(data_neuro['data'][:, :, ch_x],
                                                              data_neuro['data'][:, :, ch_y], window=signal.hann(128),
                                                              nperseg=128, nfft=256, fs=data_neuro['signal_info'][0][2],
                                                              axis=1, noverlap=96)
        spcg_t = np.array(spcg_t) + np.array(data_neuro['ts'][0])
        spcg_xy_ave = np.mean(spcg_xy, axis=0)
        plt.sca(axes2d[ch_x/2, ch_y/2])
        if ch_x is 0:
            axes2d[ch_x/2, ch_y/2].set_title('ch{}'.format(ch_y+1))
        if ch_y is 0:
            axes2d[ch_x/2, ch_y/2].set_ylabel('ch{}'.format(ch_x+1))

        # plt.subplot2grid([16, 16], [ch_x, ch_y])

        if ch_x == ch_y:
            spcg_xy_ave = np.abs(spcg_xy_ave)
            plt.pcolormesh(center2edge(spcg_t), center2edge(spcg_f), 10*np.log10(spcg_xy_ave), cmap=plt.get_cmap('inferno'))
            # plt.title('ch{}'.format(ch_x + 1))
        else:
            [_, _, spcg_xx] = spectral._spectral_helper(data_neuro['data'][:, :, ch_x], data_neuro['data'][:, :, ch_x],
                                                        window=signal.hann(128), nperseg=128, nfft=256,
                                                        fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
            [_, _, spcg_yy] = spectral._spectral_helper(data_neuro['data'][:, :, ch_y], data_neuro['data'][:, :, ch_y],
                                                        window=signal.hann(128), nperseg=128, nfft=256,
                                                        fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
            spcg_xx_ave = np.mean(spcg_xx, axis=0)
            spcg_yy_ave = np.mean(spcg_yy, axis=0)
            spcg_coh = np.abs(np.abs(spcg_xy_ave) ** 2 / (spcg_xx_ave * spcg_yy_ave))

            plt.pcolormesh( center2edge(spcg_t), center2edge(spcg_f), spcg_coh, cmap=plt.get_cmap('viridis'))
            plt.clim(0, 1)
            # plt.title( 'ch{} vs ch {}'.format(ch_x+1, ch_y+1) )
        plt.ylim(0, 80)



# spike-fild coherence:
data_neuro_LFP=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.500], type_filter='ana.*', name_filter='LFPs.*')
data_neuro_spk=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.500], type_filter='spi.*', name_filter='.*Code[1-9]', spike_bin_rate=blk.segments[0].analogsignals[0].sampling_rate)
spcg_coh_all = []
N_LFP = len(data_neuro_LFP['signal_info'])
N_spk = len(data_neuro_spk['signal_info'])
fig, axes2d = plt.subplots(nrows=N_LFP, ncols=N_spk, sharex=True, sharey=True, figsize=(16,10))
plt.tight_layout()
fig.subplots_adjust(hspace=0.05, wspace=0.05)

for ch_x in range(N_LFP):
    for ch_y in range(N_spk):
        plt.sca(axes2d[ch_x, ch_y])
        if ch_x is 0:
            axes2d[ch_x, ch_y].set_title('ch{}'.format(data_neuro_spk['signal_info'][ch_y][0]), fontsize=8 )
        if ch_y is 0:
            axes2d[ch_x, ch_y].set_ylabel('ch{}'.format(data_neuro_LFP['signal_info'][ch_x][0]) )

        [spcg_f,spcg_t,spcg_xy] = spectral._spectral_helper(data_neuro_LFP['data'][:,:,ch_x], data_neuro_spk['data'][:,:,ch_y], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
        [_,_,spcg_xx] = spectral._spectral_helper(data_neuro_LFP['data'][:,:,ch_x], data_neuro_LFP['data'][:,:,ch_x], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
        [_,_,spcg_yy] = spectral._spectral_helper(data_neuro_spk['data'][:,:,ch_y], data_neuro_spk['data'][:,:,ch_y], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
        spcg_xy_ave = np.mean(spcg_xy, axis=0)
        spcg_xx_ave = np.mean(spcg_xx, axis=0)
        spcg_yy_ave = np.mean(spcg_yy, axis=0)
        spcg_coh = np.abs(np.abs(spcg_xy_ave) ** 2 / (spcg_xx_ave * spcg_yy_ave))
        spcg_coh_all.append(spcg_coh)
        # plt.subplot2grid([len(data_neuro_LFP['signal_info']), len(data_neuro_spk['signal_info'])], [ch_x, ch_y])
        plt.pcolormesh( center2edge(spcg_t) , center2edge(spcg_f), spcg_coh, cmap=plt.get_cmap('viridis'))
        plt.clim(0, 0.1)
        plt.ylim(0,50)
        # plt.colorbar()
        # plt.title('{} {}'.format(data_neuro_LFP['signal_info'][ch_x][0], data_neuro_spk['signal_info'][ch_y][0]), fontsize=8 )
        plt.show()
for axes_row in axes2d:
    for axes_cell in axes_row:
        plt.sca(axes_cell)
        plt.clim(0, 0.01)
        plt.ylim(0, 80)



# ==========  Active task ==========
dir_dg  = '/Users/Summit/Documents/neural_data/2016-0817_Dante_U16'
file_dg = 'd_MTS_bin_081716004.dg'
path_dg = os.path.join(dir_dg, file_dg)

data_df = dg2df.dg2df(path_dg)


dir_tdt_tank  = '/Users/Summit/Documents/neural_data/2016-0817_Dante_U16/U16-160817-135857'
name_tdt_block = 'd_MTS_bin_081716004'
reader = neo.io.TdtIO(dirname=dir_tdt_tank)
seg = reader.read_segment(blockname=name_tdt_block, sortname='PLX')
blk = Block()
blk.segments.append(seg)
# standardize_TDT_blk.create_rcg(blk)

# get timestamps to align data with
# for prf task, we use stim on
id_Obsv = np.array(data_df['obsid'])      # !!! needs to be modified if multiple dg files are read
tos_StimOn = np.array(data_df['stimon'])  # tos: time of offset
ts_ObsvOn = select_obj_by_attr(blk.segments[0].events, attr='name', value='obsv')[0].times
ts_StimOn = ts_ObsvOn[np.array(id_Obsv)] + tos_StimOn * pq.ms


# align
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 1.500], type_filter='spiketrains.*', name_filter='.*Code[1-9]', spike_bin_rate=1000); print(time.time()-t)
# group
data_df['mask_opacity_int'] = np.round(data_df['NoiseOpacity']*100).astype(int)
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.neuro_sort(data_df, ['SampleFamiliarized','mask_opacity_int'], [], data_neuro); print(time.time()-t)
# plot
import PyNeuroPlot as pnp; reload(pnp); t=time.time(); pnp.NeuroPlot(data_neuro, sk_std=0.005); print(time.time()-t)




# ===== spectragram
data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 1.500], type_filter='ana.*', name_filter='LFPs.*', spike_bin_rate=50)
data_neuro=signal_align.neuro_sort(data_df, ['SampleFamiliarized', 'mask_opacity_int'], [], data_neuro); pnp.NeuroPlot(data_neuro)
[spcg_f,spcg_t,spcg] = signal.spectrogram(data_neuro['data'], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
spcg_t = np.array(spcg_t) + np.array( data_neuro['ts'][0] )
plt.pcolormesh(spcg_t, spcg_f, np.mean(spcg,axis=0)[:,0,:])
plt.ylim(0,100)




# all channels, all conditions
fig, axes2d = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True, figsize=(16,10))
plt.tight_layout()
fig.subplots_adjust(hspace=0.05, wspace=0.05)
for ch_x in range(0,16,2):
    for ch_y in range(0,16,2):
        [spcg_f, spcg_t, spcg_xy] = spectral._spectral_helper(data_neuro['data'][:, :, ch_x],
                                                              data_neuro['data'][:, :, ch_y], window=signal.hann(128),
                                                              nperseg=128, nfft=256, fs=data_neuro['signal_info'][0][2],
                                                              axis=1, noverlap=96)
        spcg_t = np.array(spcg_t) + np.array(data_neuro['ts'][0])
        spcg_xy_ave = np.mean(spcg_xy, axis=0)
        plt.sca(axes2d[ch_x/2, ch_y/2])
        if ch_x is 0:
            axes2d[ch_x/2, ch_y/2].set_title('ch{}'.format(ch_y+1))
        if ch_y is 0:
            axes2d[ch_x/2, ch_y/2].set_ylabel('ch{}'.format(ch_x+1))

        # plt.subplot2grid([16, 16], [ch_x, ch_y])

        if ch_x == ch_y:
            spcg_xy_ave = np.abs(spcg_xy_ave)
            plt.pcolormesh(center2edge(spcg_t), center2edge(spcg_f), 10*np.log10(spcg_xy_ave), cmap=plt.get_cmap('inferno'))
            # plt.title('ch{}'.format(ch_x + 1))
        else:
            [_, _, spcg_xx] = spectral._spectral_helper(data_neuro['data'][:, :, ch_x], data_neuro['data'][:, :, ch_x],
                                                        window=signal.hann(128), nperseg=128, nfft=256,
                                                        fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
            [_, _, spcg_yy] = spectral._spectral_helper(data_neuro['data'][:, :, ch_y], data_neuro['data'][:, :, ch_y],
                                                        window=signal.hann(128), nperseg=128, nfft=256,
                                                        fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
            spcg_xx_ave = np.mean(spcg_xx, axis=0)
            spcg_yy_ave = np.mean(spcg_yy, axis=0)
            spcg_coh = np.abs(np.abs(spcg_xy_ave) ** 2 / (spcg_xx_ave * spcg_yy_ave))

            plt.pcolormesh( center2edge(spcg_t), center2edge(spcg_f), spcg_coh, cmap=plt.get_cmap('viridis'))
            plt.clim(0, 1)
            # plt.title( 'ch{} vs ch {}'.format(ch_x+1, ch_y+1) )
        plt.ylim(0, 80)


"""
N_chan = 16
time_aligned = signal_align_to_evt(blk.segments[0].analogsignals[0], ts_StimOn, [-0.100, 0.500])['time_aligned']
ERP = np.zeros([N_chan,len(time_aligned)])
LFP_aligned = np.zeros([len(data_df), len(time_aligned), N_chan ])
for ch in range(0,N_chan):
    LFP_cur = signal_align_to_evt(blk.segments[0].analogsignals[ch], ts_StimOn, [-0.100, 0.500])['signal_aligned']
    LFP_aligned[:,:,ch] = LFP_cur
    ERP[ch,:] = np.mean( LFP_cur, axis=0)

# plot ERP using Matplotlib
import PyNeuroPlot as pnp
pnp.ErpPlot(ERP, time_aligned)

# plot ERP by groups
groupby_column = ['stim_familiarized','mask_opacity',]
indx_grouped = data_df.groupby(groupby_column).indices
LFP_grouped = dict( (key, LFP_aligned[ value, :, : ] ) for key, value in indx_grouped.items() )

ch = 0
fig = plt.figure( figsize=(16,9) )
for ch in range(16):
    # fig.canvas.set_window_title( '{}'.format(ch) )
    i = 1
    for key in sorted([key for key in indx_grouped]) :
        plt.subplot(2,3,i)
        plt.plot(time_aligned, np.mean(LFP_grouped[key][:,:,ch]*10**6, axis=0), linewidth=2)
        plt.show()
        plt.xlim((-0.1,0.5))
        plt.ylim((-300,300))
        plt.title(key)
        i = i+1
    fig.canvas.manager.window.raise_()
    fig.show()


# Spikes

N_chan = len( blk.segments[0].spiketrains )
time_aligned = signal_align_to_evt(blk.segments[0].spiketrains[0], ts_StimOn, [-0.100, 0.500])['time_aligned']
psth = np.zeros([N_chan,len(time_aligned)])
psth_aligned = np.zeros([len(data_df), len(time_aligned), N_chan ])
for ch in range(0,N_chan):
    psth_cur = signal_align_to_evt(blk.segments[0].spiketrains[ch], ts_StimOn, [-0.100, 0.500])['signal_aligned']
    psth_aligned[:,:,ch] = psth_cur
    psth[ch,:] = np.mean( psth_cur, axis=0)

from scipy import signal as sgn

import PyNeuroPlot as pnp
pnp.ErpPlot( sgn.convolve(psth, np.ones((1,100))/100, 'same'),  time_aligned)


groupby_column = ['stim_familiarized','mask_opacity',]
indx_grouped = data_df.groupby(groupby_column).indices
psth_grouped = dict( (key, psth_aligned[ value, :, : ] ) for key, value in indx_grouped.items() )

fig = plt.figure( figsize=(16,9) )
# fig.canvas.set_window_title( '{}'.format(unit_name) )
for ch in range(N_chan):
    unit_name = blk.segments[0].spiketrains[ch].name

    sort_code_cur = int(re.match('Chan\d+ Code(\d+)', unit_name).group(1))

    if sort_code_cur == 0 or sort_code_cur >= 31:
        continue
    i = 1

    smooth_kernel = sgn.gaussian(50,13)/sum(sgn.gaussian(50,13))
    smooth_kernel = np.expand_dims(smooth_kernel, axis=0)
    for key in sorted([key for key in indx_grouped]) :
        plt.subplot(2,3,i)
        plt.plot(time_aligned, np.mean(  sgn.convolve(psth_grouped[key][:,:,ch], smooth_kernel, 'same'), axis=0), linewidth=2 )
        plt.show()
        plt.ylim((0,60))
        plt.title(key)
        i = i+1
    fig.canvas.manager.window.raise_()
    fig.show()


# spikes using the signal_array_align_to_evt
signal_array_aligned = signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.500], type_filter='spike.*',spike_bin_rate=1000)
groupby_column = ['stim_familiarized','mask_opacity']
indx_grouped = data_df.groupby(groupby_column).indices
psth_grouped = dict( (key, signal_array_aligned['data'][ value, :, : ] ) for key, value in indx_grouped.items() )
"""
