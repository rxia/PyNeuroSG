
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
plt.ioff()
import standardize_TDT_blk
from standardize_TDT_blk import select_obj_by_attr
import quantities as pq
from signal_align import signal_align_to_evt
import re
import PyNeuroPlot as pnp
import time
from scipy import signal
from PyNeuroPlot import center2edge
import misc_tools


# read
# dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/GM32-161012-160949'
# name_tdt_blocks = ['d_V4_spot_101216002', 'd_V4_spot_101216003']
# dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/GM32-161014-151720'
# name_tdt_blocks = ['d_srv_mask_101416004']
# dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/GM32_U16-161015-135134'
# name_tdt_blocks = ['d_srv_mask_101516003', 'd_srv_mask_101516004']
# name_tdt_blocks = ['d_V4_spot_101516001', 'd_V4_spot_101516002']
# dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/GM32_U16-161029-150743'
# name_tdt_blocks = ['d_V4_spot_102916002']
# name_tdt_blocks = ['d_srv_mask_102916003','d_srv_mask_102916004']
# name_tdt_blocks = ['d_matchnot_102916005', 'd_matchnot_102916006', 'd_matchnot_102916007']
# dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/GM32_U16-161118-132556'
# # name_tdt_blocks = ['d_V4_spot_111816003']
# name_tdt_blocks = ['d_matchnot_111816004', 'd_matchnot_111816005','d_matchnot_111816006','d_matchnot_111816007','d_matchnot_111816008']
# dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/GM32_U16-161125-144728'
# name_tdt_blocks = ['d_matchnot_112516004', 'd_matchnot_112516005','d_matchnot_112516006','d_matchnot_112516007','d_matchnot_112516008','d_matchnot_112516009','d_matchnot_112516010']
dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/GM32_U16-161206-130828'
name_tdt_blocks = ['d_matchnot_120616004', 'd_matchnot_120616005','d_matchnot_120616006','d_matchnot_120616007','d_matchnot_120616008','d_matchnot_120616009','d_matchnot_120616010']




blk = Block()
reader = neo.io.TdtIO(dirname=dir_tdt_tank)
for name_tdt_block in name_tdt_blocks:
    print('loading block: {}'.format(name_tdt_block))
    seg = reader.read_segment(blockname=name_tdt_block, sortname='PLX')
    blk.segments.append(seg)
print('finish loading tdt blocks')


# read dg file
dir_dg  = '/Volumes/Labfiles/projects/analysis/shaobo/data_dg'
# file_dgs = ['d_V4_spot_101216002.dg', 'd_V4_spot_101216003.dg']
# file_dgs = ['d_srv_mask_101416004.dg']
# file_dgs = ['d_srv_mask_101516003.dg', 'd_srv_mask_101516004.dg']
# file_dgs = ['d_V4_spot_101516001.dg', 'd_V4_spot_101516002.dg']
# file_dgs  = ['d_V4_spot_102916002.dg']
# file_dgs  = ['d_srv_mask_102916003.dg', 'd_srv_mask_102916004.dg']
# file_dgs  = ['d_matchnot_102916005.dg', 'd_matchnot_102916006.dg', 'd_matchnot_102916007.dg']
# file_dgs  = ['d_V4_spot_111816003.dg']
# file_dgs = ['d_matchnot_111816004.dg', 'd_matchnot_111816005.dg','d_matchnot_111816006.dg','d_matchnot_111816007.dg','d_matchnot_111816008.dg']
# file_dgs = ['d_matchnot_112516004.dg', 'd_matchnot_112516005.dg','d_matchnot_112516006.dg','d_matchnot_112516007.dg','d_matchnot_112516008.dg','d_matchnot_112516009.dg','d_matchnot_112516010.dg']
file_dgs = ['d_matchnot_120616004.dg', 'd_matchnot_120616005.dg','d_matchnot_120616006.dg','d_matchnot_120616007.dg','d_matchnot_120616008.dg','d_matchnot_120616009.dg','d_matchnot_120616010.dg']

data_dfs = []
for file_dg in file_dgs:
    print('loading dg: {}'.format(file_dg))
    path_dg = os.path.join(dir_dg, file_dg)
    data_df = dg2df.dg2df(path_dg)
    data_dfs.append(data_df)
print('finish loading dgs')
data_df = pd.concat(data_dfs)
data_df = data_df.reset_index(range(len(data_df)))

filename_common = misc_tools.str_common(name_tdt_blocks)

dir_temp_fig = './temp_figs'


# glance spk waveforms
import PyNeuroPlot as pnp; reload(pnp);
pnp.SpkWfPlot(blk.segments[0])
plt.savefig('{}/{}_spk_waveform.png'.format(dir_temp_fig, filename_common))

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

# if active task:
data_df['stim_names'] = data_df.SampleFilename
data_df['stim_familiarized'] = data_df.SampleFamiliarized
data_df['mask_opacity'] = data_df['MaskOpacity']


data_df['']=['']*len(data_df)    # empty column
data_df['mask_opacity_int'] = np.round(data_df['mask_opacity']*100).astype(int)

# ERP
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 0.600], type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(1,48+1)); print(time.time()-t)
reload(pnp); pnp.ErpPlot(np.mean(data_neuro['data'], axis=0).transpose(), data_neuro['ts'])
plt.savefig('{}/{}_ERP.png'.format(dir_temp_fig, filename_common))


# if rf mapping
# align, RF plot
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.020, 0.200], type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=100); print(time.time()-t)
# import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.0200, 0.0700], type_filter='ana.*', name_filter='LFPs .*$', spike_bin_rate=1000); print(time.time()-t)
# group
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.neuro_sort(data_df, ['stim_pos_x','stim_pos_y'], [], data_neuro); print(time.time()-t)
# plot
import PyNeuroPlot as pnp; reload(pnp); t=time.time(); pnp.RfPlot(data_neuro, indx_sgnl=0, x_scale=0.2); print(time.time()-t)
for i in range(len(data_neuro['signal_info'])):
    pnp.RfPlot(data_neuro, indx_sgnl=i, x_scale=0.2, y_scale=100)
    try:
        plt.savefig('{}/{}_RF_{}.png'.format(dir_temp_fig, filename_common, data_neuro['signal_info'][i]['name']))
    except:
        plt.savefig('./temp_figs/RF_plot_' + misc_tools.get_time_string() + '.png')
    plt.close()
# import PyNeuroPlot as pnp; reload(pnp); t=time.time(); pnp.NeuroPlot(data_neuro, sk_std=0.005,tf_legend=False, tf_seperate_window=True); print(time.time()-t)



# align
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 1.000], type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000, chan_filter=range(1,48+1)); print(time.time()-t)
# import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 1.000], type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(1,48+1), spike_bin_rate=1000); print(time.time()-t)
# group
import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro); print(time.time()-t)
# plot
import PyNeuroPlot as pnp; reload(pnp); t=time.time(); pnp.NeuroPlot(data_neuro, sk_std=0.005,tf_legend=True, tf_seperate_window=False); print(time.time()-t)


# psth plot
data2D = data_neuro['data'][:,:,-3]
ts = data_neuro['ts']
import PyNeuroPlot as pnp; reload(pnp); t=time.time();plt.figure(); pnp.PsthPlot(data2D, ts, data_df['stim_names'],  np.logical_and (data_df['mask_opacity']==0, data_df['stim_familiarized']==0), sk_std=0.005, subpanel='raster' ); print(time.time()-t)

# psth by six conditions
cdtn0_name = 'stim_familiarized'
cdtn1_name = 'mask_opacity_int'
cdtn_l_name = 'stim_names'
N_cdtn0 = len(data_df[cdtn0_name].unique())
N_cdtn1 = len(data_df[cdtn1_name].unique())
for i_neuron in range(len(data_neuro['signal_info'] )):
    data2D = data_neuro['data'][:, :, i_neuron]
    ts = data_neuro['ts']
    [h_fig, h_ax] = plt.subplots( N_cdtn0, N_cdtn1, figsize=[12,9], sharex=True, sharey=True )
    for i_cdtn0, cdtn0 in enumerate(sorted(data_df[cdtn0_name].unique())) :
        for i_cdtn1, cdtn1 in enumerate(sorted(data_df[cdtn1_name].unique())):
            plt.axes(h_ax[i_cdtn0,i_cdtn1])
            pnp.PsthPlot(data2D, ts, data_df[cdtn_l_name],
                         np.logical_and(data_df[cdtn0_name] == cdtn0, data_df[cdtn1_name] == cdtn1),
                         tf_legend=False, sk_std=0.020, subpanel='raster')
            plt.title( [cdtn0, cdtn1] )

    plt.suptitle(data_neuro['signal_info'][i_neuron]['name'])
    try:
        plt.savefig('{}/{} PSTH {}.png'.format(dir_temp_fig, filename_common, data_neuro['signal_info'][i_neuron]['name']))
    except:
        plt.savefig('./temp_figs/' + misc_tools.get_time_string() + '.png' )
    plt.close(h_fig)


# psth by image
cdtn0_name = 'stim_names'
cdtn1_name = ''
cdtn_l_name = 'mask_opacity_int'
N_cdtn0 = len(data_df[cdtn0_name].unique())
N_cdtn1 = len(data_df[cdtn1_name].unique())
for i_neuron in range(len(data_neuro['signal_info'] )):
    data2D = data_neuro['data'][:, :, i_neuron]
    ts = data_neuro['ts']
    n_rows = int(np.ceil(np.sqrt(N_cdtn0)))
    n_cols = int(np.ceil(1.0*N_cdtn0/n_rows))
    [h_fig, h_ax] = plt.subplots( n_rows, n_cols, figsize=[12,9], sharex=True, sharey=True )
    for i_cdtn0, cdtn0 in enumerate(sorted(data_df[cdtn0_name].unique())) :
        for i_cdtn1, cdtn1 in enumerate(sorted(data_df[cdtn1_name].unique())):
            plt.axes(h_ax.flatten()[i_cdtn0])
            pnp.PsthPlot(data2D, ts, data_df[cdtn_l_name],
                         np.logical_and(data_df[cdtn0_name] == cdtn0, data_df[cdtn1_name] == cdtn1),
                         tf_legend=False, sk_std=0.020, subpanel='raster')
            plt.title( [cdtn0, cdtn1] )

    plt.suptitle(data_neuro['signal_info'][i_neuron]['name'])
    try:
        plt.savefig('{}/{} PSTH {}.png'.format(dir_temp_fig, filename_common, data_neuro['signal_info'][i_neuron]['name']))
    except:
        plt.savefig('./temp_figs/' + misc_tools.get_time_string() + '.png' )
    plt.close(h_fig)


# ==========
# decoding using spkikes
from sklearn import svm
from sklearn import cross_validation
from sklearn.preprocessing import normalize


data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.200, 1.000], type_filter='spiketrains.*', name_filter='.*Code[1-9]', chan_filter=range(1,32+1), spike_bin_rate=50)
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro); pnp.NeuroPlot(data_neuro, tf_legend=True)

Y_train = np.array(data_df['stim_names'].tolist())
# Y_train = np.array(data_df['mask_orientation'].tolist())
# Y_train = np.array(data_df['MaskFilename'].tolist())


N_window_smooth = 3
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
        print((cdtn, len(indx)))
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
        plt.fill_between(data_neuro['ts'], clf_score[i,:]-clf_score_std[i,:], clf_score[i,:]+clf_score_std[i,:], alpha=0.5)
        plt.plot(data_neuro['ts'],clf_score[i,:])
        plt.title(data_neuro['cdtn'][i])
        plt.ylim([0,1])
    plt.show()
    fig.canvas.manager.window.raise_()


# ==========
# spectroagram
data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 1.000], type_filter='ana.*', name_filter='LFPs.*', spike_bin_rate=50)
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro); pnp.NeuroPlot(data_neuro)
[spcg_f,spcg_t,spcg] = signal.spectrogram(data_neuro['data'], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
spcg_t = np.array(spcg_t) + np.array( data_neuro['ts'][0] )
time_baseline = [-1.0, 0.05]
tf_baseline = True
spcg = np.log10(spcg)
# plt.pcolormesh(spcg_t, spcg_f, np.mean(spcg,axis=0)[:,0,:])
# plt.ylim(0,100)

N_sgnl = len(data_neuro['signal_info'])
spcg_cdtn = []
for i in range(len(data_neuro['cdtn'])):
    spcg_cdtn.append(np.mean(spcg[data_neuro['cdtn_indx'][data_neuro['cdtn'][i]],:,:,:],axis=0))
clim_max = [np.stack(spcg_cdtn,axis=-1)[:,j,:,:].max() for j in range(N_sgnl)]
plt.ioff()
for j in range(N_sgnl):
    fig = plt.figure(figsize=(16,9))
    fig.canvas.set_window_title('chan_{}'.format(j+1))
    fig.suptitle(data_neuro['grpby'])
    for i in range(len(data_neuro['cdtn'])):
        plt.subplot(2, 3, i + 1)
        if tf_baseline:
            spcg_baseline = np.mean(spcg_cdtn[i][: ,j,np.logical_and(spcg_t>time_baseline[0],spcg_t<time_baseline[1])],axis=1, keepdims=True)
            spcg_plot = spcg_cdtn[i][:,j,:] - spcg_baseline
            plt.pcolormesh(center2edge(spcg_t), center2edge(spcg_f), spcg_plot, cmap=plt.get_cmap('coolwarm'))
            # plt.clim(-clim_max[j], clim_max[j])
        else:
            spcg_plot = spcg_cdtn[i][:, j, :]
            plt.pcolormesh(center2edge(spcg_t), center2edge(spcg_f), spcg_plot, cmap=plt.get_cmap('inferno'))
            plt.clim(0, clim_max[j])


        if False:
            plt.pcolormesh(center2edge(spcg_t), center2edge(spcg_f), 10*np.log10(spcg_cdtn[i][:, j, :]/clim_max[j]), cmap=plt.get_cmap('inferno'))
            # plt.colorbar()
            plt.clim(-30, 0)
        plt.ylim(0, 120)
        plt.title(data_neuro['cdtn'][i])
        # if i==len(data_neuro['cdtn'])-1:
        #     plt.colorbar()
    # plt.get_current_fig_manager().window.raise_()
    # plt.show()
    plt.savefig('{}/{} spectragram {}.png'.format(dir_temp_fig, filename_common, data_neuro['signal_info'][j]['name']))
    plt.close()
plt.ion()


# ==========
# coherence


from scipy.signal import spectral
ch_x=05-1
ch_y=38-1
[spcg_f,spcg_t,spcg_xy] = spectral._spectral_helper(data_neuro['data'][:,:,ch_x], data_neuro['data'][:,:,ch_y], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
[_,_,spcg_xx] = spectral._spectral_helper(data_neuro['data'][:,:,ch_x], data_neuro['data'][:,:,ch_x], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
[_,_,spcg_yy] = spectral._spectral_helper(data_neuro['data'][:,:,ch_y], data_neuro['data'][:,:,ch_y], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)

spcg_t = np.array(spcg_t) + np.array( data_neuro['ts'][0] )
data_neuro = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro )
data_neuro_full = data_neuro

if False:
    data_neuro = signal_align.neuro_sort(data_df, ['stim_names'], np.logical_and(data_df['mask_opacity_int']==0, data_df['stim_familiarized']==0), data_neuro )
    indx_used = data_neuro['cdtn_indx'][temp['cdtn'][0]]
    [spcg_f,spcg_t,spcg_xy] = spectral._spectral_helper(data_neuro['data'][indx_used,:,ch_x], data_neuro['data'][indx_used,:,ch_y], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)

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
    plt.clim(0, 0.5)
    # plt.colorbar()
    plt.title(data_neuro['cdtn'][i])


# coherence for every image:
ch_x=48-1   # signal 0
ch_y= 5-1   # signal 1

data_neuro_0=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 0.600], type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(0,48+1))
data_neuro_1=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 0.600], type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(0,48+1))
# data_neuro_1=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 0.600], type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=data_neuro_LFP['signal_info'][0][2]);
print( data_neuro_0['signal_info'][ch_x]['name'] )
print( data_neuro_1['signal_info'][ch_y]['name'] )

[_, _, spcg_xx] = spectral._spectral_helper(data_neuro_0['data'][:, :, ch_x], data_neuro_0['data'][:, :, ch_x],
                                            window=signal.hann(128), nperseg=128, nfft=256,
                                            fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
[_, _, spcg_yy] = spectral._spectral_helper(data_neuro_1['data'][:, :, ch_y], data_neuro_1['data'][:, :, ch_y],
                                            window=signal.hann(128), nperseg=128, nfft=256,
                                            fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
[_spcg_f,spcg_t, spcg_xy] = spectral._spectral_helper(data_neuro_0['data'][:, :, ch_x], data_neuro_1['data'][:, :, ch_y],
                                            window=signal.hann(128), nperseg=128, nfft=256,
                                            fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
spectrum_plot = 'phase'
# spectrum_plot = 'phase'
for name_img in data_df.stim_names.unique():
    data_neuro = signal_align.neuro_sort(data_df, ['mask_orientation', 'mask_opacity_int'], data_df.stim_names==name_img, data_neuro)

    # by condition
    plt.figure(figsize=(16,9))
    spcg_cdtn = []
    for i in range(len(data_neuro['cdtn'])):
        cdtn_indx = data_neuro['cdtn_indx'][data_neuro['cdtn'][i]]
        spcg_xy_ave = np.mean(spcg_xy[cdtn_indx,:],axis=0)
        spcg_xx_ave = np.mean(spcg_xx[cdtn_indx, :], axis=0)
        spcg_yy_ave = np.mean(spcg_yy[cdtn_indx, :], axis=0)
        spcg_coh= np.abs(np.abs(spcg_xy_ave)**2/(spcg_xx_ave * spcg_yy_ave))
        spcg_phs= np.angle(spcg_xy_ave)

        plt.subplot(2, 3, i + 1)
        if spectrum_plot is 'coherence':
            spcg_cdtn.append(spcg_coh)
            plt.pcolormesh(center2edge(spcg_t), center2edge(spcg_f), spcg_cdtn[i], cmap=plt.get_cmap('viridis'))
            plt.clim(0, 0.8)
        elif spectrum_plot is 'phase':
            spcg_cdtn.append(spcg_phs)
            plt.pcolormesh(center2edge(spcg_t), center2edge(spcg_f), spcg_cdtn[i], cmap=plt.get_cmap('hsv'))
            plt.pcolormesh(center2edge(spcg_t), center2edge(spcg_f), spcg_cdtn[i], cmap=plt.get_cmap('hsv'))
            plt.clim(-np.pi, np.pi)
        # clim_max = [np.stack(spcg_cdtn,axis=-1)[:,j,:].max() for j in range(16)]
        # plt.colorbar()
        plt.ylim(0, 80)
        plt.title(data_neuro['cdtn'][i])
    plt.suptitle(name_img)



# ----------
# all pairs
# all channels, all conditions
fig, axes2d = plt.subplots(nrows=24, ncols=24, sharex=True, sharey=True, figsize=(16,10))
plt.tight_layout()
fig.subplots_adjust(hspace=0.05, wspace=0.05)
chan_gap = 2
for ch_x in range(0,48,chan_gap):
    for ch_y in range(0,48,chan_gap):
        [spcg_f, spcg_t, spcg_xy] = spectral._spectral_helper(data_neuro['data'][:, :, ch_x],
                                                              data_neuro['data'][:, :, ch_y], window=signal.hann(128),
                                                              nperseg=128, nfft=256, fs=data_neuro['signal_info'][0][2],
                                                              axis=1, noverlap=96)
        spcg_t = np.array(spcg_t) + np.array(data_neuro['ts'][0])
        spcg_xy_ave = np.mean(spcg_xy, axis=0)
        plt.sca(axes2d[ch_x/chan_gap, ch_y/chan_gap])
        if ch_x is 0:
            axes2d[ch_x/chan_gap, ch_y/chan_gap].set_title('ch{}'.format(ch_y+1))
        if ch_y is 0:
            axes2d[ch_x/chan_gap, ch_y/chan_gap].set_ylabel('ch{}'.format(ch_x+1))

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
            plt.clim(0, 0.5)
            # plt.title( 'ch{} vs ch {}'.format(ch_x+1, ch_y+1) )
        plt.ylim(0, 80)




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



