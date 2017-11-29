""" script to analyze conherence and phase slope index between brain areas to study feed-forward/feedback information flow  """


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
import mne


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

sys.path.append('./script')
import data_description

try:
    dir_tdt_tank='/shared/lab/projects/encounter/data/TDT/'
    list_name_tanks = os.listdir(dir_tdt_tank)
except:
    dir_tdt_tank = '/Volumes/Labfiles/projects/encounter/data/TDT/'
    list_name_tanks = os.listdir(dir_tdt_tank)

""" ===== ===== process and store data for all sessions, between V4 and IT ===== ===== """


keyword_tank = '.*GM32.*U16'
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_name_tanks_0 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is None]
list_name_tanks_1 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is not None]
list_name_tanks = sorted(list_name_tanks_0) + sorted(list_name_tanks_1)

block_type = 'srv_mask'
# block_type = 'matchnot'
if block_type == 'matchnot':
    t_plot = [-0.200, 1.200]
else:
    t_plot = [-0.200, 0.80]

region_pair = 'in_IT'
if region_pair == 'V4_IT':
    ch_indx_V4 = np.arange(0, 32)
    ch_indx_IT = np.arange(32, 48)
    ch_pair = (np.repeat(ch_indx_V4, len(ch_indx_IT)), np.tile(ch_indx_IT, len(ch_indx_V4))  )
elif region_pair == 'in_IT':
    ch_pair_from = []
    ch_pair_to = []
    ch_list = np.arange(33,48, 2)
    for c_i in ch_list:
        for c_j in ch_list:
            if c_j>c_i:
                ch_pair_from.append(c_i)
                ch_pair_to.append(c_j)
    ch_pair = (np.array(ch_pair_from), np.array(ch_pair_to))


def get_data(tankname, signal_type='lfp'):
    try:
        [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*{}.*'.format(block_type), tankname, tf_interactive=False,
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

    if signal_type=='spk':
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                                       name_filter='.*Code[1-9]$', spike_bin_rate=1000,
                                                       chan_filter=range(1, 48 + 1))
    else:
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*',
                                                       name_filter='LFPs.*',
                                                       chan_filter=range(1, 48 + 1))
    data_neuro = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int', 'stim_sname'], [], data_neuro)
    return data_neuro, data_df

def down_sample_transpose_data(data_neuro, r=5):
    """ prepare data for mne, transpose and dwown-sample """
    data = np.transpose(data_neuro['data'], axes=(0,2,1))
    data_smooth = sp.ndimage.convolve1d(data, np.ones(r))
    data_ds = data_smooth[:,:,::r]
    ts = data_neuro['ts'][::r]
    fs = data_neuro['signal_info'][0]['sampling_rate']/4
    return data_ds, ts, fs

def shuffle_trials_within_cdtn(data_mne, indx_grps):
    N,M,T = data_mne.shape
    data_mne_shfl = np.zeros(data_mne.shape)
    for m in range(M):
        data_mne_shfl[:, m, :] = pna.group_shuffle_data(data_mne[:,m,:], indx_grps=indx_grps)
    return data_mne_shfl

def compute_coh_psi_one_session(name_tank):

    data_neuro, data_df = get_data(name_tank, signal_type='lfp')
    date_str = re.match('.*-(\d{6})-\d{6}', name_tank).group(1)
    data_mne, ts, fs = down_sample_transpose_data(data_neuro, r=5)

    data_mne_shfl_in_cdtn = shuffle_trials_within_cdtn(data_mne, indx_grps=data_df.groupby(['stim_familiarized', 'mask_opacity_int']).indices.values())
    data_mne_shfl_in_imag = shuffle_trials_within_cdtn(data_mne, indx_grps=data_df.groupby(['stim_familiarized', 'mask_opacity_int', 'stim_sname']).indices.values())

    freq = 10**np.arange(1,1.9,0.07)

    cdtn = data_neuro['cdtn']

    def compute_coh_psi(data_mne=data_mne):
        coh_all = np.zeros([len(cdtn), len(ch_pair[0]), len(freq), len(ts)])
        psi_all = np.zeros([len(cdtn), len(ch_pair[0]), len(freq), len(ts)])
        for i, c in enumerate(cdtn):
            indx_trial = data_neuro['cdtn_indx'][c]
            result_coh = mne.connectivity.spectral_connectivity(data_mne[indx_trial,:,:], sfreq=fs, verbose=False, mode='cwt_morlet',
                                                                cwt_n_cycles=5.0,
                                                                cwt_frequencies=freq, fmin=freq*0.8, fmax=freq*1.2,
                                                                indices=ch_pair)

            result_psi = mne.connectivity.phase_slope_index(data_mne[indx_trial,:,:], sfreq=fs, verbose=False, mode='cwt_morlet',
                                                            cwt_n_cycles=5.0,
                                                            cwt_frequencies=freq, fmin=freq*0.8, fmax=freq*1.2,
                                                            indices=ch_pair)
            coh_all[i, :, :, :] = result_coh[0]
            psi_all[i, :, :, :] = result_psi[0]
        return coh_all, psi_all



    coh_all_empr, psi_all_empr = compute_coh_psi(data_mne)
    coh_all_shfl_in_cdtn, psi_all_shfl_in_cdtn = compute_coh_psi(data_mne_shfl_in_cdtn)
    coh_all_shfl_in_imag, psi_all_shfl_in_imag = compute_coh_psi(data_mne_shfl_in_imag)

    r_ds=3
    t_grid = ts[::r_ds]
    def down_sample_result(result, ts=ts, r=r_ds):
        result_smooth = sp.ndimage.convolve1d(result, np.ones(r)/r)
        result_ds = result_smooth[:, :, :, ::r]
        return result_ds
    def ave_over_image(result):
        d1, d2, d3, d4= result.shape
        result_ave = np.zeros([6, d2, d3, d4])
        for i in range(6):
            result_ave[i, :,:,:] = np.mean(result[i*10:(i+1)*10,:,:,:], axis=0)
        return result_ave

    coh_empr = ave_over_image(down_sample_result(coh_all_empr))
    psi_empr = ave_over_image(down_sample_result(psi_all_empr))
    coh_shfl_in_cdtn = ave_over_image(down_sample_result(coh_all_shfl_in_cdtn))
    psi_shfl_in_cdtn = ave_over_image(down_sample_result(psi_all_shfl_in_cdtn))
    coh_shfl_in_imag = ave_over_image(down_sample_result(coh_all_shfl_in_imag))
    psi_shfl_in_imag = ave_over_image(down_sample_result(psi_all_shfl_in_imag))

    return (coh_empr, psi_empr,
            coh_shfl_in_cdtn, psi_shfl_in_cdtn,
            coh_shfl_in_imag, psi_shfl_in_imag,
            cdtn, ch_pair, freq, t_grid, date_str)


coh_psi_all_session = []
for name_tank in list_name_tanks:
    try:
        result_coh_psi = compute_coh_psi_one_session(name_tank)
        coh_psi_all_session.append(result_coh_psi)
    except:
        print('tank {} can not be processed'.format(name_tank))
    pickle.dump(coh_psi_all_session,
                open('../support_data/coh_psi_{}_{}.pickle'.format(region_pair, block_type),
                     "wb"))






""" plot all sessions data, between V4 and IT """
# block_type = 'srv_mask'
block_type = 'matchnot'
if block_type == 'matchnot':
    t_show = [-0.100, 0.6]
else:
    t_show = [-0.100, 0.6]


""" load data """
# coh_psi_all_session = pickle.load(open('../support_data/coh_psi_all_sessions_V4_IT_{}.pickle'.format(block_type)))
# coh_psi_all_session = pickle.load(open('../support_data/coh_psi_all_sessions_in_IT_{}.pickle'.format(block_type)))

list_data_str = [coh_psi_one_session[10] for coh_psi_one_session in coh_psi_all_session]
ch_pair, freq, t_grid = coh_psi_all_session[0][7], coh_psi_all_session[0][8], coh_psi_all_session[0][9]

region_pair_focus = 'IT_sG_iG'
ch_V4_valid = np.array([1,2,3,4,5,6,7,8,11,12,14,15,16,18,19,20,21,26,28,32])-1

list_ch_pair_keep = []
tf_reverse_sign = np.ones(len(list_data_str))   # used for the psi between IT sG and iG case only, to reverse the sign
if region_pair_focus == 'IT_sG':
    list_depth_pair = []
    for date_str in list_data_str:
        ch2depth = data_description.depth_from_g[date_str]
        depth_pair_day = ( ch2depth[ch_pair[0]-32], ch2depth[ch_pair[1]-32] )
        pair_keep_day = (depth_pair_day[0]>2) * (depth_pair_day[1]>2)
        list_depth_pair.append(depth_pair_day)
        list_ch_pair_keep.append(pair_keep_day)
elif region_pair_focus == 'IT_iG':
    list_depth_pair = []
    for date_str in list_data_str:
        ch2depth = data_description.depth_from_g[date_str]
        depth_pair_day = ( ch2depth[ch_pair[0]-32], ch2depth[ch_pair[1]-32] )
        pair_keep_day = (depth_pair_day[0]<-2) * (depth_pair_day[1]<-2)
        list_depth_pair.append(depth_pair_day)
        list_ch_pair_keep.append(pair_keep_day)
elif region_pair_focus == 'IT_sG_iG':
    list_depth_pair = []
    tf_reverse_sign = []
    for date_str in list_data_str:
        ch2depth = data_description.depth_from_g[date_str]
        depth_pair_day = (ch2depth[ch_pair[0] - 32], ch2depth[ch_pair[1] - 32])
        pair_keep_day = (np.abs(depth_pair_day[0]) > 2) * (np.abs(depth_pair_day[1]) > 2) * ( (depth_pair_day[0]*depth_pair_day[1])<0)
        list_depth_pair.append(depth_pair_day)
        list_ch_pair_keep.append(pair_keep_day)
        tf_reverse_sign_cur_1d = (depth_pair_day[0] > depth_pair_day[1]) * 2.0 - 1
        tf_reverse_sign_cur = np.ones([1,len(tf_reverse_sign_cur_1d),1,1])
        tf_reverse_sign_cur[0,:,0,0] = tf_reverse_sign_cur_1d
        tf_reverse_sign.append( tf_reverse_sign_cur )
elif region_pair_focus == 'V4_IT_sG':
    for date_str in list_data_str:
        ch2depth = data_description.depth_from_g[date_str]
        depth_IT_day = ch2depth[ch_pair[1]-32]
        pair_keep_day = np.in1d(ch_pair[0], ch_V4_valid) * (depth_IT_day[1]>2)
        list_ch_pair_keep.append(pair_keep_day)
elif region_pair_focus == 'V4_IT_iG':
    for date_str in list_data_str:
        ch2depth = data_description.depth_from_g[date_str]
        depth_IT_day = ch2depth[ch_pair[1]-32]
        pair_keep_day =  np.in1d(ch_pair[0], ch_V4_valid) * (depth_IT_day[1]<-2)
        list_ch_pair_keep.append(pair_keep_day)
else:
    list_ch_pair_keep = [np.ones(ch_pair[0].shape, dtype=bool) for i in range(len(coh_psi_all_session))]

def cal_ave_data(list_ch_pair_keep=list_ch_pair_keep, i_signal=0, tf_reverse_sign=np.ones(len(list_data_str))):
    return np.nanmean(np.concatenate([(coh_psi_all_session[i_date][i_signal]*tf_reverse_sign[i_date])[:,list_ch_pair_keep[i_date],:,:] for i_date in range(len(list_data_str))], axis=1), axis=1)


coh_ave_empr = cal_ave_data(i_signal=0)
psi_ave_empr = cal_ave_data(i_signal=1, tf_reverse_sign= tf_reverse_sign)
coh_ave_shfl_in_cdtn = cal_ave_data(i_signal=2)
psi_ave_shfl_in_cdtn = cal_ave_data(i_signal=3, tf_reverse_sign= tf_reverse_sign)
coh_ave_shfl_in_imag = cal_ave_data(i_signal=4)
psi_ave_shfl_in_imag = cal_ave_data(i_signal=5, tf_reverse_sign= tf_reverse_sign)



list_plot_type = ['empr','empr-evok', 'empr-stim', 'stim-evok', 'evok']
for plot_type in list_plot_type:
    if plot_type =='empr':
        coh_ave_plot = coh_ave_empr
        psi_ave_plot = psi_ave_empr
    elif plot_type =='empr-evok':
        coh_ave_plot = coh_ave_empr - coh_ave_shfl_in_cdtn
        psi_ave_plot = psi_ave_empr - psi_ave_shfl_in_cdtn
    elif plot_type == 'empr-stim':
        coh_ave_plot = coh_ave_empr - coh_ave_shfl_in_imag
        psi_ave_plot = psi_ave_empr - psi_ave_shfl_in_imag
    elif plot_type == 'stim-evok':
        coh_ave_plot = coh_ave_shfl_in_imag - coh_ave_shfl_in_cdtn
        psi_ave_plot = psi_ave_shfl_in_imag - psi_ave_shfl_in_cdtn
    elif plot_type == 'evok':
        coh_ave_plot = coh_ave_shfl_in_cdtn
        psi_ave_plot = psi_ave_shfl_in_cdtn

    h_fig, h_ax = plt.subplots(2,3, figsize=[8,6])
    h_ax = h_ax.ravel()
    for i in range(6):
        plt.axes(h_ax[i])
        pnp.SpectrogramPlot( coh_ave_plot[i,:,:], t_grid, freq, c_lim_style='diverge', t_lim=t_show)
        if i==0:
            plt.xlabel('t')
            plt.ylabel('freq')
    pnp.share_clim(h_ax)
    h_fig.colorbar(plt.gci(), ax=h_ax.tolist())
    plt.suptitle('coherence {}, {}'.format(region_pair_focus, plot_type))
    plt.savefig('./temp_figs/coherence_{}_all_session_{}_{}.png'.format(region_pair_focus, plot_type, block_type))
    plt.savefig('./temp_figs/coherence_{}_all_session_{}_{}.pdf'.format(region_pair_focus,plot_type, block_type))

    _, h_ax = plt.subplots(2,3, figsize=[8,6])
    h_ax = h_ax.ravel()
    for i in range(6):
        plt.axes(h_ax[i])
        pnp.SpectrogramPlot(psi_ave_plot[i,:,:], t_grid, freq, c_lim_style='diverge', t_lim=t_show)
        if i==0:
            plt.xlabel('t')
            plt.ylabel('freq')
    # plt.xlim([-0.1, 0.4])
    pnp.share_clim(h_ax)
    h_fig.colorbar(plt.gci(), ax=h_ax.tolist())
    plt.suptitle('psi {}, {}'.format(region_pair_focus, plot_type))
    plt.savefig('./temp_figs/psi_{}_all_sessions_{}_{}.png'.format(region_pair_focus, plot_type, block_type))
    plt.savefig('./temp_figs/psi_{}_all_sessions_{}_{}.pdf'.format(region_pair_focus, plot_type, block_type))

    # plt.close('all')






















""" ########## ########## ########## ########## """
""" ########## ########## ########## ########## """
""" ########## ########## ########## ########## """
""" ########## ########## ########## ########## """
""" ########## ########## ########## ########## """
"""  legacy code """


""" temp: resacle results """
name_datafile = '../support_data/coh_psi_all_sessions_V4_IT_matchnot.pickle'
name_datafile_cor = '../support_data/coh_psi_all_sessions_V4_IT_matchnot_corrected.pickle'
coh_psi_all_session = pickle.load(open(name_datafile))
coh_psi_all_session_corrected = []
for coh_psi_one_session in coh_psi_all_session:
    coh_psi_one_session  = list(coh_psi_one_session)
    for i in range(6):
        coh_psi_one_session[i] = coh_psi_one_session[i]/5
    coh_psi_all_session_corrected.append(tuple(coh_psi_one_session))
pickle.dump(coh_psi_all_session_corrected, open(name_datafile_cor, "wb"))





""" test one session data """
