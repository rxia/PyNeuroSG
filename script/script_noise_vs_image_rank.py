""" 2017-0509 script to comapare the response between 1) preferred image, but noisy, 2) non-preferred image, but not noisy """

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


import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc

import data_load_DLSH       # package specific for DLSH lab data



""" load data """
try:
    dir_tdt_tank='/shared/lab/projects/encounter/data/TDT/'
    list_name_tanks = os.listdir(dir_tdt_tank)
except:
    dir_tdt_tank = '/Volumes/Labfiles/projects/encounter/data/TDT/'
    list_name_tanks = os.listdir(dir_tdt_tank)
keyword_tank = '.*GM32.*U16'
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_name_tanks_0 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is None]
list_name_tanks_1 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is not None]
list_name_tanks = sorted(list_name_tanks_0) + sorted(list_name_tanks_1)

block_type = 'srv_mask'
# block_type = 'matchnot'
if block_type == 'matchnot':
    t_plot = [-0.200, 1.100]
else:
    t_plot = [-0.100, 0.500]


def GetGroupAve(tankname, signal_type='spk'):
    """ trials are sorted by ['stim_familiarized', 'stim_sname', 'mask_opacity_int'] """
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
    data_neuro = signal_align.neuro_sort(data_df, ['stim_familiarized', 'stim_sname', 'mask_opacity_int'], [], data_neuro)

    ts = data_neuro['ts']
    signal_info = data_neuro['signal_info']
    cdtn = data_neuro['cdtn']
    data_groupave = pna.GroupAve(data_neuro)

    return [data_groupave, ts, signal_info, cdtn]



""" store spk or lfp of all sessions """
signal_type='spk'
# signal_type='lfp'

list_data_groupave = []
list_data_neuro = []
list_data_df = []
list_ts = []
list_cdtn = []
list_signal_info = []
list_date = []


for tankname in list_name_tanks:
    try:
        [data_groupave, ts, signal_info, cdtn] = GetGroupAve(tankname, signal_type=signal_type)
        list_data_groupave.append(data_groupave)
        list_ts.append(ts)
        list_signal_info.append(signal_info)
        list_cdtn.append(cdtn)
        list_date.append(re.match('.*-(\d{6})-\d{6}', tankname).group(1))
        pickle.dump([list_data_groupave, list_ts, list_signal_info, list_cdtn, list_date],
                    open('/shared/homes/sguan/Coding_Projects/support_data/GroupAve_by_name_fam_noise_{}_{}'.format(block_type, signal_type), "wb"))
    except:
        pass
pickle.dump([list_data_groupave, list_ts, list_signal_info, list_cdtn, list_date],
                    open('/shared/homes/sguan/Coding_Projects/support_data/GroupAve_by_name_fam_noise_{}_{}'.format(block_type, signal_type), "wb"))




""" load the stored data """
block_type = 'srv_mask'
# block_type = 'matchnot'
signal_type='spk'
# signal_type='lfp'
[list_data_groupave, list_ts, list_signal_info, list_cdtn, list_date] = pickle.load(open('/shared/homes/sguan/Coding_Projects/support_data/GroupAve_by_name_fam_noise_{}_{}'.format(block_type, signal_type)))

ts = list_ts[0]


"""
process the data, for every channel/neuron, assign the effectiveness of every image according to the rank tuning in  no noise condition
"""

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def rank_stim(data_groupave, cdtn, t_window=[0.05, 0.15]):
    """ takes in the data from one day labeled with (fam, sname, noise), returns the data labeled with (fam, rank, noise) """
    if signal_type == 'spk':
        stat = 'mean'
    elif signal_type == 'lfp':
        stat = 'std'
    data_groupave_ranked = np.zeros(data_groupave.shape)
    cdtn_pd = pd.DataFrame(cdtn, columns=['fam','sname','noise'])
    tf_cdtn__0_0 = np.logical_and(cdtn_pd['fam'] == 0, cdtn_pd['noise'] == 0)   # nov, no noise
    tf_cdtn__1_0 = np.logical_and(cdtn_pd['fam'] == 1, cdtn_pd['noise'] == 0)   # fam, no noise
    for ch in range(data_groupave.shape[2]):     # for every image
        # name of image from most preferred to least preferred
        sname_ranked_nov, _ = pna.TuningCurve(data_groupave[:, :, ch], cdtn_pd['sname'], type='rank', ts=ts,
                                           t_window=t_window, limit=tf_cdtn__0_0, stat=stat)
        sname_ranked_fam, _ = pna.TuningCurve(data_groupave[:, :, ch], cdtn_pd['sname'], type='rank', ts=ts,
                                           t_window=t_window, limit=tf_cdtn__1_0, stat=stat)
        # dict maps sname to rank
        sname_to_rank = dict(zip(sname_ranked_nov, range(len(sname_ranked_nov))) + zip(sname_ranked_fam, range(len(sname_ranked_fam))))
        # replace sname with rank for cdtn
        cdtn_with_rank = [(fam, sname_to_rank[sname], noise) for (fam, sname, noise) in cdtn]
        # sort ty rank
        data_groupave_ranked[:,:,ch] = data_groupave[argsort(cdtn_with_rank), :, ch]
    cdtn_ranked = sorted(cdtn_with_rank)
    return [data_groupave_ranked, cdtn_ranked]

def GetDataCat( list_data_groupave, list_ts, list_signal_info, list_cdtn, list_date):
    # add date to signal_info, list_data_groupave is turned into [N_cdtn, N_ts, N_ch] array
    list_signal_info_date = []
    for signal_info, date in zip(list_signal_info, list_date):
        signal_info_date = pd.DataFrame(signal_info)
        signal_info_date['date'] = date
        list_signal_info_date.append(signal_info_date)
    signal_info_concat = pd.concat(list_signal_info_date, ignore_index=True)
    # re-label the stimulus using rank tuning
    list_data_groupave_ranked = []
    for data_groupave, cdtn in zip(list_data_groupave, list_cdtn):
        data_groupave_ranked, cdtn_ranked = rank_stim(data_groupave, cdtn)
        list_data_groupave_ranked.append(data_groupave_ranked)
    return [np.dstack(list_data_groupave_ranked), list_ts[0], signal_info_concat, cdtn_ranked]

# concatenate and process all data
[data_groupave, ts, signal_info, cdtn] = GetDataCat( list_data_groupave, list_ts, list_signal_info, list_cdtn, list_date )


def PlotByCdtn(data, cdtn, step_fixed=True, step=0, step_window=None, step_window_factor=1):
    """
    plot for this particular data organization
    every row: nov/fam
    every column: noise 0, nose 50, noise 70
    every trace in a subplot: from most preferred image to least preferred stimulus (10 for every subplot)

    To better compare traces across conditions, we shift traces in y axis
    if step_fixed is true: shift trace by a fixed step as indicated in step
    otherwise, shift trace by the firing rate in step_window multiplied by step_window_factor
    """
    if step_window is None:
        step_window = [ts[0], ts[-1]]

    cdtn_pd = pd.DataFrame(cdtn, columns=['fam','rank','noise'])
    h_fig, h_ax = plt.subplots(2,3, sharex=True, sharey=True)
    h_fig.set_size_inches([12,12], forward=True )
    for i, i_fam in enumerate(np.unique(cdtn_pd['fam'])):          # row indicates nov/fam
        for j, j_noise in enumerate(np.unique(cdtn_pd['noise'])):  # col indicates noise level
            plt.axes(h_ax[i,j])
            plt.title((i_fam, j_noise))
            # average over all neurons in this condition, and smooth trace
            psth_cur = pna.SmoothTrace(data[np.logical_and(cdtn_pd['fam']==i_fam, cdtn_pd['noise']==j_noise) ,:], ts=ts, sk_std=0.01)
            N_stim = psth_cur.shape[0]
            if step_fixed:
                if step==0:    # do not shift traces
                    step_add = np.zeros(N_stim)
                else:          # shift every trace by a fixed level, (most preferred image on top)
                    step_add = np.flip(np.arange(0,step*N_stim, step), 0)
            else:              # shift trace by the mean firing rate in step_window
                if signal_type == 'spk':
                    step_add = np.mean(psth_cur[:, np.logical_and(ts>=step_window[0], ts<step_window[1])], axis=1) * step_window_factor
                elif signal_type == 'lfp':
                    step_add = np.std(psth_cur[:, np.logical_and(ts >= step_window[0], ts < step_window[1])],
                                       axis=1) * step_window_factor
            # shifted traces to plot
            psth_cur_step = psth_cur + np.expand_dims(step_add, axis=1)
            # plot
            plt.plot(ts, psth_cur_step.transpose())
    plt.xticks(np.arange(-0.1,0.6,0.1))

# all neuruons
PlotByCdtn(np.mean(data_groupave, axis=2), cdtn)
PlotByCdtn(np.mean(data_groupave, axis=2), cdtn, step=2)
PlotByCdtn(np.mean(data_groupave, axis=2), cdtn, step_fixed=False, step_window=[0.050,0.200], step_window_factor=5)

""" by three brain areas """

if signal_type == 'lfp':
    data_groupave = data_groupave*10**6
    sk_std = None
    ylabel = 'uV'
    laminar_range_ave = 1
else:
    sk_std = 0.007
    # ylabel = 'spk/sec'
    ylabel = 'firing rate'
    laminar_range_ave = 3

# the brain area of the recording day
date_area = dict()
date_area['161015'] = 'IT'
date_area['161023'] = 'STS'
date_area['161026'] = 'STS'
date_area['161029'] = 'IT'
date_area['161118'] = 'STS'
date_area['161121'] = 'STS'
date_area['161125'] = 'STS'
date_area['161202'] = 'STS'
date_area['161206'] = 'IT'
date_area['161222'] = 'STS'
date_area['161228'] = 'IT'
date_area['170103'] = 'IT'
date_area['170106'] = 'STS'
date_area['170113'] = 'IT'
date_area['170117'] = 'IT'
date_area['170214'] = 'STS'
date_area['170221'] = 'STS'

# the channel index (count from zero) of the granular layer
indx_g_layer = dict()
indx_g_layer['161015'] = 9
indx_g_layer['161023'] = np.nan
indx_g_layer['161026'] = 8
indx_g_layer['161029'] = 8
indx_g_layer['161118'] = 9
indx_g_layer['161121'] = 6
indx_g_layer['161125'] = 7
indx_g_layer['161202'] = 5
indx_g_layer['161206'] = 12
indx_g_layer['161222'] = 7
indx_g_layer['161228'] = 12
indx_g_layer['170103'] = 12
indx_g_layer['170106'] = 4
indx_g_layer['170113'] = 7
indx_g_layer['170117'] = 5
indx_g_layer['170214'] = 8
indx_g_layer['170221'] = 6


depth_from_g = dict()
for (date, area) in date_area.items():
    if area == 'STS':
        depth = np.arange(16) - indx_g_layer[date]
    elif area == 'IT':
        depth = np.arange(16,0,-1)-1 - indx_g_layer[date]
    depth_from_g[date] = depth


signal_info['area'] = [date_area[i] for i in signal_info['date'].tolist()]
signal_info['area'][signal_info['channel_index']<=32] = 'V4'

depth_neuron = np.zeros(len(signal_info))*np.nan
for i in range(len(signal_info)):
    if signal_info['channel_index'][i]>32:
        ch = signal_info['channel_index'][i]-33
        depth = depth_from_g[signal_info['date'][i]][ch]
        depth_neuron[i]=depth
signal_info['depth'] = depth_neuron


overlay_style= 'original'
# overlay_style= 'by mean response'
# overlay_style= 'by mean response zoom'
if signal_type == 'spk':
    step_window_factor = 5
elif signal_type == 'lfp':
    step_window_factor = 20
for brain_area in ['V4', 'IT', 'STS']:
    if overlay_style == 'by mean response' or overlay_style == 'by mean response zoom':
        PlotByCdtn(np.mean(data_groupave[:, :, signal_info['area'] == brain_area], axis=2), cdtn, step_fixed=False,
                   step_window=[0.050, 0.200], step_window_factor=step_window_factor)
    else:
        PlotByCdtn(np.mean(data_groupave[:,:,signal_info['area']==brain_area], axis=2), cdtn)
    if overlay_style == 'by mean response zoom':
        if brain_area=='V4':
            y_lim = [60,100]
        elif brain_area=='IT':
            y_lim = [90, 150]
        elif brain_area=='STS':
            y_lim = [55, 90]
        plt.ylim(y_lim)
    plt.xlim([0,0.5])
    plt.suptitle('PSTH by stim_rank_{}'.format(brain_area))
    plt.savefig('./temp_figs/PSTH_by_stim_rank_{}_{}_{}_{}.pdf'.format(block_type, signal_type, brain_area, overlay_style))
    plt.savefig('./temp_figs/PSTH_by_stim_rank_{}_{}_{}_{}.png'.format(block_type, signal_type, brain_area, overlay_style))
