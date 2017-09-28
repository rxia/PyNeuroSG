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

list_date = [re.match('.*-(\d{6})-\d{6}', tankname).group(1) for tankname in list_name_tanks]

block_type = 'srv_mask'
# block_type = 'matchnot'
tf_cumsum = True

if block_type == 'matchnot':
    t_plot = [-0.200, 0.800]
else:
    t_plot = [-0.200, 0.800]
if tf_cumsum == True:
    t_plot = [0, 0.800]


def compute_decoding(data_neuro, data_df, label_name='stim_sname', electrode_type='GM32', cdtn=(1,70)):
    label = data_df[label_name]
    limit_tr = pna.index_int2bool(data_neuro['cdtn_indx'][cdtn], len(data_df))
    signal_info = data_neuro['signal_info']
    ts = data_neuro['ts']
    if electrode_type == 'GM32':
        limit_ch = signal_info['channel_index'] <=32
    else:
        limit_ch = signal_info['channel_index'] > 32
    list_image_name = sorted(np.unique(data_df['stim_sname'][limit_tr]))
    M = len(list_image_name)
    all_clf_score = np.zeros([M, M, len(ts)])*np.nan
    for indx_img1 in range(len(list_image_name)):
        for indx_img2 in range(indx_img1-1):
            fltr_tr = (data_df['stim_sname']==list_image_name[indx_img1]) | (data_df['stim_sname']==list_image_name[indx_img2])
            clf_score = pna.decode_over_time(data_neuro['data'], label=label, limit_tr=limit_tr & fltr_tr, limit_ch=limit_ch)
            all_clf_score[indx_img1, indx_img2, :] =  clf_score
    mean_clf_score = np.nanmean(all_clf_score, axis=(0,1))
    # plt.plot(ts, np.mean(np.vstack(all_clf_score), axis=0))
    return all_clf_score


def compute_decoding_of_day(tankname):

    date = re.match('.*-(\d{6})-\d{6}', tankname).group(1)

    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*{}.*'.format(block_type), tankname, tf_interactive=False,
                                                               dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank',
                                                               dir_dg='/shared/homes/sguan/neuro_data/stim_dg')

    """ Get StimOn time stamps in neo time frame """
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'

    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    spike_bin_interval =0.050
    data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                                   name_filter='.*Code[1-9]$', spike_bin_rate=1/spike_bin_interval,
                                                   chan_filter=range(1, 48 + 1))

    data_neuro = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro)

    data_neuro['ts'] = data_neuro['ts']+spike_bin_interval/2
    ts = data_neuro['ts']
    signal_info = data_neuro['signal_info']
    cdtn = data_neuro['cdtn']

    if tf_cumsum:
        data_neuro['data'] = np.cumsum(data_neuro['data'], axis=1)
    else:
        data_neuro['data'] = pna.SmoothTrace(data_neuro['data'], ts=ts, sk_std=0.050)

    decoding_img_GM32 = {}
    decoding_img_U16  = {}
    decoding_noi_GM32 = {}
    decoding_noi_U16  = {}
    for cdtn in data_neuro['cdtn']:
        print(cdtn)
        decoding_img_GM32[cdtn] = compute_decoding(data_neuro, data_df, label_name='stim_sname', electrode_type='GM32', cdtn=cdtn)
        decoding_img_U16 [cdtn] = compute_decoding(data_neuro, data_df, label_name='stim_sname', electrode_type='U16' , cdtn=cdtn)
        decoding_noi_GM32[cdtn] = compute_decoding(data_neuro, data_df, label_name='mask_orientation', electrode_type='GM32', cdtn=cdtn)
        decoding_noi_U16 [cdtn] = compute_decoding(data_neuro, data_df, label_name='mask_orientation', electrode_type='U16' , cdtn=cdtn)

    decoding={}
    decoding['img_GM32'] = decoding_img_GM32
    decoding['img_U16']  = decoding_img_U16
    decoding['noi_GM32'] = decoding_noi_GM32
    decoding['noi_U16']  = decoding_noi_U16
    decoding['date']     = date
    decoding['ts']       = ts

    h_fig, h_axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10,7.5))
    h_axes=np.ravel(h_axes)
    for i, cdtn in enumerate(data_neuro['cdtn']):
        plt.axes(h_axes[i])
        plt.plot(ts, np.nanmean(decoding_img_GM32[cdtn], axis=(0, 1)))
        plt.plot(ts, np.nanmean(decoding_noi_GM32[cdtn], axis=(0, 1)))
        plt.title(cdtn)
        plt.gca().xaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.2)))
        plt.gca().xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.1)))
        plt.grid(which='minor')
    plt.ylim([0.4,1.0])
    plt.xlim([t_plot[0], t_plot[1]])
    h_fig.text(0.5, 0.04, 'ts (s)', ha='center')
    h_fig.text(0.04, 0.5, 'decoding accuracy', va='center', rotation='vertical')
    plt.suptitle('binary_decoding_GM32_{}'.format(date))
    plt.savefig('./temp_figs/binary_decoding_GM32_{}.pdf'.format(date))
    plt.savefig('./temp_figs/binary_decoding_GM32_{}.png'.format(date))

    h_fig, h_axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10,7.5))
    h_axes=np.ravel(h_axes)
    for i, cdtn in enumerate(data_neuro['cdtn']):
        plt.axes(h_axes[i])
        plt.plot(ts, np.nanmean(decoding_img_U16[cdtn], axis=(0, 1)))
        plt.plot(ts, np.nanmean(decoding_noi_U16[cdtn], axis=(0, 1)))
        plt.title(cdtn)
        plt.gca().xaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.2)))
        plt.gca().xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.1)))
        plt.grid(which='minor')
    plt.ylim([0.4,1.0])
    plt.xlim([t_plot[0], t_plot[1]])
    h_fig.text(0.5, 0.04, 'ts (s)', ha='center')
    h_fig.text(0.04, 0.5, 'decoding accuracy', va='center', rotation='vertical')
    plt.suptitle('binary_decoding_U16_{}'.format(date))
    plt.savefig('./temp_figs/binary_decoding_U16_{}.pdf'.format(date))
    plt.savefig('./temp_figs/binary_decoding_U16_{}.png'.format(date))

    return decoding


"""" run code for all sesions, store decodign results (very slow) """
list_decoding_result = []
for tankname in list_name_tanks:
    try:
        print('decoding {}'.format(tankname))
        decoding_result = compute_decoding_of_day(tankname)
        list_decoding_result.append(decoding_result)
        plt.close('all')
    except:
        print('can not plot decoding results for {}'.format(tankname))

if tf_cumsum == True:
    signal_cum = '_cum'
else:
    signal_cum = ''
pickle.dump(list_decoding_result,
                    open('/shared/homes/sguan/Coding_Projects/support_data/Decoding_Binary_{}{}'.format(block_type, signal_cum), "wb"))

"""" load data, plot """
list_cdtn = sorted(list_decoding_result[0]['img_GM32'].keys())
ts = np.array([-0.175, -0.125, -0.075, -0.025,  0.025,  0.075,  0.125,  0.175,
        0.225,  0.275,  0.325,  0.375,  0.425,  0.475,  0.525,  0.575,
        0.625,  0.675,  0.725,  0.775])
if tf_cumsum:
    ts =  np.array([0.025,  0.075,  0.125,  0.175,
            0.225,  0.275,  0.325,  0.375,  0.425,  0.475,  0.525,  0.575,
            0.625,  0.675,  0.725,  0.775])
list_decoding_xy =  ['img_GM32', 'noi_GM32', 'img_U16', 'noi_U16']
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


decoding_ave ={}
decoding_ave['img_V4'] = {}
decoding_ave['noi_V4'] = {}
decoding_ave['img_IT'] = {}
decoding_ave['noi_IT'] = {}
decoding_ave['img_STS'] = {}
decoding_ave['noi_STS'] = {}
for i, cdtn in enumerate(list_cdtn):
    decoding_ave['img_V4'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['img_GM32'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result]), axis=0)

    decoding_ave['noi_V4'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['noi_GM32'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result]), axis=0)

    decoding_ave['img_IT'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['img_U16'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result if date_area[decoding_cur['date']]=='IT']), axis=0)

    decoding_ave['noi_IT'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['noi_U16'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result if date_area[decoding_cur['date']]=='IT']), axis=0)

    decoding_ave['img_STS'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['img_U16'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result if date_area[decoding_cur['date']]=='STS']), axis=0)

    decoding_ave['noi_STS'][cdtn] = np.mean(np.vstack(
        [np.nanmean(decoding_cur['noi_U16'][cdtn], axis=(0, 1)) for decoding_cur in list_decoding_result if date_area[decoding_cur['date']]=='STS']), axis=0)

h_fig, h_axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 7.5))
h_axes = np.ravel(h_axes)
for i, cdtn in enumerate(list_cdtn):
    plt.axes(h_axes[i])
    plt.plot(ts, decoding_ave['img_V4'][cdtn])
    plt.plot(ts, decoding_ave['noi_V4'][cdtn])
    plt.title(cdtn)
    plt.gca().xaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.2)))
    plt.gca().xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.1)))
    plt.grid(which='minor')
plt.ylim([0.4, 1.0])
plt.xlim([t_plot[0], t_plot[1]])
h_fig.text(0.5, 0.04, 'ts (s)', ha='center')
h_fig.text(0.04, 0.5, 'decoding accuracy', va='center', rotation='vertical')
plt.suptitle('binary_decoding_ave_V4')
plt.savefig('./temp_figs/binary_decoding_ave_V4.pdf')
plt.savefig('./temp_figs/binary_decoding_ave_V4.png')


h_fig, h_axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 7.5))
h_axes = np.ravel(h_axes)
for i, cdtn in enumerate(list_cdtn):
    plt.axes(h_axes[i])
    plt.plot(ts, decoding_ave['img_IT'][cdtn])
    plt.plot(ts, decoding_ave['noi_IT'][cdtn])
    plt.title(cdtn)
    plt.gca().xaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.2)))
    plt.gca().xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.1)))
    plt.grid(which='minor')
plt.ylim([0.4, 1.0])
plt.xlim([t_plot[0], t_plot[1]])
h_fig.text(0.5, 0.04, 'ts (s)', ha='center')
h_fig.text(0.04, 0.5, 'decoding accuracy', va='center', rotation='vertical')
plt.suptitle('binary_decoding_ave_IT')
plt.savefig('./temp_figs/binary_decoding_ave_IT.pdf')
plt.savefig('./temp_figs/binary_decoding_ave_IT.png')


h_fig, h_axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 7.5))
h_axes = np.ravel(h_axes)
for i, cdtn in enumerate(list_cdtn):
    plt.axes(h_axes[i])
    plt.plot(ts, decoding_ave['img_STS'][cdtn])
    plt.plot(ts, decoding_ave['noi_STS'][cdtn])
    plt.title(cdtn)
    plt.gca().xaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.2)))
    plt.gca().xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(-0.2, 0.81, 0.1)))
    plt.grid(which='minor')
plt.ylim([0.4, 1.0])
plt.xlim([t_plot[0], t_plot[1]])
h_fig.text(0.5, 0.04, 'ts (s)', ha='center')
h_fig.text(0.04, 0.5, 'decoding accuracy', va='center', rotation='vertical')
plt.suptitle('binary_decoding_ave_STS')
plt.savefig('./temp_figs/binary_decoding_ave_STS.pdf')
plt.savefig('./temp_figs/binary_decoding_ave_STS.png')
