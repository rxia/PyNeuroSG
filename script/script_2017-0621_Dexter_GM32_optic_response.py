""" script to visualize the optic response in the GM32 drive """

# ----- standard modules -----
import os
import sys
sys.path.append('/shared/homes/sguan/Coding_Projects/PyNeuroSG')

import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import time                 # time code execution
import datetime
import pickle

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

""" load data """

dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT/'
dir_dg = '/shared/lab/projects/analysis/shaobo/data_dg/'
keyword_tank = 'Dexter-170622.*'
keyword_blk = '.*laser.*'

def load_blk(keyword_blk=keyword_blk, keyword_tank=keyword_tank, dir_tdt_tank=dir_tdt_tank,
             tf_verbose=True, tf_interactive=False, sortname=''):
    [name_tdt_blocks, path_tdt_tank] = \
        data_load_DLSH.get_file_name(keyword_blk, keyword_tank,
                                     tf_interactive=tf_interactive, dir_tdt_tank=dir_tdt_tank, mode='tdt')

    name_datafiles = name_tdt_blocks
    if tf_verbose:
        print('')
        print('the data files to be loaded are: {}'.format(name_datafiles))

    """ ----- load neural data ----- """
    blk = neo.core.Block()  # block object containing multiple segments, each represents data form one file
    reader = neo.io.TdtIO(dirname=path_tdt_tank)  # reader for loading data
    for name_tdt_block in name_tdt_blocks:  # for every data file
        if tf_verbose:
            print('loading TDT block: {}'.format(name_tdt_block))
        seg = reader.read_segment(blockname=name_tdt_block,
                                  sortname=sortname)  # read one TDT file as a segment of block
        blk.segments.append(seg)  # append to blk object
    if tf_verbose:
        print('finish loading tdt blocks')

    blk = data_load_DLSH.standardize_blk(blk)
    return blk

# blk = load_blk()

def get_event_onset_time(blk, event_name):
    ts_evt = []
    for segment in blk.segments:
        for event in segment.events:
            if event.name==event_name:
                ts_evt.append(np.array(event.times))
    return ts_evt

# ts_evt_org = get_event_onset_time(blk, 'la1_')

def get_laser_train_onset_time(ts_blk_org, min_intervel=0.05):
    ts_blk_flt = []
    for ts_seg_org in ts_blk_org:
        tf_keep = np.insert(np.diff(ts_seg_org)>=min_intervel, 0, True)
        ts_blk_flt.append(ts_seg_org[tf_keep])
    return ts_blk_flt

# get_laser_train_onset_time(ts_evt_org)

def get_laser_pulse_time_in_train(ts_blk_org, ts_blk_flt):
    return ts_blk_org[0][np.logical_and (ts_blk_org[0] >= ts_blk_flt[0][0], ts_blk_org[0] < ts_blk_flt[0][1])] - ts_blk_flt[0][0]

def GetEventEvokedResonse(blk, name_evt, signal_type='spk', t_range= (-0.1, 0.8) ):

    ts_evt_org = get_event_onset_time(blk, name_evt)
    ts_evt_flt = get_laser_train_onset_time(ts_evt_org)
    ts_StimOn = ts_evt_flt
    ts_evt_ticks = get_laser_pulse_time_in_train(ts_evt_org, ts_evt_flt)

    if signal_type=='spk':
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_range, type_filter='spiketrains.*',
                                                       name_filter='.*Code[0-9]$', spike_bin_rate=1000,
                                                       chan_filter=range(1, 48 + 1))
    else:
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_range, type_filter='ana.*',
                                                       name_filter='LFPs.*',
                                                       chan_filter=range(1, 48 + 1))

    ts = data_neuro['ts']
    signal_info = data_neuro['signal_info']
    return [data_neuro, ts, signal_info, ts_evt_ticks]


""" load block """
keyword_tank = 'Dexter_GM32-171009'
blk = load_blk(keyword_blk='.*laser.*', keyword_tank='Dexter-170622.*')
blk = load_blk(keyword_blk='.*laser.*', keyword_tank='Dexter-170623.*')
blk = load_blk(keyword_blk='.*opto.*', keyword_tank=keyword_tank)

""" specify signal and event """
# signal_type = 'LFP'  # 'spk' or 'LFP'
signal_type = 'spk'  # 'spk' or 'LFP'
name_evt = 'la4_'
laser_name_ch = {'la1_': 7, 'la2_': 9, 'la3_': 24, 'la4_': 26,}

""" align data to laser train onset, get average response and smooth data """
[data_neuro, ts, signal_info, ts_evt_ticks] = GetEventEvokedResonse(blk, name_evt=name_evt, signal_type=signal_type, t_range=[-0.1,0.8])
ERP = np.mean(data_neuro['data'], axis=0).transpose()
sk_std = 0.01 if signal_type == 'spk' else 0.002
ERP = pna.SmoothTrace(ERP, sk_std=sk_std, ts=ts)

""" plot ERP in GM32 layout """
h_fig, h_axes = pnp.ErpPlot(ERP, ts=ts, array_layout=layout_GM32)

""" add laser onset tick """
[ylim_min, ylim_max] = plt.gca().get_ylim()
evt_tick_offset = (ylim_max+ylim_min)*0.5
evt_tick_length = (ylim_max-ylim_min)*0.9
for ch, ax_loc in layout_GM32.items():
    plt.axes(h_axes[ax_loc])
    evt_tick_color = ['limegreen'] if ch==laser_name_ch[name_evt] else ['peru']
    evt_tick_alpha = 0.9 if ch==laser_name_ch[name_evt] else 0.7
    plt.eventplot(ts_evt_ticks, lineoffsets=evt_tick_offset, linelengths=evt_tick_length, linewidths=0.5,
                  colors=evt_tick_color, alpha=evt_tick_alpha)
plt.ylim(ylim_min, ylim_max)

""" save fig """
plt.savefig('./temp_figs/laser_response_{}_{}_{}.png'.format(keyword_tank, signal_type, name_evt))




""" psth to stim onset """

def GetPSTH(tankname, signal_type='spk'):

    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('x_.*{}.*'.format('grating'), tankname, tf_interactive=False,
                                                               dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg, sortname='')

    """ Get StimOn time stamps in neo time frame """
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon',
                                            neo_name_obson='obv/', neo_name_obsoff='obv\\')

    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'

    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    if signal_type=='spk':
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                                       name_filter='.*Code[0-9]$', spike_bin_rate=1000,
                                                       chan_filter=range(1, 48 + 1))
    else:
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*',
                                                       name_filter='LFPs.*',
                                                       chan_filter=range(1, 48 + 1))
    # data_neuro = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro)
    PSTH = np.mean(data_neuro['data'], axis=0).transpose()
    ts = data_neuro['ts']
    signal_info = data_neuro['signal_info']


    return [data_neuro, ts, signal_info, PSTH]

signal_type = 'LFP'
[data_neuro, ts, signal_info, ERP] = GetPSTH(tankname='Dexter-170623.*', signal_type=signal_type)
sk_std = 0.01 if signal_type == 'spk' else 0.002
ERP = pna.SmoothTrace(ERP, sk_std=sk_std, ts=ts)
pnp.ErpPlot(ERP, ts=ts, array_layout=layout_GM32)
plt.savefig('./temp_figs/laser_response_{}_{}_{}.png'.format(keyword_tank, signal_type, name_evt))
