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
from neo.core import (Block, Segment, RecordingChannelGroup, RecordingChannel, AnalogSignal, Unit)
import matplotlib.pyplot as plt
import standardize_TDT_blk
from standardize_TDT_blk import select_obj_by_attr
import quantities as pq
from signal_align import signal_align_to_evt


# read dg file
dir_dg  = '/Users/Summit/Documents/neural_data/2016-0817_Dante_U16'
file_dg = 'd_srv_mask_081716002.dg'
path_dg = os.path.join(dir_dg, file_dg)

data_df = dg2df.dg2df(path_dg)

# read tdt file
dir_tdt_tank  = '/Users/Summit/Documents/neural_data/2016-0817_Dante_U16/U16-160817-135857'
name_tdt_block = 'd_srv_mask_081716002'
reader = neo.io.TdtIO(dirname=dir_tdt_tank)
seg = reader.read_segment(blockname=name_tdt_block)
blk = Block()
blk.segments.append(seg)
standardize_TDT_blk.create_rcg(blk)

# get timestamps to align data with
# for prf task, we use stim on
id_Obsv = np.array(data_df['obsid'])      # !!! needs to be modified if multiple dg files are read
tos_StimOn = np.array(data_df['stimon'])  # tos: time of offset
ts_ObsvOn = select_obj_by_attr(blk.segments[0].eventarrays, attr='name', value='obsv')[0].times
ts_StimOn = ts_ObsvOn[np.array(id_Obsv)] + tos_StimOn * pq.ms
type_StimOn = data_df['mask_opacity']


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
groupby_column = ['mask_opacity','stim_familiarized']
indx_grouped = data_df.groupby(groupby_column).indices
LFP_grouped = dict( (key, LFP_aligned[ value, :, : ] ) for key, value in indx_grouped.items() )

ch = 0
for ch in range(16):
    fig = plt.figure( figsize=(8,6) )
    fig.canvas.set_window_title( '{}'.format(ch) )
    i = 1
    for key in sorted([key for key in indx_grouped]) :
        plt.subplot(3,2,i)
        plt.plot(time_aligned, np.mean(LFP_grouped[key][:,:,ch]*10**6, axis=0) )
        plt.show()
        plt.ylim((-200,200))
        plt.title(key)
        i = i+1
    fig.canvas.manager.window.raise_()
    fig.show()