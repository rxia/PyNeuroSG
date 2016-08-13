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
import standardize_TDT_blk
from standardize_TDT_blk import select_obj_by_attr
import quantities as pq
from signal_align import signal_align_to_evt


# read dg file
dir_dg  = '/Users/Summit/Documents/neural_data/2016-0810_Dante_U16'
file_dg = 'd_srv_mask_081016005.dg'
path_dg = os.path.join(dir_dg, file_dg)

data_df = dg2df.dg2df(path_dg)

# read tdt file
dir_tdt_tank  = '/Users/Summit/Documents/neural_data/2016-0810_Dante_U16/U16-160810-130702'
name_tdt_block = 'd_srv_mask_081016005'
reader = neo.io.TdtIO(dirname=dir_tdt_tank)
seg = reader.read_segment(blockname=name_tdt_block)
blk = Block()
blk.segments.append(seg)
standardize_TDT_blk.create_rcg(blk)

# get timestamps to align data with
# for prf task, we use stim on
id_Obsv = np.array(data_df['obsid'])      # !!! needs to be modified if multiple dg files are read
tos_StimOn = np.array(data_df['stimon'])  # tos: time of offset
ts_ObsvOn = select_obj_by_attr(blk.segments[0].eventarrays, attr='name', value='obv/')[0].times
ts_StimOn = ts_ObsvOn[np.array(id_Obsv)] + tos_StimOn * pq.ms

signal_align_to_evt(blk.segments[0].spiketrains[0], ts_StimOn, [-100, 500])

N_chan = 16
signal_analog = np.zeros((16,600))
for ch in range(0,16):
    signal_analog[ch,:] = np.mean( signal_align_to_evt(blk.segments[0].analogsignals[ch], ts_StimOn, [-100, 500]), axis=0)


# plot using PyQtGraph

# from pyqtgraph.Qt import QtGui, QtCore
# import numpy as np
# import pyqtgraph as pg
#
# app = QtGui.QApplication([])
#
# win = pg.GraphicsWindow(title="Basic plotting examples")
# win.resize(1000,600)
# win.setWindowTitle('pyqtgraph example: Plotting')
#
# # Enable antialiasing for prettier plots
# pg.setConfigOptions(antialias=True)
#
# plt = win.addPlot(title="Basic array plotting")
#
# plt.plot(np.mean(signal_analog[:,:],axis=0), pen=(255,0,0), name="Red curve")

# plot using bokeh
from bokeh.charts import Line, output_file, show
offset_plot_chan = (signal_analog.max()-signal_analog.min())/5
line_plot = Line(signal_analog - np.arange(16).reshape((16, 1))*offset_plot_chan, title="line", legend="top_left", ylabel='Languages')
# output_file('temp_plot.html')
show(line_plot)