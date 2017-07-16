# -*- coding: utf-8 -*-
"""
This is an example for reading files with neo.io, and get psth
"""
# exec(open("./test_psth.py").read())

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import neo
from neo.core import (Block, Segment, RecordingChannelGroup, RecordingChannel, AnalogSignal, Unit)
from standardize_TDT_blk import *
from signal_align import *

# data files
data_path = '/Users/Summit/Dropbox/Coding_Projects_Data/python-neo/Dexter_four-160429-151442'
segment_name = 'x_lasergui_042916004'
output_path = '/Users/Summit/Dropbox/Coding_Projects_Data/python-neo/Dexter_four-160429-151442.h5'

#create a reader
reader = neo.io.TdtIO(dirname=data_path)

# read segment and attach to block
seg = reader.read_segment('x_lasergui_042916002')
blk = Block()
blk.segments.append(seg)

# standardize blk
create_rcg(blk)

# creat the event to align the signals
evt_align_on  = select_obj_by_attr(seg.eventarrays, attr='name', value='la0/')[0].times
evt_align_off = select_obj_by_attr(seg.eventarrays, attr='name', value='la0\\')[0].times
evt_align =  evt_align_on[evt_align_off-evt_align_on>0.1]

# from signal_align import signal_align_to_evt
signal_align_analog = signal_align_to_evt(seg.analogsignals[0], evt_align, [-100,500])
plt.figure()
plt.plot( np.mean(signal_align_analog, axis=0) )

signal_align_spktrn = signal_align_to_evt(seg.spiketrains[0], evt_align, [-100,500])
plt.figure()
for i in range(len(signal_align_spktrn)):
    plt.scatter(signal_align_spktrn[i], np.ones(len(signal_align_spktrn[i]))*i )

print ('finish')

