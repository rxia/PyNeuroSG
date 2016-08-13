# test alignment of trials
# exec(open("./test_align.py").read())

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import neo
from neo.core import (Block, Segment, RecordingChannelGroup, RecordingChannel, AnalogSignal, Unit)
import standardize_TDT_blk

# path to tdt data
dir_tdt_tank  = '/Volumes/Labfiles/tdt/temp_transfer/Dexter_four-160601-124424'
name_tdt_block = 'x_optomulti_amp_060116002'

# create neo block and load data form tdt
reader = neo.io.TdtIO(dirname=dir_tdt_tank)
seg = reader.read_segment(blockname=name_tdt_block)
blk = Block()
blk.segments.append(seg)
standardize_TDT_blk.create_rcg(blk)

