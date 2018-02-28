# -*- coding: utf-8 -*-
"""
This is a function to standardize the neo objects read from TDT

----- Input object format -----
blk (neo block, corresponding to tdt tank):
    segments (list):
        segments[0]:
            analogsignals (list):
                analogsignals[0]: self is a quantity array of signal
                    attributes: name, channel_index, sampling_rate
                ...
                analogsignals[N]
            spiketrains (list):
                spiketrains[0]:  self is a quantity array of spike times
                    attributes: name, channel_index, times, annotations['channel_index']
        segments[1]
        ...
        segments[N]

"""
import re
import numpy as np
from neo.core import (Block, Segment, ChannelIndex, AnalogSignal, Unit)

# ----------- get_chan_indexes() ----------
def get_chan_indexes(blk):
    # get all channel indexes in the block
    # use spiketrains of segments[0] to determine the indexes of channels
    # returns a list of integers
    try:
        some_spiketrains = blk.segments[0].spiketrains
        chan_indexes =  list(set( [ some_spiketrains[i].annotations['channel_index'] for i in range(len(some_spiketrains)) ] ))
    except:
        try:
            some_analogsignals = blk.segments[0].analogsignals
            chan_indexes = list(set([ some_analogsignals[i].channel_index for i in range(len(some_analogsignals))]))
        except:
            print('function get_chan_index() can not get any channel')
            chan_indexes = []
    return chan_indexes


# get total number from channels
def get_num_chan(chan_indexes):
    num_chan = len(chan_indexes)
    return num_chan


# ----------- select_obj_by_attr ----------
def select_obj_by_attr(list_obj, attr='channel_index', value=[]):
    # select the obj from the list of obj if the attr matches value
    # if value is not provided: returns a list of the attr
    # if value is     provided: returns a list of the obj having such attributes
    n = len(list_obj)
    if value == []:
        return [list_obj[i].__getattribute__(attr) for i in range(n)]
    else:
        return [list_obj[i] for i in range(n) if list_obj[i].__getattribute__(attr)==value ]


# ----------- create_list_recordingchannels() ----------
# create the list for neo.recordingchannel object in neo.blk
def create_rcg(blk):
    chan_indexes = get_chan_indexes(blk)              # get all channel indexes in the block

    blk.recordingchannelgroups = []

    for i in chan_indexes:
        rcg = RecordingChannelGroup(name =str(i), channel_indexes=np.array([i], dtype='i'))  # create neo.recordingchannelgroup object, append to blk
        rcg.channel_indexes = np.concatenate( (rcg.channel_indexes, chan_indexes) )
        blk.recordingchannelgroups.append(rcg)

        # create fake neo.recordingchannels in rcg, for continuous signal like LFP
        chan = RecordingChannel(index=i, name='chan %d')
        # add multi-to-multi reference
        chan.recordingchannelgroups.append(rcg)
        rcg.recordingchannels.append(chan)

        # create fake unit in rcg, for spiking activity
        unit = Unit(name='unsorted', channel_indexes=np.array([i], dtype='i') )
        rcg.units.append(unit)
        for seg in blk.segments:                      # add analog signals from all segments
            chan.analogsignals += select_obj_by_attr(seg.analogsignals, attr='channel_index', value=i)
            for spktrn in seg.spiketrains:
                if spktrn.annotations['channel_index'] == i:
                    unit.spiketrains.append(spktrn)


def create_chan_indx_for_spktrains(blk):
    # add 'channel_index' attribute to spiketrains, based on the annotation info
    chan_indexes = get_chan_indexes(blk)              # get all channel indexes in the block



def convert_name_to_unicode(obj):
    """
    convert the all names to unicode name: i.g. convert b'abcde123' to 'abcde123'

    used for python3, input obj could be nay neo object, like Block, Segments, .etc
    """
    if hasattr(obj, 'name') and isinstance(obj.name, str):    # use regular expression to replace string
        obj.name = re.sub(r"b'([^']*)'", r'\1', obj.name)

    if hasattr(obj, 'children'):
        for subobj in obj.children:
            convert_name_to_unicode(subobj)



# reload(standardize_TDT_blk); from standardize_TDT_blk import *;