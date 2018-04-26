# -*- coding: utf-8 -*-
"""
This is an example for reading files with neo.io, and get psth
Shaobo GUAN, Sheinberg lab, Brown University
2016-0501
"""

import numpy as np
import quantities as pq
import neo
import warnings
import re

def signal_align_to_evt(signal, evt_align_ts, window_offset, spike_bin_rate=1000):
    """
    Function break a single neo signal object according to given timestamps and align them together

    used by funciton signal_array_align_to_evt()
    
    :param signal:         neo signal object, either neo.core.analogsignal.AnalogSignal or neo.core.spiketrain.SpikeTrain
    :param evt_align_ts:   timestamps to align signal with, in sec
    :param window_offset:  [t_start, t_stop], relative timestamps, in sec
    :param spike_bin_rate: bin rate for spikes, default to 1000:
    :return:               a dict {'data': signal_aligned, 'ts': time_aligned, 'sampling_rate': sampling_rate}

                           * data:          2D numpy array, [N_trials * N_ts_in_trial]
                           * ts:            1D numpy array, in ts
                           * sampling_rate: a florat number, in Hz
    """

    # if evt_align_ts or window_offset does not have a unit, assume it is 'sec'
    # if pq.Quantity(evt_align_ts).simplified.units == pq.Quantity(1):
    evt_align_ts = pq.Quantity(evt_align_ts, 's')
    # if pq.Quantity(window_offset).simplified.units == pq.Quantity(1):
    window_offset = pq.Quantity(window_offset, 's')
    num_evts = len(evt_align_ts)
    time_aligned = []

    t_start = signal.t_start
    sampling_rate = signal.sampling_rate

    if type(signal) == neo.core.analogsignal.AnalogSignal:
        # remove units from inputs
        t_start       = np.array(signal.t_start.simplified)
        sampling_rate = np.array(signal.sampling_rate.simplified)
        evt_align_ts  = np.array(evt_align_ts.simplified)
        window_offset = np.array(window_offset.simplified)

        # align using tool function
        result_aligned = align_continuous(signal, t_start, sampling_rate, evt_align_ts, window_offset)

        signal_aligned = pq.Quantity( result_aligned['signal_aligned'], 'V' )

    elif type(signal) == neo.core.spiketrain.SpikeTrain:
        if 0: # old, if return real time stamps, rather than binned counts
            time_aligned = []
            signal_aligned = np.array(num_evts, object)
            signal= np.array(pq.Quantity(signal, 's'))
            evt_align_ts = np.array(evt_align_ts)
            ts_window = np.array(np.expand_dims(evt_align_ts, axis=1) + np.expand_dims(window_offset, axis=0))

            signal_aligned = [ signal[(signal>ts_window[i,0]) * (signal<ts_window[i,1])] - evt_align_ts[i]  for i in range(num_evts)]

        t_start       = np.array(signal.t_start.simplified)
        t_stop        = np.array(signal.t_stop .simplified)
        sampling_rate = np.array( pq.Quantity(spike_bin_rate,  'Hz').simplified)
        evt_align_ts  = np.array(evt_align_ts.simplified)
        window_offset = np.array(window_offset.simplified)
        signal        = np.array(signal.simplified)

        num_bins = (t_stop * sampling_rate).astype('int')
        spk_binned  = np.bincount(np.round((signal-t_start)*sampling_rate).astype('int'), minlength=num_bins)[0:num_bins]
        spk_binned = spk_binned * sampling_rate

        result_aligned = align_continuous(spk_binned, t_start, sampling_rate, evt_align_ts, window_offset)
        signal_aligned = pq.Quantity( result_aligned['signal_aligned'], 'Hz' )

    else:
        print('input for function signal_align_to_evt is not recognizable, signal type: {}'.format(signal))

    time_aligned   = pq.Quantity( result_aligned['time_aligned'], 's' )
    sampling_rate  = pq.Quantity( sampling_rate, 'Hz' )

    signal_aligned = np.array(signal_aligned)
    time_aligned = np.array(time_aligned)
    sampling_rate = np.array(sampling_rate)


    return {'data': signal_aligned, 'ts': time_aligned, 'sampling_rate': sampling_rate}


def signal_array_align_to_evt(segment, evt_align_ts, window_offset, type_filter='.*', name_filter='.*', chan_filter=[], spike_bin_rate=1000):
    """
    Function to break neo signal objects (in a segment) according to given timestamps and align them together

    uses function signal_align_to_evt

    :param segment:        neo segment object, contains analogsignals or spiketrain
    :param evt_align_ts:   timestamps to align signal with, in sec
    :param window_offset:  [t_start, t_stop], relative timestamps, in sec
    :param type_filter:    regular expression, used to select signal types, eg., '.*', spiketrains' or 'analogsignals'
    :param name_filter:    regular expression, used to select signal names, eg., '.*', '.*Code[1-9]$', 'LFPs.*'
    :param chan_filter:    list that contains integers, used to select channels, eg. [], [0], range(1,32+1)
    :param spike_bin_rate: bin rate for spikes, default to 1000
    :return:               a dict {'data': signal_aligned, 'ts': time_aligned, 'sampling_rate': sampling_rate}

                            * data:          3D numpy array, [N_trials * N_ts_in_trial * N_signals]
                            * ts:            1D numpy array, in ts
                            * signal_info:   1D numpy array, N_signals * [('name', 'U32'), ('type', 'U32'),
                            ('sampling_rate', float), ('channel_index', int), ('sort_code', int)]
    """

    signal_name = []
    signal_type  = []
    signal_aligned = []
    signal_sampling_rate = []
    signal_chan = []
    signal_sortcode = []
    data = np.array([], ndmin=3)
    ts   = np.array([], ndmin=1)
    neo_data_object_types = ['spiketrains', 'analogsignals']
    for neo_data_object_type in neo_data_object_types:
        if re.match(type_filter, neo_data_object_type) is not None:
            neo_data_object = getattr(segment, neo_data_object_type)
            if len(chan_filter)==0:
                chan_filter = range( len(neo_data_object)*1000 )
            for i in range(len(neo_data_object)):
                cur_name = neo_data_object[i].name
                if re.match(name_filter, cur_name) is not None:
                    try:
                        cur_chan = neo_data_object[i].annotations['channel_index']
                    except:
                        cur_chan = 0
                    if cur_chan in chan_filter:
                        signal_name.append(cur_name)
                        signal_type.append(neo_data_object_type)
                        cur_signal_aligned = signal_align_to_evt(neo_data_object[i], evt_align_ts, window_offset, spike_bin_rate)
                        signal_aligned.append( cur_signal_aligned['data'] )
                        signal_sampling_rate.append( cur_signal_aligned['sampling_rate'] )
                        try:
                            cur_chan = neo_data_object[i].annotations['channel_index']
                        except:
                            cur_chan = 0
                        try:
                            cur_sortcode = neo_data_object[i].annotations['sort_code']
                        except:
                            cur_sortcode = 0
                        signal_chan.append(cur_chan)
                        signal_sortcode.append(cur_sortcode)
    signal_info = list(zip(signal_name, signal_type, signal_sampling_rate, signal_chan, signal_sortcode))
    """
    signal_info = np.array( signal_info, dtype=[('name', 'S32'),
                                                ('type', 'S32'),
                                                ('sampling_rate', float),
                                                ('channel_index', int),
                                                ('sort_code', int)])
    """
    signal_info = np.array(signal_info, dtype=[('name', 'U32'),
                                               ('type', 'U32'),
                                               ('sampling_rate', 'float'),
                                               ('channel_index', 'int'),
                                               ('sort_code', 'int')])

    if len(signal_aligned) == 0:
        warnings.warn('no signals in the segment match the selection filter for alignment')
        print('WARNING of function signal_array_align_to_evt: no signals in the segment match the selection filter for alignment')
    elif len(np.unique(signal_sampling_rate)) > 1:   # if sampling rate is not unique, throw warning
        warnings.warn('signals are of different sampling rates, can not be combined:')
        print('WARNING of function signal_array_align_to_evt: signals are of different sampling rates, can not be combined:')
        for item in signal_info:
            print(item)
    else:    # if sampling rate is unique
        data = np.array( np.dstack(signal_aligned)   )
        ts   = cur_signal_aligned['ts']

    # spiketrain_names = [segment.spiketrains[i].name  for  ]
    #
    # signal_names = [segment.analogsignals[i].name for i in range(len(segment.analogsignals))]
    # print signal_names
    return {'data': data, 'ts': ts, 'signal_info': signal_info}


def blk_align_to_evt(blk, blk_evt_align_ts, window_offset, type_filter='.*', name_filter='.*', chan_filter=[], spike_bin_rate=1000):
    """
    Function to break neo signal objects (in a block contains multiple segments) according to given timestamps and align them together

    uses function signal_array_align_to_evt

    :param segment:        neo block object, contains multiple (N_seg) segments that contain analogsignals or spiketrain
    :param evt_align_ts:   list of N_seg arrays that contain timestamps to align signal with, in sec
    :param window_offset:  [t_start, t_stop], relative timestamps, in sec
    :param type_filter:    regular expression, used to select signal types, eg., '.*', spiketrains' or 'analogsignals'
    :param name_filter:    regular expression, used to select signal names, eg., '.*', '.*Code[1-9]$', 'LFPs.*'
    :param chan_filter:    list that contains integers, used to select channels, eg. [], [0], range(1,32+1)
    :param spike_bin_rate: bin rate for spikes, default to 1000
    :return:               a dict {'data': signal_aligned, 'ts': time_aligned, 'sampling_rate': sampling_rate}:

                            * data:          3D numpy array, [N_trials * N_ts_in_trial * N_signals]
                            * ts:            1D numpy array, in ts
                            * signal_info:   1D numpy array, N_signals * [('name', 'U32'), ('type', 'U32'),
                            ('sampling_rate', float), ('channel_index', int), ('sort_code', int)]
    """

    data_neuro_list = []
    for i in range(len(blk.segments)):
        segment = blk.segments[i]
        evt_align_ts = blk_evt_align_ts[i]
        data_neuro = signal_array_align_to_evt(segment, evt_align_ts, window_offset, type_filter=type_filter, name_filter=name_filter, chan_filter=chan_filter, spike_bin_rate=spike_bin_rate)
        data_neuro_list.append(data_neuro)
        """ to be worked on """
    data_neuro = data_concatenate(data_neuro_list)
    return data_neuro


def data_concatenate(list_data_neuro):
    """
    Tool function for blk_align_to_evt, make sure they contains the same number of signals

    :param list_data_neuro:  a list of data_neuro
    :return:                 concatenated data_neuro
    """

    data_neuro_all = {}
    for i, data_neuro in enumerate(list_data_neuro):
        if i==0:                                 # if the first block, copy it
            data_neuro_all = data_neuro
        else:                                    # for next incoming blocks
            if len(data_neuro['ts']) == len(data_neuro_all['ts']):   # check if ts length matches, otherwise raise error
                # check if signals match, if not match, fill the missing signal with all zeros
                if not np.array_equal( data_neuro['signal_info'], data_neuro_all['signal_info'] ):
                    for indx_signal_new, signal_new in enumerate(data_neuro['signal_info']):      # if emerging signal
                        if signal_new not in data_neuro_all['signal_info']:
                            data_neuro_all['signal_info'] = np.insert(data_neuro_all['signal_info'], indx_signal_new, signal_new)
                            data_neuro_all['data'] = np.insert(data_neuro_all['data'], indx_signal_new, 0.0, axis=2)
                    for indx_signal_old, signal_old in enumerate(data_neuro_all['signal_info']):  # if mising signal
                        if signal_old not in data_neuro['signal_info']:
                            data_neuro['signal_info'] = np.insert(data_neuro['signal_info'], indx_signal_old, signal_old)
                            data_neuro['data'] = np.insert(data_neuro['data'], indx_signal_old, 0.0, axis=2)
                # concatenate
                data_neuro_all['data'] = np.concatenate((data_neuro_all['data'], data_neuro['data']), axis=0)
            else:
                print('function data_concatenate can not work with data of different "ts" length')
                warnings.warn('function data_concatenate can not work with data of different "ts" length')

    return data_neuro_all


def align_continuous(signal, t_start, sampling_rate, evt_align_ts, window_offset):
    """
    Tool function used by signal_align_to_evt
    """

    # tool function to align continuous signals, all inputs do not have units
    signal = np.squeeze(signal)  # make sure it is 1D array
    indx_align = (( evt_align_ts - t_start ) * sampling_rate) .round().astype(int)

    # get the offset indexes of start and stop
    indx_window_offset = ( window_offset * sampling_rate) .round().astype(int)


    # get indexes of all windows ( N_events * N_samples_per_window )
    indx_window = np.array(indx_align, ndmin=2).transpose((1, 0)) + \
                    np.arange(indx_window_offset[0], indx_window_offset[1])

    # get timestamps of all frames
    time_aligned = (np.arange(indx_window_offset[0], indx_window_offset[1])/sampling_rate)

    # get aligned signals, considering signal out of bound
    try:
        signal_aligned = signal[indx_window]
    except IndexError:  # if index out of range
        # print warning message
        n_rows_out_range = np.sum(np.any( np.logical_or(indx_window<0, indx_window>signal.size-1), axis=1))
        warnings.warn('function signal_align_to_evt() encounters {} events that are out of range'.format(n_rows_out_range) )
        # clip the index into range [0, n-1], i.e., replace with the beginning/ending frame for out of range frames
        indx_window = np.clip(indx_window, 0, signal.size-1)
        signal_aligned = signal[indx_window]

    return {'signal_aligned': signal_aligned, 'time_aligned': time_aligned}


def select_signal(data_neuro, indx=None, name_filter=None, chan_filter=None, sortcode_filter=None):
    """
    Select a subset of channels from data_neuro

    :param data_neuro:      data_neuro, see function signal_array_align_to_evt() for details
    :param indx:            index of channels to select
    :param name_filter:     if indx is None; used to select signal based on data_neuro['signal_info'][i]['name']
    :param chan_filter:     if indx is None; used to select signal based on data_neuro['signal_info'][i]['channel_index']
    :param sortcode_filter: if indx is None; used to select signal based on data_neuro['signal_info'][i]['sort_code']
    :return:                data_neuro, with a subset of signals
    """

    N_signal = data_neuro['data'].shape[2]
    if 'signal_info' in data_neuro.keys():
        N_signal0= len(data_neuro['signal_info'])
        if N_signal != N_signal0:
            raise Exception("data_neuro['data'] and data_neuro['signal_info'] show different number of signals")

    if indx is None:    # if indx is not give, use name_filter, chan_filter and sortcode_filter to select
        indx = np.array([True] * N_signal)

        try:
            if name_filter is not None:
                for i in range(N_signal):
                    cur_name = data_neuro['signal_info'][i]['name']
                    if re.match(name_filter, cur_name) is None:
                        indx[i] = False
        except:
            warnings.warn('name_filter not working properly')
        try:
            if chan_filter is not None:
                for i in range(N_signal):
                    cur_chan = data_neuro['signal_info'][i]['channel_index']
                    if cur_chan not in chan_filter:
                        indx[i] = False
        except:
            warnings.warn('chan_filter not working properly')
        try:
            if sortcode_filter is not None:
                for i in range(N_signal):
                    cur_sortcode = data_neuro['signal_info'][i]['sort_code']
                    if cur_sortcode not in sortcode_filter:
                        indx[i] = False
        except:
            warnings.warn('sortcode_filter not working properly')

    data_neuro_new = dict(data_neuro)
    data_neuro_new['data'] = data_neuro['data'][:, :, indx]
    if 'signal_info' in data_neuro_new.keys():
        data_neuro_new['signal_info'] = data_neuro['signal_info'][indx]
    if 'signal_id' in data_neuro_new.keys():
        data_neuro_new['signal_id'] = data_neuro['signal_id'][indx]

    return data_neuro_new


def neuro_sort(tlbl, grpby=[], fltr=[], neuro={}):
    """
    funciton to sort a table arrording to some columns, returns the index of each condition

    :param tlbl:   trial label, information of every trial, a pandas data frame
    :param grpby:  group by   , list of string that specifies the columns to sort and group
    :param fltr:   filter     , binary array of length N_trials for keeping/discarding the trials
    :param neuro:  neural data, a dict, neuro['data'] is a 3D array ( N_trials * len_window * N_signals )
    :return:       ({'grpby': grpby, 'fltr': fltr, 'cdtn': sorted(cdtn_indx.keys()), 'cdtn_indx': cdtn_indx})
    """

    # process inputs
    if grpby is None:
        grpby = []
    if fltr is None:
        fltr = []

    fltr = np.array(fltr)
    if type(grpby) is str:
        grpby = [grpby]
    if len(fltr) == 0:
        fltr = np.ones(len(tlbl), dtype=bool)

    # use pandas to group
    cdtn_indx = tlbl.groupby(grpby).indices

    # filter the results by the binary mask of fltr
    indx_fltrd = np.where(fltr)[0]

    for cdtn_name in list(cdtn_indx.keys()):
        cdtn_indx[cdtn_name] = np.intersect1d(cdtn_indx[cdtn_name], indx_fltrd)
        if len(cdtn_indx[cdtn_name]) == 0:
            del cdtn_indx[cdtn_name]
    neuro.update({'grpby': grpby, 'fltr': fltr, 'cdtn': sorted(cdtn_indx.keys()), 'cdtn_indx': cdtn_indx})

    return neuro


def data3Dto2D(data3D):
    """
    data3D to data2D, move the last dimension to stack vertically

    :param data3D:  np array: [N_trials * N_ts * N_signals]
    :return:        np array: [(N_trials*N_signals) * N_ts ]
    """
    [N1, N2, N3] = data3D.shape
    return np.vstack(data3D[:,:,i] for i in range(N3))


# ==================== Some test script  ====================
# to test
if False:
    test_analog_signal = neo.core.AnalogSignal(np.random.rand(10000), units='mV', sampling_rate = 1000*pq.Hz)
    test_evt_align_ts  = np.random.rand(100)*20 *pq.sec
    test_window_offset = [-0.1,0.2] * pq.sec
    t = time.time(); ch=0; temp =  signal_align_to_evt(test_analog_signal, test_evt_align_ts, test_window_offset); elapsed = time.time() - t; print(elapsed)

