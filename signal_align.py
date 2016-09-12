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
        if 0:   # old method, slow by using "histogram" compared with "bincount"
            sampling_interval = 1/sampling_rate
            ts_bin = np.arange(t_start, t_stop, sampling_interval)
            ts_bin_edge = np.append(ts_bin, ts_bin[-1]+sampling_interval )- sampling_interval/2
            spk_binned  = np.histogram(signal, ts_bin_edge)[0]

        result_aligned = align_continuous(spk_binned, t_start, sampling_rate, evt_align_ts, window_offset)
        signal_aligned = pq.Quantity( result_aligned['signal_aligned'], 'Hz' )

    else:
        print('input for function signal_align_to_evt is not recognizable, signal type: {}'.format(signal))

    time_aligned   = pq.Quantity( result_aligned['time_aligned'], 's' )
    sampling_rate  = pq.Quantity( sampling_rate, 'Hz' )

    return {'data': signal_aligned, 'ts': time_aligned, 'sampling_rate': sampling_rate, 'time_aligned': time_aligned, 'signal_aligned': signal_aligned}


def signal_array_align_to_evt(segment, evt_align_ts, window_offset, type_filter='.*', name_filter='.*', spike_bin_rate=1000):
    """
    function to align signal arrays to event, returns a 3D array ( N_trials * len_window * N_signals )
    inputs:
        segment       :  neo segment object
        evt_align_ts  : timestamps to align signal with, default in sec, e.g. ts_StimOn=[0, 1.5, 1.7, ...]
        window_offset : [start, stop] of window relative to event_align_ts, default in sec, eg., [-0.1, 0.5]
        type_filter   : string of signal types to use, in regular expression, e.g., 'spiketrains' or '.*'
        name_filter   : string of signal names to use, in regular expression, e.g., 'LFPs.*'
        spike_bin_rate: frequency to bin spikes, default is 1000 Hz
    outputs:
        signal_array_align
    """

    signal_name = []
    signal_type  = []
    signal_aligned = []
    signal_sampling_rate = []
    data = np.array([], ndmin=3)
    ts   = np.array([], ndmin=1)
    neo_data_object_types = ['spiketrains', 'analogsignals']
    for neo_data_object_type in neo_data_object_types:
        if re.match(type_filter, neo_data_object_type) is not None:
            neo_data_object = getattr(segment, neo_data_object_type)
            for i in range(len(neo_data_object)):
                cur_name = neo_data_object[i].name
                if re.match(name_filter, cur_name) is not None:
                    signal_name.append(cur_name)
                    signal_type.append(neo_data_object_type)
                    cur_signal_aligned = signal_align_to_evt(neo_data_object[i], evt_align_ts, window_offset, spike_bin_rate)
                    signal_aligned.append( cur_signal_aligned['data'] )
                    signal_sampling_rate.append( cur_signal_aligned['sampling_rate'] )
    signal_info = zip(signal_name, signal_type, signal_sampling_rate)
    signal_info = np.array( signal_info, dtype=[('name', 'S32'), ('type', 'S32'), ('sampling_rate', float)]  )

    if len(signal_aligned) == 0:
        warnings.warn('no signals in the segment match the selection filter for alignment')
        print('WARNING of function signal_array_align_to_evt: no signals in the segment match the selection filter for alignment')
    elif len(np.unique(signal_sampling_rate)) > 1:   # if sampling rate is not unique, throw warning
        warnings.warn('signals are of different sampling rates, can not be combined:')
        print('WARNING of function signal_array_align_to_evt: signals are of different sampling rates, can not be combined:')
        for item in signal_info:
            print(item)
    else:    # if sampling rate is unique
        data = pq.Quantity( np.dstack(signal_aligned)   )
        ts   = cur_signal_aligned['ts']

    # spiketrain_names = [segment.spiketrains[i].name  for  ]
    #
    # signal_names = [segment.analogsignals[i].name for i in range(len(segment.analogsignals))]
    # print signal_names
    return {'data': data, 'ts': ts, 'signal_info': signal_info}


def align_continuous(signal, t_start, sampling_rate, evt_align_ts, window_offset):
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



def neuro_sort(tlbl, grpby=[], fltr=[], neuro={}, tf_plt=False):
    """
    :param neuro:  neural data, a dict, neuro['data'] is a 3D array ( N_trials * len_window * N_signals )
    :param tlbl:   trial label, information of every trial, a pandas data frame
    :param grpby:  group by   , list of string that specifies the columns to sort and group
    :param fltr:   filter     , binary array of length N_trials for keeping/discarding the trials
    :return:
    """

    # process inputs
    if type(grpby) is str:
        grpby = [grpby]
    if len(fltr) == 0:
        fltr = np.ones(len(tlbl), dtype=bool)

    # use pandas to group
    cdtn_indx = tlbl.groupby(grpby).indices

    # filter the results by the binary mask of fltr
    indx_fltrd = np.where(fltr)[0]

    for cdtn_name in cdtn_indx.keys():
        cdtn_indx[cdtn_name] = np.intersect1d(cdtn_indx[cdtn_name], indx_fltrd)
        if len(cdtn_indx[cdtn_name]) == 0:
            del cdtn_indx[cdtn_name]
    neuro.update({'grpby': grpby, 'fltr': fltr, 'cdtn': sorted(cdtn_indx.keys()), 'cdtn_indx': cdtn_indx})

    return neuro


# ==================== Some test script  ====================
# to test
if 0:
    import time
    import signal_align; reload(signal_align); from signal_align import signal_align_to_evt
    test_analog_signal = neo.core.AnalogSignal(np.random.rand(10000), units='mV', sampling_rate = 1000*pq.Hz)
    test_evt_align_ts  = np.random.rand(100)*20 *pq.sec
    test_window_offset = [-0.1,0.2] * pq.sec
    t = time.time(); ch=0; temp =  signal_align_to_evt(test_analog_signal, test_evt_align_ts, test_window_offset); elapsed = time.time() - t; print(elapsed)

# to test signal_array_align_to_evt
if 0:
    import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.signal_array_align_to_evt(blk.segments[0], ts_StimOn, [-0.100, 0.500], type_filter='spiketrains.*',spike_bin_rate=1000); print(time.time()-t)

# to test neuro_sort
if 0:
    import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity'], [], data_neuro); print(time.time()-t)