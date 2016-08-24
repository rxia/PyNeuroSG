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


def signal_align_to_evt(signal, evt_align_ts, window_offset):

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
        signal_aligned = result_aligned['signal_aligned']*pq.V
        time_aligned   = result_aligned['time_aligned']  *pq.s

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
        sampling_rate = np.array( (1000*pq.Hz).simplified)
        evt_align_ts  = np.array(evt_align_ts.simplified)
        window_offset = np.array(window_offset.simplified)

        sampling_interval = 1/sampling_rate
        ts_bin = np.arange(t_start, t_stop, sampling_interval)
        ts_bin_edge = np.append(ts_bin, ts_bin[-1]+sampling_interval )- sampling_interval/2
        spk_binned  = np.histogram(signal, ts_bin_edge)[0]

        result_aligned = align_continuous(spk_binned, t_start, sampling_rate, evt_align_ts, window_offset)
        signal_aligned = result_aligned['signal_aligned']*sampling_rate*pq.Hz
        time_aligned   = result_aligned['time_aligned']  *pq.s


    return {'signal_aligned': signal_aligned, 'time_aligned': time_aligned}

def align_continuous(signal, t_start, sampling_rate, evt_align_ts, window_offset):
    # tool function to align continuous signals, all inputs do not have units

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


# to test
if 0:
    import time
    import signal_align; reload(signal_align); from signal_align import signal_align_to_evt
    test_analog_signal = neo.core.AnalogSignal(np.random.rand(10000), units='mV', sampling_rate = 1000*pq.Hz)
    test_evt_align_ts  = np.random.rand(100)*20 *pq.sec
    test_window_offset = [-0.1,0.2] * pq.sec
    t = time.time(); ch=0; temp =  signal_align_to_evt(test_analog_signal, test_evt_align_ts, test_window_offset); elapsed = time.time() - t; print(elapsed)