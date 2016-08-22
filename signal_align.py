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
    if pq.Quantity(evt_align_ts).simplified.units == pq.Quantity(1):
        evt_align_ts = pq.Quantity(evt_align_ts, 's')
    if pq.Quantity(window_offset).simplified.units == pq.Quantity(1):
        window_offset = pq.Quantity(window_offset, 's')
    num_evts = len(evt_align_ts)
    time_aligned = []

    if type(signal) == neo.core.analogsignal.AnalogSignal:

        # get the starting indexes of the all the windows
        indx_start = (( evt_align_ts - signal.t_start ) * signal.sampling_rate) .simplified.astype(int)
        # get the offset indexes of start and stop
        indx_window_offset = ( window_offset * signal.sampling_rate) .simplified.astype(int)
        indx_window = np.array(indx_start, ndmin=2).transpose((1, 0)) + \
                        np.arange(indx_window_offset[0], indx_window_offset[1])
        # get timestamps of all frames
        time_aligned = (np.arange(indx_window_offset[0], indx_window_offset[1])/signal.sampling_rate).simplified
        try:
            signal_aligned = signal[indx_window]
        except IndexError:  # if index out of range
            # print warning message
            n_rows_out_range = np.sum(np.any( np.logical_or(indx_window<0, indx_window>signal.size-1), axis=1))
            warnings.warn('function signal_align_to_evt() encounters {} events that are out of range'.format(n_rows_out_range) )
            # clip the index into range [0, n-1], i.e., replace with the beginning/ending frame for out of range frames
            indx_window = np.clip(indx_window, 0, signal.size-1)
            signal_aligned = signal[indx_window]
        # by avoiding loops, we gained a significant speed up

    elif type(signal) == neo.core.spiketrain.SpikeTrain:
        time_aligned = []
        signal_aligned = []
        for i in range(num_evts):
            spks_time = signal[(signal>evt_align_ts[i]+window_offset[0]) * (signal<evt_align_ts[i] +window_offset[1])] - evt_align_ts[i]
            signal_aligned.append(spks_time)
    return {'signal_aligned': signal_aligned, 'time_aligned': time_aligned}


# to test
if 0:
    import time
    import signal_align; reload(signal_align); from signal_align import signal_align_to_evt
    test_analog_signal = neo.core.AnalogSignal(np.random.rand(10000), units='mV', sampling_rate = 1000*pq.Hz)
    test_evt_align_ts  = np.random.rand(100)*20 *pq.sec
    test_window_offset = [-0.1,0.2] * pq.sec
    t = time.time(); ch=0; temp =  signal_align_to_evt(test_analog_signal, test_evt_align_ts, test_window_offset); elapsed = time.time() - t; print(elapsed)