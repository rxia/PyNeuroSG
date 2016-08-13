# -*- coding: utf-8 -*-
"""
This is an example for reading files with neo.io, and get psth
Shaobo GUAN, Sheinberg lab, Brown University
2016-0501
"""

import numpy as np
import quantities as pq
import neo

def signal_align_to_evt(signal, evt_align_time, window_size):
    num_evts = len(evt_align_time)
    if type(signal) == neo.core.analogsignal.AnalogSignal:
        signal_aligned = np.zeros([num_evts, window_size[1]-window_size[0]])
        for i in range(num_evts):
            indx_start = np.argwhere(signal.times>evt_align_time[i]).flatten()[0]
            signal_aligned[i,:] = signal[indx_start+window_size[0]:indx_start+window_size[1]]
    elif type(signal) == neo.core.spiketrain.SpikeTrain:
        signal_aligned = []
        for i in range(num_evts):
            spks_time = signal[(signal>evt_align_time[i]-0.1*pq.s) * (signal<evt_align_time[i]+0.5*pq.s)] - evt_align_time[i]
            signal_aligned.append(spks_time)
    return signal_aligned
