""" get spike waveforms """

""" script to load all dataset and get the six conditions, designed to run on Pogo """

import os
import sys
import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import time                 # time code execution
import cPickle as pickle
import warnings
import sklearn
import sklearn.decomposition as decomposition
import sklearn.mixture as mixture

import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import quantities as pq
import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc

import data_load_DLSH       # package specific for DLSH lab data

from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge


from GM32_layout import layout_GM32

""" ========== prepare data and save ========== """

dir_data_save = '/shared/homes/sguan/Coding_Projects/support_data'

""" get tank name """
try:
    dir_tdt_tank='/shared/lab/projects/encounter/data/TDT/'
    list_name_tanks = os.listdir(dir_tdt_tank)
except:
    dir_tdt_tank = '/Volumes/Labfiles/projects/encounter/data/TDT/'
    list_name_tanks = os.listdir(dir_tdt_tank)
keyword_tank = '.*GM32.*U16'
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_name_tanks_0 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is None]
list_name_tanks_1 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is not None]
list_name_tanks = sorted(list_name_tanks_0) + sorted(list_name_tanks_1)

tankname = list_name_tanks[0]

def GetSpkWF(tankname):
    """ get spike waveform info """
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv_mask.*', tankname, tf_interactive=False,
                                                           dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank',
                                                           dir_dg='/shared/homes/sguan/neuro_data/stim_dg',
                                                           tf_verbose=True)
    date = re.match('.*-(\d{6})-\d{6}', tankname).group(1)
    N_samples = 100
    blk = data_load_DLSH.standardize_blk(blk)
    seg = blk.segments[0]
    N_signal = len(seg.spiketrains)

    signal_name = np.zeros(N_signal, dtype='S32')
    channel_index = np.zeros(N_signal)
    sort_code = np.zeros(N_signal)
    num_spk   = np.zeros(N_signal)

    _, _, T = seg.spiketrains[0].waveforms.shape
    wf_ave = np.zeros([N_signal, T])
    wf_sample = np.zeros([N_signal, N_samples, T])


    for i, spiketrain in enumerate(seg.spiketrains):
        N, _, T = spiketrain.waveforms.shape

        signal_name[i] = spiketrain.name
        channel_index[i] = spiketrain.annotations['channel_index']
        sort_code[i]     = spiketrain.annotations['sort_code']
        num_spk[i]       = N

        # average waveforms
        wf_ave_cur = np.mean(spiketrain.waveforms[:,0,:], axis=0)
        wf_ave[i, :] = wf_ave_cur

        # example waveforms
        if N <= N_samples:
            wf_sample_cur = np.zeros([N_samples, T])*np.nan
            wf_sample_cur[:N, :] = spiketrain.waveforms[:, 0, :]
        else:
            wf_sample_cur = spiketrain.waveforms[np.sort(np.random.permutation(N)[:N_samples]), 0, :]
        wf_sample[i, :, :] = wf_sample_cur

    signal_info = zip([date]*N_signal, signal_name.tolist(), channel_index.tolist(), sort_code.tolist(), num_spk.tolist())
    signal_info = np.array(signal_info, dtype=[('date', 'S32'), ('signal_name', 'S32'),
                                                   ('channel_index', int), ('sort_code', int), ('num_spk', int)])

    return signal_info, wf_ave, wf_sample

list_signal_info = []
list_wf_ave = []
list_wf_sample = []
for tankname in list_name_tanks:
    try:
        signal_info_tank, wf_ave_tank, wf_sample_tank = GetSpkWF(tankname)
        list_signal_info.append(signal_info_tank)
        list_wf_ave.append(wf_ave_tank)
        list_wf_sample.append(wf_sample_tank)
    except:
        warnings.warn('tank {} can not be processed'.format(tankname))

signal_info = np.concatenate(list_signal_info)
wf_ave = np.concatenate(list_wf_ave, axis=0)
wf_sample = np.concatenate(list_wf_sample, axis=0)


pickle.dump([signal_info, wf_ave, wf_sample],
            open('{}/spk_waveforms_all'.format(dir_data_save), "wb"))





""" analyze saved spike waveforms """

[signal_info, wf_ave, wf_sample] = pickle.load(open('{}/spk_waveforms_all'.format(dir_data_save)))


# select subset of neurons
bool_keep = (signal_info['channel_index']>32) * (signal_info['sort_code'] >=2)
# bool_keep = (signal_info['channel_index']<=32) * (signal_info['sort_code'] >=2)

signal_info_keep = signal_info[bool_keep]
wf_ave_keep = wf_ave[bool_keep]

# normalized waveforms
wf = wf_ave_keep/np.abs(np.min(wf_ave_keep, axis=1, keepdims=True))

if False:
    plt.figure()
    plt.plot(wf.transpose())

# compute spk duration

def compute_wf_dur(wf):
    """ compute the trough-to-peak durations """
    return np.array([ np.argmax(w[np.argmin(w):]) for w in wf ])

wf_dur = compute_wf_dur(wf)

if False:
    plt.figure()
    plt.hist(wf_dur, bins=np.arange(np.max(wf_dur)+1))
    pnp.add_sub_axes(plt.gca(), loc='bottom')
    plt.scatter(wf_dur+(np.random.rand(*wf_dur.shape)-0.5), (np.random.rand(*wf_dur.shape)-0.5), 20, marker='x')

# cluster based on spike duration
def determine_wf_type(wf_dur):
    """ assign labels to every waveform """
    bgm = mixture.BayesianGaussianMixture(n_components=2)
    bgm.fit(wf_dur[:,None])
    labels = bgm.predict(wf_dur[:,None])
    if np.nanmean(wf_dur[labels==0]) > np.nanmean(wf_dur[labels==1]):
        labels = 1-labels
    return np.array(['BS' if label==1 else 'NS' for label in labels], dtype='S32')

wf_type = determine_wf_type(wf_dur)

# plot
color_BS = 'steelblue'
color_NS = 'coral'
color_code = np.array([color_NS if temp=='NS' else color_BS for temp in wf_type])

plt.figure(figsize=[8,4])
plt.subplot(1,2,1)
plt.hist(wf_dur, bins=np.arange(np.max(wf_dur)+1), color='grey')
plt.ylabel('distribution')
plt.title('spike duration')
pnp.add_sub_axes(plt.gca(), loc='bottom')
plt.scatter(wf_dur+(np.random.rand(*wf_dur.shape)-0.5), (np.random.rand(*wf_dur.shape)-0.5), 20, c=color_code, marker='x')
plt.xlabel('trough-to-peak duration')
plt.gca().set_yticklabels([])
plt.subplot(1,2,2)
plt.title('normalized waveforms')
h_bs = plt.plot(wf[wf_type=='BS',:].transpose(), c=color_BS, alpha=0.5, label='broad spiking, N={}'.format(np.sum(wf_type=='BS')))
h_ns = plt.plot(wf[wf_type=='NS',:].transpose(), c=color_NS, alpha=0.5, label='broad spiking, N={}'.format(np.sum(wf_type=='NS')))
plt.legend(handles=[h_bs[0], h_ns[0]])
plt.xlabel('time')
plt.gca().set_yticklabels([])
plt.suptitle('IT_spike_waveforms')
# plt.savefig('./temp_figs/IT_spk_waveforms_BS_NS.png')
# plt.savefig('./temp_figs/IT_spk_waveforms_BS_NS.pdf')

def aligh_wf_to_trough(wf):
    """ not used """
    N, T = wf.shape
    wf_trough_aligned = np.zeros([N, 3*T])
    for i in range(wf.shape[0]):
        t_trough = np.argmin(wf[i,:])
        wf_trough_aligned[i, T-t_trough:2*T-t_trough] = wf[i,:]
    return wf_trough_aligned



""" save spike type data back to the signal_info table, together with area and laminar depth """
signal_info = pd.DataFrame(signal_info)

# neuron type: 'BS', 'NS', or ''
wf_type_all = np.array(['']*len(signal_info), dtype='S32')
wf_type_all[bool_keep] = wf_type
signal_info['wf_type'] = wf_type_all
# waveform
signal_info['wf'] = [temp for temp in wf_ave]



# brain area

# the brain area of the recording day
date_area = dict()
date_area['161015'] = 'TEd'
date_area['161023'] = 'TEm'
date_area['161026'] = 'TEm'
date_area['161029'] = 'TEd'
date_area['161118'] = 'TEm'
date_area['161121'] = 'TEm'
date_area['161125'] = 'TEm'
date_area['161202'] = 'TEm'
date_area['161206'] = 'TEd'
date_area['161222'] = 'TEm'
date_area['161228'] = 'TEd'
date_area['170103'] = 'TEd'
date_area['170106'] = 'TEm'
date_area['170113'] = 'TEd'
date_area['170117'] = 'TEd'
date_area['170214'] = 'TEd'
date_area['170221'] = 'TEd'

signal_area = np.array([date_area[i] for i in signal_info['date'].tolist()], dtype='S32')
signal_area[signal_info['channel_index']<=32] = 'V4'
signal_info['area'] = signal_area

# the channel index (count from zero) of the granular layer
indx_g_layer = dict()
indx_g_layer['161015'] = 8
indx_g_layer['161023'] = np.nan
indx_g_layer['161026'] = 8
indx_g_layer['161029'] = 9
indx_g_layer['161118'] = 6
indx_g_layer['161121'] = 4
indx_g_layer['161125'] = 5
indx_g_layer['161202'] = 5
indx_g_layer['161206'] = 8
indx_g_layer['161222'] = 6
indx_g_layer['161228'] = 7
indx_g_layer['170103'] = 7
indx_g_layer['170106'] = 2
indx_g_layer['170113'] = 9
indx_g_layer['170117'] = 6
indx_g_layer['170214'] = np.nan
indx_g_layer['170221'] = np.nan

depth_from_g = dict()
for (date, area) in date_area.items():
    if area == 'TEm':
        depth = np.arange(16,0,-1)-1 - indx_g_layer[date]
    elif area == 'TEd':
        depth =   np.arange(16) - indx_g_layer[date]
    depth_from_g[date] = depth

depth_neuron = np.zeros(len(signal_info))*np.nan
for i in range(len(signal_info)):
    if signal_info['channel_index'][i]>32:
        ch = signal_info['channel_index'][i]-33
        depth = depth_from_g[signal_info['date'][i]][ch]
        depth_neuron[i]=depth
signal_info['depth'] = depth_neuron


signal_info.to_csv('{}/spike_wf_info_Dante.csv'.format(dir_data_save))
signal_info.to_pickle('{}/spike_wf_info_Dante.pkl'.format(dir_data_save))