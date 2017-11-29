
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

""" get data """
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


def GetTuningCurve(tankname, t_window=[0,050, 0.350]):
    try:
        [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv_mask.*', tankname, tf_interactive=False,
                                                               dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank',
                                                               dir_dg='/shared/homes/sguan/neuro_data/stim_dg',
                                                               tf_verbose=True)
    except:
        [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv_mask.*', tankname, tf_interactive=False)

    """ Get StimOn time stamps in neo time frame """
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'

    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    t_plot = [-0.100, 0.500]

    data_neuro_spk = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000, chan_filter=range(1,48+1))
    data_neuro_spk = signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro_spk)

    ts = data_neuro_spk['ts']
    signal_info = data_neuro_spk['signal_info']
    cdtn = data_neuro_spk['cdtn']

    list_rank_tuning_all = []
    for i in range(len(signal_info)):
        list_rank_tuning = []
        for cdtn_cur in data_neuro_spk['cdtn']:
            [_, rank_tuning_cur] = pna.TuningCurve(data_neuro_spk['data'][:, :, i], label=data_df['stim_sname'],
                                     limit=data_neuro_spk['cdtn_indx'][cdtn_cur],
                                     ts=data_neuro_spk['ts'], t_window=t_window)
            list_rank_tuning.append(rank_tuning_cur)
        rank_tuning_neuron = np.vstack(list_rank_tuning)
        list_rank_tuning_all.append(rank_tuning_neuron)
    rank_tuning = np.dstack(list_rank_tuning_all)

    return [rank_tuning, ts, signal_info, cdtn]

list_rank_tuning = []
list_rank_tuning_early = []
list_rank_tuning_late = []
list_ts = []
list_cdtn = []
list_signal_info = []
list_tankname = []

for tankname in list_name_tanks:
    try:
        list_tankname.append(tankname)
        [rank_tuning, ts, signal_info, cdtn] = GetTuningCurve(tankname, t_window=[0.050, 0.350])
        [rank_tuning_early, _, _, _] = GetTuningCurve(tankname, t_window=[0.050, 0.150])
        [rank_tuning_late,  _, _, _] = GetTuningCurve(tankname, t_window=[0.200, 0.350])
        list_rank_tuning.append(rank_tuning)
        list_rank_tuning_early.append(rank_tuning_early)
        list_rank_tuning_late.append(rank_tuning_late)
        list_ts.append(ts)
        list_signal_info.append(signal_info)
        list_cdtn.append(cdtn)
        pickle.dump([list_rank_tuning, list_rank_tuning_early, list_rank_tuning_late, list_ts, list_signal_info, list_cdtn], open('/shared/homes/sguan/Coding_Projects/support_data/RankTuning_srv_mask', "wb"))
    except:
        pass
pickle.dump([list_rank_tuning, list_rank_tuning_early, list_rank_tuning_late, list_ts, list_signal_info, list_cdtn], open('/shared/homes/sguan/Coding_Projects/support_data/RankTuning_srv_mask', "wb"))







""" ========== load saved data ========== """

[list_rank_tuning, list_rank_tuning_early, list_rank_tuning_late, list_ts, list_signal_info, list_cdtn] = pickle.load(open('/shared/homes/sguan/Coding_Projects/support_data/RankTuning_srv_mask'))
list_date = ['161015','161023','161026','161029','161118','161121','161125','161202','161206','161222','161228','170103','170106','170113','170117','170214','170221']

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


def GetDataCat( list_rank_tuning, list_ts, list_signal_info, list_cdtn ):
    list_signal_info_date=[]
    for signal_info, date in zip(list_signal_info, list_date):
        signal_info_date = pd.DataFrame(signal_info)
        signal_info_date['date'] = date
        list_signal_info_date.append(signal_info_date)
    return [np.dstack(list_rank_tuning), list_ts[0], pd.concat(list_signal_info_date, ignore_index=True), list_cdtn[0]]

tuning_time_window = 'full'
if tuning_time_window == 'full':
    [data_RankTuning, ts, signal_info, cdtn] = GetDataCat(list_rank_tuning, list_ts, list_signal_info, list_cdtn)
    tuning_time_window_str = '50-350ms'
elif tuning_time_window == 'early':
    [data_RankTuning, ts, signal_info, cdtn] = GetDataCat(list_rank_tuning_early, list_ts, list_signal_info, list_cdtn)
    tuning_time_window_str = '50-150ms'
elif tuning_time_window == 'late':
    [data_RankTuning, ts, signal_info, cdtn] = GetDataCat(list_rank_tuning_late, list_ts, list_signal_info, list_cdtn)
    tuning_time_window_str = '250-350ms'


signal_info['area'] = [date_area[i] for i in signal_info['date'].tolist()]

colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9), pnp.gen_distinct_colors(3, luminance=0.6)])
linestyles = ['--', '--', '--', '-', '-', '-']
[h_fig, h_ax]=plt.subplots(nrows=1, ncols=3, sharex='all', sharey='all', figsize=[12,5])

plt.axes(h_ax[0])
neuron_keep = signal_info['channel_index'] <= 32
for i in range(data_RankTuning.shape[0]):
    plt.plot(np.mean(data_RankTuning[i, :, neuron_keep], axis=0), color=colors[i],
             linestyle=linestyles[i])
plt.title('V4, N={}'.format(np.sum(neuron_keep)))
plt.xlabel('image rank')
plt.ylabel('spk/sec')

plt.axes(h_ax[1])
neuron_keep = (signal_info['channel_index']>32)*(signal_info['area']=='TEd')
for i in range(data_RankTuning.shape[0]):
    plt.plot( np.mean(data_RankTuning[i,:, neuron_keep ], axis=0), color=colors[i], linestyle=linestyles[i])
plt.title('TEd, N={}'.format(np.sum(neuron_keep)))

plt.axes(h_ax[2])
neuron_keep = (signal_info['channel_index']>32)*(signal_info['area']=='TEm')
for i in range(data_RankTuning.shape[0]):
    plt.plot( np.mean(data_RankTuning[i,:, neuron_keep ], axis=0), color=colors[i], linestyle=linestyles[i])
plt.legend(cdtn)
plt.title('TEm, N={}'.format(np.sum(neuron_keep)))

plt.suptitle('population rank tuning during {}'.format(tuning_time_window_str))
plt.savefig('./temp_figs/RangkTuning_srv_mask_{}.pdf'.format(tuning_time_window))
plt.savefig('./temp_figs/RangkTuning_srv_mask_{}.png'.format(tuning_time_window))

#





""" legacy code """
data_neuro_cur = signal_align.select_signal(data_neuro_spk, chan_filter=range( 0,32), sortcode_filter=range(1,4))
data_neuro_cur = signal_align.select_signal(data_neuro_spk, chan_filter=range(33,48), sortcode_filter=range(1,4))
plt.figure()
for i in range(data_neuro_cur['data'].shape[2]):
    [x, y] = pna.TuningCurve(data_neuro_cur['data'][:, :, i], label=data_df['stim_sname'],
                             limit=data_neuro_cur['cdtn_indx'][data_neuro_cur['cdtn'][3]],
                             ts = data_neuro_cur['ts'], t_window=[0.050, 0.350])
    plt.plot(y)



