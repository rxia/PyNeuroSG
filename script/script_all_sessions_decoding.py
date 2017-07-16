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
from sklearn import svm
from sklearn import cross_validation

from GM32_layout import layout_GM32


""" load data """
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

block_type = 'srv_mask'
# block_type = 'matchnot'
if block_type == 'matchnot':
    t_plot = [-0.200, 1.100]
else:
    t_plot = [-0.100, 0.500]


def GetDecoding(tankname, signal_type='spk'):
    try:
        [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*{}.*'.format(block_type), tankname, tf_interactive=False,
                                                               dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank',
                                                               dir_dg='/shared/homes/sguan/neuro_data/stim_dg')
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

    spike_bin_interval =0.050
    data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                                   name_filter='.*Code[1-9]$', spike_bin_rate=1/spike_bin_interval,
                                                   chan_filter=range(1, 48 + 1))

    data_neuro = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro)

    data_neuro['ts'] = data_neuro['ts']+spike_bin_interval/2
    ts = data_neuro['ts']
    signal_info = data_neuro['signal_info']
    cdtn = data_neuro['cdtn']

    """ decode """
    def decode_activity(data_neuro, target_name='stim_names'):
        X_normalize = (data_neuro['data'] - np.mean(data_neuro['data'], axis=(0, 1), keepdims=True)) / np.std(
            data_neuro['data'], axis=(0, 1), keepdims=True)

        clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=1)
        [N_tr, N_ts, N_sg] = data_neuro['data'].shape
        N_cd = len(data_neuro['cdtn'])
        clf_score = np.zeros([N_cd, N_ts])
        clf_score_std = np.zeros([N_cd, N_ts])
        for i in range(N_cd):
            cdtn = data_neuro['cdtn'][i]
            indx = np.array(data_neuro['cdtn_indx'][cdtn])
            for t in range(N_ts):
                cfl_scores = cross_validation.cross_val_score(clf, X_normalize[indx, t, :],
                                                              data_df[target_name][indx].tolist(), cv=5)
                clf_score[i, t] = np.mean(cfl_scores)
                clf_score_std[i, t] = np.std(cfl_scores)
        return clf_score

    decode_GM32_image = decode_activity(signal_align.select_signal(data_neuro, chan_filter=np.arange(1, 32 + 1)),
                                  'stim_names')
    decode_GM32_noise = decode_activity(signal_align.select_signal(data_neuro, chan_filter=np.arange(1, 32 + 1)),
                                  'mask_orientation')
    decode_U16_image = decode_activity(signal_align.select_signal(data_neuro, chan_filter=np.arange(33, 48 + 1)),
                                  'stim_names')
    decode_U16_noise = decode_activity(signal_align.select_signal(data_neuro, chan_filter=np.arange(33, 48 + 1)),
                                  'mask_orientation')

    decode_result = np.dstack([decode_GM32_image, decode_GM32_noise, decode_U16_image, decode_U16_noise])
    return [decode_result, ts, signal_info, cdtn]



list_decode_result = []
list_ts = []
list_cdtn = []
list_signal_info = []
list_date = []

for tankname in list_name_tanks:
    try:
        [data_groupave, ts, signal_info, cdtn] = GetDecoding(tankname)
        list_decode_result.append(data_groupave)
        list_ts.append(ts)
        list_signal_info.append(signal_info)
        list_cdtn.append(cdtn)
        list_date.append(re.match('.*-(\d{6})-\d{6}', tankname).group(1))
        pickle.dump([list_decode_result, list_ts, list_signal_info, list_cdtn, list_date],
                    open('/shared/homes/sguan/Coding_Projects/support_data/Decode_{}'.format(block_type), "wb"))
    except:
        print('tank {} can not be processed'.format(tankname))
pickle.dump([list_decode_result, list_ts, list_signal_info, list_cdtn, list_date],
                    open('/shared/homes/sguan/Coding_Projects/support_data/Decode_{}'.format(block_type), "wb"))





[list_decode_result, list_ts, list_signal_info, list_cdtn, list_date] = pickle.load(open('/shared/homes/sguan/Coding_Projects/support_data/Decode_{}'.format(block_type)))
decode_result = np.stack(list_decode_result, axis=-1)


date_area = dict()
date_area['161015'] = 'IT'
date_area['161023'] = 'STS'
date_area['161026'] = 'STS'
date_area['161029'] = 'IT'
date_area['161118'] = 'STS'
date_area['161121'] = 'STS'
date_area['161125'] = 'STS'
date_area['161202'] = 'STS'
date_area['161206'] = 'IT'
date_area['161222'] = 'STS'
date_area['161228'] = 'IT'
date_area['170103'] = 'IT'
date_area['170106'] = 'STS'
date_area['170113'] = 'IT'
date_area['170117'] = 'IT'
date_area['170214'] = 'STS'
date_area['170221'] = 'STS'

area = np.array([date_area[i] for i in list_date])

plot_highlight = ''
if plot_highlight == 'nov':
    alphas = [1,1,1,0,0,0]
    alphas_p = [0, 0, 0, 1, 0]
elif plot_highlight == 'fam':
    alphas = [0,0,0,1,1,1]
    alphas_p = [0, 0, 0, 0, 1]
elif plot_highlight == '00':
    alphas = [1,0,0,1,0,0]
    alphas_p = [1, 0, 0, 0, 0]
elif plot_highlight == '50':
    alphas = [0,1,0,0,1,0]
    alphas_p = [0, 1, 0, 0, 0]
elif plot_highlight == '70':
    alphas = [0,0,1,0,0,1]
    alphas_p = [0, 0, 1, 0, 0]
elif plot_highlight == '':
    alphas = [1, 1, 1, 1, 1, 1]
    alphas_p = [1, 1, 1, 1, 1]
else:
    alphas = [1, 1, 1, 1, 1, 1]
    alphas_p = [1, 1, 1, 1, 1]

colors = np.vstack([pnp.gen_distinct_colors(3, luminance=0.9), pnp.gen_distinct_colors(3, luminance=0.7)])
linestyles = ['-', '-', '-', '--', '--', '--']

[h_fig, h_ax]=plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=[12,8])
plt.axes(h_ax[0,0])
for i in range(6):
    plt.plot(ts, np.mean(decode_result, axis=-1)[i,:,0].transpose(), color=colors[i], linestyle=linestyles[i], alpha=alphas[i] )
plt.title('V4')
plt.ylabel('Image Decoding Accuracy')

plt.axes(h_ax[1, 0])
for i in range(6):
    plt.plot(ts,np.mean(decode_result, axis=-1)[i, :, 1].transpose(), color=colors[i], linestyle=linestyles[i], alpha=alphas[i])
plt.xlabel('time (s)')
plt.ylabel('Noise Decoding Accuracy')

plt.axes(h_ax[0,1])
for i in range(6):
    plt.plot(ts, np.mean(decode_result[:,:,:,area=='IT'], axis=-1)[i,:,2].transpose(), color=colors[i], linestyle=linestyles[i], alpha=alphas[i] )
plt.title('IT')

plt.axes(h_ax[1,1])
for i in range(6):
    plt.plot(ts, np.mean(decode_result[:,:,:,area=='IT'], axis=-1)[i,:,3].transpose(), color=colors[i], linestyle=linestyles[i], alpha=alphas[i] )
plt.xlabel('time (s)')

plt.axes(h_ax[0, 2])
for i in range(6):
    plt.plot(ts, np.mean(decode_result[:,:,:,area=='STS'], axis=-1)[i,:,2].transpose(), color=colors[i], linestyle=linestyles[i], alpha=alphas[i] )
plt.title('STS')

plt.axes(h_ax[1,2])
for i in range(6):
    plt.plot(ts, np.mean(decode_result[:,:,:,area=='STS'], axis=-1)[i,:,3].transpose(), color=colors[i], linestyle=linestyles[i], alpha=alphas[i] )

plt.xlabel('time (s)')
plt.xticks(np.arange(-0.1,0.5+0.01,0.1))
plt.legend(cdtn)
plt.suptitle('decoding result by brain region')
plt.savefig('./temp_figs/decoding_by_area_{}.pdf'.format(plot_highlight))
plt.savefig('./temp_figs/decoding_by_area_{}.png'.format(plot_highlight))

