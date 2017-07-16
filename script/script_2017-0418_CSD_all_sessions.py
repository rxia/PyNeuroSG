# ----- standard modules -----
import os
import sys
sys.path.append('/shared/homes/sguan/Coding_Projects/PyNeuroSG')

import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import time                 # time code execution
import datetime
import pickle

# ----- modules used to read neuro data -----
import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import quantities as pq

# ----- modules of the project PyNeuroSG -----
import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc

# ----- modules for the data location and organization in Sheinberg lab -----
import data_load_DLSH       # package specific for DLSH lab data
from GM32_layout import layout_GM32


# load data computed in scrip_ERP_all_sessions
tf_first_stim = True
if tf_first_stim:
    erp_df_full = pd.read_pickle('./temp_data/erp_all_info_first_stim')
else:
    erp_df_full = pd.read_pickle('./temp_data/erp_all_info')


erp_df = erp_df_full[erp_df_full['chan']>=33]

ERP = np.array(erp_df['ERP'].tolist())
ts = np.linspace(-0.1,0.5,num=611)

chan = np.array(erp_df['chan'].tolist())
date = np.array(erp_df['date'].tolist())
N=16

chan_bad_all = dict()
chan_bad_all['161015'] = [5, 7, 8, 14, 15]
chan_bad_all['161023'] = [5, 7, 8, 14, 15]
chan_bad_all['161026'] = []
chan_bad_all['161029'] = [2,10]
chan_bad_all['161118'] = [2]
chan_bad_all['161121'] = [2,10,12]
chan_bad_all['161125'] = [2,10,12]
chan_bad_all['161202'] = [2,10,12]
chan_bad_all['161206'] = [2,10,12]
chan_bad_all['161222'] = [2,8,10]
chan_bad_all['161228'] = [2,8,10]
chan_bad_all['170103'] = [2,8,10]
chan_bad_all['170106'] = [2,8,10]
chan_bad_all['170113'] = [2,8,10]
chan_bad_all['170117'] = [2,8,10]
chan_bad_all['170214'] = [2,8,10]
chan_bad_all['170221'] = [2,8,12]

for date_i in np.unique(date):
    plt.close()

    chan_bad = chan_bad_all[date_i]
    lambda_dev = np.ones(16)
    lambda_dev[chan_bad]=0
    lfp = ERP[date==date_i, :]
    lfp_na = lfp

    _, h_axes = plt.subplots(2,3, figsize=[12,8], sharex=True, sharey=True)
    plt.axes(h_axes[0,0])
    pnp.ErpPlot_singlePanel(lfp, ts)
    plt.plot([0]*len(chan_bad), chan_bad, 'ok')
    plt.title('LFP original')
    plt.xlabel('time (s)')
    plt.ylabel('chan')
    plt.xticks(np.arange(-0.1,0.51,0.1))

    try:
        spike_quanlity = spikes_U16[date_U16==date_i,:]
        plt.scatter(np.repeat(ts[0],N), np.arange(N), c=spike_quanlity, vmin=-3, vmax=3, cmap='Spectral', edgecolors='k', s=100)
    except:
        pass

    if True:
        lfp_sm = pna.lfp_cross_chan_smooth(lfp, method='der', lambda_dev=lambda_dev, lambda_der=5, sigma_t=0.5)
        lfp_nr = lfp_sm/np.std(lfp_sm, axis=1, keepdims=True)

        csd_na = pna.cal_1dCSD(lfp_na, axis_ch=0, tf_edge=True)
        csd_sm = pna.cal_1dCSD(lfp_sm, axis_ch=0, tf_edge=True)
        csd_nr = pna.cal_1dCSD(lfp_nr, axis_ch=0, tf_edge=True)

        plt.axes(h_axes[0,1])
        pnp.ErpPlot_singlePanel(lfp_sm, ts)
        plt.title('LFP smoothed')
        plt.axes(h_axes[0,2])
        pnp.ErpPlot_singlePanel(lfp_nr, ts)
        plt.title('LFP normalized')

        plt.axes(h_axes[1, 0])
        pnp.ErpPlot_singlePanel(csd_na, ts, tf_inverse_color=True)
        plt.title('CSD native')
        plt.xlabel('time (s)')
        plt.ylabel('chan')

        plt.axes(h_axes[1, 1])
        pnp.ErpPlot_singlePanel(csd_sm, ts, tf_inverse_color=True)
        plt.title('CSD smoothed')
        try:
            spike_quanlity = spikes_U16[date_U16 == date_i, :]
            plt.scatter(np.repeat(ts[0], N), np.arange(N), c=spike_quanlity, vmin=-3, vmax=3, cmap='Spectral',
                        edgecolors='k', s=100)
        except:
            pass
        plt.axes(h_axes[1, 2])
        pnp.ErpPlot_singlePanel(csd_nr, ts, tf_inverse_color=True)
        plt.title('CSD normalized')
        try:
            spike_quanlity = spikes_U16[date_U16 == date_i, :]
            plt.scatter(np.repeat(ts[0], N), np.arange(N), c=spike_quanlity, vmin=-3, vmax=3, cmap='Spectral',
                        edgecolors='k', s=100)
        except:
            pass

    plt.suptitle('IT U-probe date {}'.format(date_i))
    plt.savefig('./temp_figs/CSD_IT_{}.pdf'.format(date_i))
    plt.savefig('./temp_figs/CSD_IT_{}.png'.format(date_i))



""" ===== get spike quality information, run this section first if need variable 'spike_quanlity' ===== """
import csv
import re
import numpy as np
import datetime
import warnings
import pickle
from matplotlib import pyplot as plt
import PyNeuroPlot as pnp

path_to_original_csv = '/shared/homes/sguan/neuro_data/' + 'Dante GrayMatter 32 V4 Log - Advance log.csv'
path_to_result_pickle = './temp_data/GM32_log_info.pkl'


item_total = 1
item_count = item_total
list_date = []
list_spikes = []
list_wm     = []

# use csv reader to gather information
with open(path_to_original_csv, 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if re.match('\d{1,2}/\d{1,2}/\d{2,4}', row[0]):   # get date information
            cur_date = datetime.datetime.strptime(row[0], '%m/%d/%Y')
            tf_U16 = False
        if row[0] == 'U16':
            tf_U16 = True
            if item_count != item_total or item_count != item_total:
                warnings.warn('fails to get all {} items before {}'.format(item_total, row[0]) )
            list_date.append(cur_date)
            item_count = 0
        if (row[0] == 'type') and tf_U16==True:

            list_spikes.append(row[1:])
            item_count += 1
            tf_spikes = True

date_str = np.array([datetime.date.strftime(date_cur, '%y%m%d') for date_cur in list_date])
spikes = np.array(list_spikes)[:,:16]
spikes[np.logical_or(spikes=='', spikes==' ')]='0'
spikes = spikes.astype(int)

date_U16 = date_str
spikes_U16=spikes
