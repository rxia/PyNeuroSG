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


dir_tdt_tank='/shared/lab/projects/encounter/data/TDT'
dir_dg='/shared/lab/projects/analysis/shaobo/data_dg'


keyword_block = 'y_.*003'
keyword_tank = 'Murphy.*170811.*'

[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(keyword=keyword_block, keyword_tank=keyword_tank, mode='tdt',
                                                           tf_interactive=True, dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg)

ts_StimOn = [np.array(blk.segments[0].events[2].times)]

data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.100, 0.600], type_filter='analog.*',
                                           name_filter='.*LFPs', spike_bin_rate=1000)


ERPs = np.mean(data_neuro['data'], axis=0).transpose()
pnp.ErpPlot(ERPs)
lfp = ERPs
ts = data_neuro['ts']

# chan_bad = chan_bad_all[date_i]
lambda_dev = np.ones(16)
# lambda_dev[chan_bad]=0
# lfp = ERP[date==date_i, :]
lfp_na = lfp

lfp_sm = pna.lfp_cross_chan_smooth(lfp, method='der', lambda_dev=lambda_dev, lambda_der=5, sigma_t=1)

reload(pna); temp=time.time(); lfp_sm = pna.lfp_csd_smooth(lfp, lambda_dev=lambda_dev, lambda_der=5, sigma_t=3); print(time.time()-temp);
lfp_nr = lfp_sm / np.std(lfp_sm, axis=1, keepdims=True)

csd_na = pna.cal_1dCSD(lfp_na, axis_ch=0, tf_edge=True)
csd_sm = pna.cal_1dCSD(lfp_sm, axis_ch=0, tf_edge=True)
csd_nr = pna.cal_1dCSD(lfp_nr, axis_ch=0, tf_edge=True)

_, h_axes = plt.subplots(2, 3, figsize=[12, 8], sharex=True, sharey=True)
plt.axes(h_axes[0, 0])
pnp.ErpPlot_singlePanel(lfp, ts)
# plt.plot([0] * len(chan_bad), chan_bad, 'ok')
plt.title('LFP original')
plt.xlabel('time (s)')
plt.ylabel('chan')
plt.xticks(np.arange(-0.1, 0.51, 0.1))

plt.axes(h_axes[0, 1])
pnp.ErpPlot_singlePanel(lfp_sm, ts)
plt.title('LFP smoothed')
plt.axes(h_axes[0, 2])
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

plt.axes(h_axes[1, 2])
pnp.ErpPlot_singlePanel(csd_nr, ts, tf_inverse_color=True)
plt.title('CSD normalized')

plt.savefig('./temp_figs/CSD_Murphy.png')


"""  test new CSD code """
reload(pna); temp=time.time(); lfp_sm = pna.lfp_robust_smooth(lfp, lambda_dev=lambda_dev, lambda_der=5, sigma_t=3); print(time.time()-temp);

reload(pna); temp=time.time(); lfp_sm = pna.lfp_robust_smooth(lfp, lambda_dev=lambda_dev, lambda_der=5, sigma_t=3, tf_grad=False); print(time.time()-temp);

reload(pna); temp=time.time(); lfp_sm = pna.lfp_robust_smooth(lfp, lambda_dev=lambda_dev, lambda_der=5, sigma_t=3, tf_grad=False, tf_x0_inherent=False); print(time.time()-temp);

reload(pna); temp=time.time(); csd_sm = pna.cal_robust_csd(lfp, lambda_dev=lambda_dev, lambda_der=5, sigma_t=3); print(time.time()-temp);
pnp.ErpPlot_singlePanel(csd_sm, ts, tf_inverse_color=True)