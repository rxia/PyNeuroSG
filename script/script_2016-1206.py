

""" Import modules """
# ----- standard modules -----
import os
import sys
import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import time                 # time code execution

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


from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge


plt.ioff()

""" ========== load data ========== """
tankname = '.*GM32.*U16.*161206'
try:
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*matchnot.*', tankname, tf_interactive=False,
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

t_plot = [-0.100, 0.500]

data_neuro_LFP = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='analog.*',
                                               name_filter='LFPs.*', spike_bin_rate=1000,
                                               chan_filter=range(1, 48 + 1))
data_neuro_LFP = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro_LFP)


""" glance spk waveforms """
pnp.SpkWfPlot(blk.segments[0])
plt.savefig('{}/{}_spk_waveform.png'.format(dir_temp_fig, filename_common))


""" ERP plot """
t_plot = [-0.500, 1.200]

data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*', name_filter='LFPs.*')
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro)
ERP = np.mean(data_neuro['data'], axis=0).transpose()
pnp.ErpPlot(ERP, data_neuro['ts'])
plt.savefig('{}/{}_ERP_all.png'.format(dir_temp_fig, filename_common))

# plot GM32
pnp.DataNeuroSummaryPlot(signal_align.select_signal(data_neuro_LFP, chan_filter=range(1,32+1)), sk_std=0.01, signal_type='auto', suptitle='LFP_GM32  {}'.format(filename_common))
plt.savefig('{}/{}_LFP_GM32.png'.format(dir_temp_fig, filename_common))
# plot U16
pnp.DataNeuroSummaryPlot(signal_align.select_signal(data_neuro_LFP, chan_filter=range(33,48+1)), sk_std=0.01, signal_type='auto', suptitle='LFP_U16  {}'.format(filename_common))
plt.savefig('{}/{}_LFP_U16.png'.format(dir_temp_fig, filename_common))

""" spectrum """

[spcg, spcg_t, spcg_f] = pna.ComputeSpectrogram(data_neuro['data'], fs=data_neuro['signal_info'][0][2], f_lim=[0, 120],
                                                t_ini=np.array( data_neuro['ts'][0] ), t_bin=0.2, t_step=None, t_axis=1)
time_baseline = [-0.050, 0.05]
tf_baseline = True
N_sgnl = len(data_neuro['signal_info'])

for i_neuron in range(len(data_neuro['signal_info'] )):
    name_signal = data_neuro['signal_info'][i_neuron]['name']
    functionPlot = lambda x: pnp.SpectrogramPlot(x, spcg_t, spcg_f, tf_log=True, tf_phase=True, tf_mesh_f=True,
                                                 f_lim=[0, 120], time_baseline=None,
                                                 rate_interp=8)
    pnp.SmartSubplot(data_neuro, functionPlot, spcg[:, :, i_neuron, :])
    plt.suptitle('file {},   LFP power spectrum {}'.format(filename_common, name_signal, fontsize=20))
    plt.savefig('{}/{} LFPs power spectrum by condition {}.png'.format(dir_temp_fig, filename_common, name_signal))
    plt.close()


""" coherence """
list_ch0 = [1]
list_ch1 = [48]

for ch0 in list_ch0:
    for ch1 in list_ch1:
        if ch0 == ch1:
            continue
        def functionPlot(x):
            [cohg, spcg_t, spcg_f] = pna.ComputeCoherogram(x, data1=None, tf_phase=True, tf_vs_shuffle=False, fs=data_neuro['signal_info'][0][2],
                                                           t_ini=np.array( data_neuro['ts'][0] ), t_bin=0.2, t_step=None, f_lim=[0, 120])
            pnp.SpectrogramPlot(cohg, spcg_t, spcg_f, tf_log=False, f_lim=[0, 120], tf_phase=True, tf_mesh_f=True,
                                time_baseline=None, rate_interp=8, c_lim_style='from_zero',
                                name_cmap='viridis', tf_colorbar=False)
            del(cohg)
        pnp.SmartSubplot(data_neuro, functionPlot, data_neuro['data'][:,:,[ch0-1,ch1-1]], suptitle='coherence {}_{},    {}'.format(ch0, ch1, filename_common), tf_colorbar=True)
        plt.savefig('{}/{} LFPs coherence by condition {}-{}.png'.format(dir_temp_fig, filename_common, ch0, ch1))
        plt.close()