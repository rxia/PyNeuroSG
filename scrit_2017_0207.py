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

""" load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
# name_file = '.*V4_spot.*'
# name_file = '.*MT_mapping.*'
name_file = '.*V4_texture.*'
[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(name_file, '.*GM32.*170207.*', tf_interactive=True,)

""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'

""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk     = data_load_DLSH.standardize_blk(blk)


# rf spot mapping
if re.match('.*spot.*', filename_common) is not None:
    # align, RF plot
    data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.020, 0.200], type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=100)
    # group
    data_neuro=signal_align.neuro_sort(data_df, ['stim_pos_x','stim_pos_y'], [], data_neuro)
    # plot
    pnp.RfPlot(data_neuro, indx_sgnl=0, t_scale=0.2)
    for i in range(len(data_neuro['signal_info'])):
        pnp.RfPlot(data_neuro, indx_sgnl=i, t_focus=[0.050,0.100])
        try:
            plt.savefig('{}/{}_RF_{}.png'.format(dir_temp_fig, filename_common, data_neuro['signal_info'][i]['name']))
        except:
            plt.savefig('./temp_figs/RF_plot_' + misc_tools.get_time_string() + '.png')
        plt.close()

elif re.match('.*MT_mapping.*', filename_common) is not None:
    data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.020, 0.500], type_filter='spiketrains.*',
                                               name_filter='.*Code[1-9]$', spike_bin_rate=1000)
    data_neuro = signal_align.neuro_sort(data_df, ['rotationDirections',], [], data_neuro)
    # plot
    for i in range(len(data_neuro['signal_info'])):
        if False:   # one pannel
            pnp.PsthPlot(data_neuro['data'][:, :, i], data_neuro['ts'], cdtn=data_df['rotationDirections'],
                             subpanel='auto', sk_std=0.025, tf_legend='True', color_style='continuous')
            plt.gcf().set_size_inches([6.4, 4.8])
        else:       # seperate panels
            functionPlot = lambda x: pnp.PsthPlot(x, data_neuro['ts'],
                                                  subpanel='auto', sk_std=0.015, tf_legend='True')
            pnp.SmartSubplot(data_neuro, functionPlot, data_neuro['data'][:, :, i],
                             suptitle='{}  {}'.format(data_neuro['signal_info'][i]['name'], filename_common))
            plt.gcf().set_size_inches([8, 6])
        plt.savefig('{}/{}_motion_dir_{}.png'.format(dir_temp_fig, filename_common, data_neuro['signal_info'][i]['name']))
        plt.close()

elif re.match('.*V4_texture.*', filename_common) is not None:
    data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.020, 0.200], type_filter='spiketrains.*',
                                               name_filter='.*Code[1-9]$', spike_bin_rate=1000)
    data_neuro = signal_align.neuro_sort(data_df, [''], [], data_neuro)
    functionPlot = lambda x: pnp.PsthPlot(x, data_neuro['ts'],
                                          subpanel='auto', sk_std=0.015, tf_legend=False)
    for i in range(len(data_neuro['signal_info'])):
        if True:   # one pannel
            pnp.PsthPlot(data_neuro['data'][:, :, i], data_neuro['ts'], cdtn=data_df['stim_names'],
                             subpanel='', sk_std=0.010, tf_legend=False)
            plt.suptitle('{}  {}'.format(data_neuro['signal_info'][i]['name'], filename_common))
            plt.gcf().set_size_inches([6.4, 4.8])
        else:
            pnp.SmartSubplot(data_neuro, functionPlot, data_neuro['data'][:, :, i],
                             suptitle='{}  {}'.format(data_neuro['signal_info'][i]['name'], filename_common))
            plt.gcf().set_size_inches([8, 6])
        plt.savefig('{}/{}_texture_{}.png'.format(dir_temp_fig, filename_common, data_neuro['signal_info'][i]['name']))
        plt.close()