"""" Import modules """
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

dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank'
dir_dg='/shared/homes/sguan/neuro_data/stim_dg'

keyword_tank = '.*GM32.*U16'
list_name_tanks = os.listdir(dir_tdt_tank)
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_name_tanks_0 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is None]
list_name_tanks_1 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is not None]
list_name_tanks = sorted(list_name_tanks_0) + sorted(list_name_tanks_1)


def RF_feature_mapping(name_file, tankname):
    """ load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(name_file, tankname, tf_interactive=False,
                                                               dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg)

    """ Get StimOn time stamps in neo time frame """
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'

    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    """ waveform plot """
    # pnp.SpkWfPlot(blk.segments[0])

    """ ERP plot """
    data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.1, 0.5], type_filter='ana.*', name_filter='LFPs.*')
    ERP = np.mean(data_neuro['data'], axis=0).transpose()
    # pnp.ErpPlot(ERP[0:32, :], data_neuro['ts'], array_layout=layout_GM32)


    # rf spot mapping
    if re.match('.*spot.*', filename_common) is not None:

        # align, RF plot
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.020, 0.200], type_filter='spiketrains.*',
                                                   name_filter='.*Code[1-9]$', spike_bin_rate=100)
        # group
        data_neuro = signal_align.neuro_sort(data_df, ['stim_pos_x', 'stim_pos_y'], [], data_neuro)

        # plot
        h_fig, h_axes = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
        h_fig.set_size_inches([10, 9], forward=True)
        plt.tight_layout()
        for i in sorted(range(len(data_neuro['signal_info'])), reverse=True):
            ch = data_neuro['signal_info'][i]['channel_index']
            if ch<=32:
                plt.axes(h_axes[ch-1])
                pnp.RfPlot(data_neuro, indx_sgnl=i, t_focus=[0.050,0.150], psth_overlay=False)
        plt.suptitle(tankname)
        # plot every chan one figure
        # for i in range(len(data_neuro['signal_info'])):
        #     pnp.RfPlot(data_neuro, indx_sgnl=i, t_focus=[0.050, 0.150])
        #     plt.gcf().set_size_inches([12, 9])
        #     plt.show()
        #     plt.pause(1.0)


    elif re.match('.*srv_mask.*', filename_common) is not None:
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.1, 0.5], type_filter='spiketrains.*',
                                                   name_filter='.*Code[1-9]$', spike_bin_rate=1000)
        data_neuro = signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro)
        functionPlot = lambda x: pnp.PsthPlot(x, data_neuro['ts'],
                                              subpanel='auto', sk_std=0.015, tf_legend=False)
        ts = data_neuro['ts']

        for i_neuron in range(len(data_neuro['signal_info'])):
            name_signal = data_neuro['signal_info'][i_neuron]['name']
            pnp.PsthPlotCdtn(data_neuro['data'][:, :, i_neuron], data_df, data_neuro['ts'], cdtn_l_name='stim_names',
                             cdtn0_name='stim_familiarized', cdtn1_name='mask_opacity_int', subpanel='auto',
                             sk_std=0.010)
            plt.suptitle('file {},   signal {}'.format(filename_common, name_signal, fontsize=20))
            plt.gcf().set_size_inches(8, 4)
            plt.show()
            plt.close()

for tankname in list_name_tanks:
    try:
        RF_feature_mapping('.*V4_spot', tankname=tankname)
        plt.savefig('./temp_figs/RF_GM32_{}.png'.format(tankname))
        plt.close()
    except:
        print('fail to plot rf for tank {}'.format(tankname))