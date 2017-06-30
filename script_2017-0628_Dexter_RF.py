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

# dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank'
# dir_dg='/shared/homes/sguan/neuro_data/stim_dg'
dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT'
dir_dg='/shared/lab/projects/analysis/ruobing/data_dg'


keyword_tank = '.*Dexter.*170629.*'
keyword_block= 'x.*spotopto.*'

name_file =  keyword_block
tankname =   keyword_tank
# signal_type = 'spk'
signal_type = 'LFP'
tf_plot_movie = True

def RF_feature_mapping(name_file, tankname, signal_type='spk'):
    """ load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(name_file, tankname, tf_interactive=True,
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
    if signal_type == 'spk':
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.020, 0.200], type_filter='spiketrains.*',
                                                   name_filter='.*Code[1-9]$', spike_bin_rate=100)
    elif signal_type == 'LFP':
        data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.1, 0.5], type_filter='ana.*', name_filter='LFPs.*')
    # ERP plot
    # ERP = np.mean(data_neuro['data'][data_df['stimulate_type'] == 0,:,:], axis=0).transpose()
    # pnp.ErpPlot(ERP[0:32, :], data_neuro['ts'], array_layout=layout_GM32)

    # group by x,y cooridnate
    data_neuro = signal_align.neuro_sort(data_df, ['stim_pos_x', 'stim_pos_y'], data_df['stimulate_type']==3, data_neuro)

    if signal_type == 'LFP':
        pass

    # plot
    def plot_RF(t_window_plot=[0.0,0.15]):
        h_fig, h_axes = pnp.create_array_layout_subplots(layout_GM32, tf_linear_indx=True)
        h_fig.set_size_inches([10, 9], forward=True)
        plt.tight_layout()
        for i in sorted(range(len(data_neuro['signal_info'])), reverse=True):
            ch = data_neuro['signal_info'][i]['channel_index']
            if ch <= 32:
                plt.axes(h_axes[ch - 1])
                pnp.RfPlot(data_neuro, indx_sgnl=i, t_focus=t_window_plot,
                           psth_overlay=False, tf_scr_ctr=True)

    if tf_plot_movie:
        for t_start in np.arange(0, 500, 50):
            plot_RF([t_start * 0.001, t_start * 0.001 + 0.05])
            plt.suptitle(tankname)
            # plt.savefig('./temp_figs/Dexter_{}_RF_{:0>4}.png'.format(keyword_tank, t_start))
    else:
        plot_RF()

    # plot every chan one figure
    # for i in range(len(data_neuro['signal_info'])):
    #     pnp.RfPlot(data_neuro, indx_sgnl=i, t_focus=[0.050, 0.150])
    #     plt.gcf().set_size_inches([12, 9])
    #     plt.show()
    #     plt.pause(1.0)




RF_feature_mapping(keyword_block, tankname=keyword_tank)

for tankname in list_name_tanks:
    try:
        RF_feature_mapping('.*V4_spot', tankname=tankname)
        plt.savefig('./temp_figs/RF_GM32_{}.png'.format(tankname))
        plt.close()
    except:
        print('fail to plot rf for tank {}'.format(tankname))