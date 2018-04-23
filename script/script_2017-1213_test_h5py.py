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
import pickle
import warnings
import h5py

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


""" load data """

dir_data_save = '/shared/homes/sguan/Coding_Projects/support_data'
dir_tdt_tank='/shared/lab/projects/encounter/data/TDT/'
list_name_tanks = os.listdir(dir_tdt_tank)

keyword_tank = '.*GM32.*U16'
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_name_tanks_0 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is None]
list_name_tanks_1 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is not None]
list_name_tanks = sorted(list_name_tanks_0) + sorted(list_name_tanks_1)

block_type = 'srv_mask'
# block_type = 'matchnot'
if block_type == 'matchnot':
    t_plot = [-0.200, 1.200]
else:
    t_plot = [-0.200, 0.600]


def LoadDataOneDay(tankname):

    date_code = re.match('.*-(\d{6})-.*', tankname).group(1)

    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*{}.*'.format(block_type), tankname, tf_interactive=False,
                                                               dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank',
                                                               dir_dg='/shared/homes/sguan/neuro_data/stim_dg')

    """ Get StimOn time stamps in neo time frame """
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'

    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    data_neuro_lfp = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*',
                                                   name_filter='LFPs.*',
                                                   chan_filter=range(1, 48 + 1))

    data_neuro_spk = signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*',
                                                       name_filter='.*Code[1-9]$', spike_bin_rate=data_neuro_lfp['signal_info'][0]['sampling_rate'],
                                                       chan_filter=range(1, 48 + 1))


    return [date_code, data_neuro_spk, data_neuro_lfp, data_df]


def get_signal_id(signal_info, date_code):
    signal_info = pd.DataFrame(signal_info)
    signal_info['date'] = date_code
    signal_info['signal_id'] = signal_info['date'].apply(lambda x: '{:0>6}'.format(x)).str.cat(
        [signal_info['channel_index'].apply(lambda x: '{:0>2}'.format(x)),
        signal_info['sort_code'].apply(lambda x: '{:0>1}'.format(x))],
        sep='_'
        )
    return np.array(signal_info['signal_id']).astype('str')


def get_df_valid_for_hdf5(data_df):
    keep_col = []
    for col in data_df.keys():
        if col=='':
            pass
        elif data_df[col].dtype != np.dtype('O'):
            keep_col.append(col)
        elif type(data_df[col][0]) is str:
            keep_col.append(col)
        else:
            pass
    return data_df[keep_col]


def SaveDataOneDay(date_code, data_neuro_spk, data_neuro_lfp, data_df):

    signal_id_spk = get_signal_id(data_neuro_spk['signal_info'], date_code)
    signal_id_lfp = get_signal_id(data_neuro_lfp['signal_info'], date_code)
    data_df_str = data_df.to_json(path_or_buf=None)
    data_df_valid_for_h5 = get_df_valid_for_hdf5(data_df)

    hf = h5py.File('{}/all_data_dante_{}.hdf5'.format(dir_data_save, block_type), 'a')

    if date_code not in hf.keys():
        h_data_day = hf.create_group(date_code)
        hf[date_code].create_dataset('trial_info_json', data=data_df_str)
    if 'spk' not in hf[date_code]:
        hf[date_code].create_group('spk')
        hf[date_code]['spk'].create_dataset('data', data=data_neuro_spk['data'])
        hf[date_code]['spk'].create_dataset('ts', data=data_neuro_spk['ts'])
        hf[date_code]['spk'].create_dataset('signal_id', data=signal_id_spk)
    if 'lfp' not in hf[date_code]:
        hf[date_code].create_group('lfp')
        hf[date_code]['lfp'].create_dataset('data', data=data_neuro_lfp['data'])
        hf[date_code]['lfp'].create_dataset('ts', data=data_neuro_lfp['ts'])
        hf[date_code]['lfp'].create_dataset('signal_id', data=signal_id_lfp)
    hf.close()
    hf_pandas = pd.HDFStore('{}/all_data_dante_{}.hdf5'.format(dir_data_save, block_type))
    hf_pandas.put(key='{}/trial_info'.format(date_code), value=data_df_valid_for_h5, format='table', data_columns=True)
    hf_pandas.close()

for tankname in list_name_tanks:
    try:
        [date_code, data_neuro_spk, data_neuro_lfp, data_df] = LoadDataOneDay(tankname)
        SaveDataOneDay(date_code, data_neuro_spk, data_neuro_lfp, data_df)
    except:
        warnings.warn('tank {} cannot be processed'.format(tankname))








""" ========== below: temp script for testing ========== """

signal_info_detail = pd.read_pickle('/shared/homes/sguan/Coding_Projects/support_data/spike_wf_info_Dante.pkl')
signal_info = pd.DataFrame(data_neuro['signal_info'])
signal_info['date'] = date_code
signal_info = signal_info.merge(signal_info_detail, how='inner', on=['date','channel_index','sort_code'], copy=False)

def df_col_type(df):
    for key in df.keys():
        print type(df[key][0])
df_col_type(df)

data_df_str = data_df.to_json(path_or_buf=None)

# write to hdf5
hf = h5py.File('{}/test_h5py.hdf5'.format(dir_data_save), 'a')

if date_code not in hf.keys():
    h_data_day = hf.create_group(date_code)
if 'data_spk' not in hf[date_code]:
    hf[date_code].create_group('data_spk')
    hf[date_code]['data_spk'].create_dataset('data', data = data_neuro['data'])
    hf[date_code]['data_spk'].create_dataset('ts', data=data_neuro['ts'])
    hf[date_code]['data_spk'].create_dataset('trial_info', data=data_df_str)
# if 'pd' not in hf.keys():
hf.close()



hf = pd.HDFStore('{}/test_h5py.hdf5'.format(dir_data_save))
hf.put(key='pd', value=signal_info, format='table', data_columns=[])
hf.close()


#data_sub.to_hdf('{}/test_h5py.hdf5'.format(dir_data_save), key='data_df')
temp = pd.read_hdf('{}/test_h5py.hdf5'.format(dir_data_save), key='data_df')






data_df_valid = get_df_valid_for_hdf5(data_df)
get_df_valid_for_hdf5(signal_info_detail).keys()



hf = pd.HDFStore('{}/test_h5py.hdf5'.format(dir_data_save))
hf.put(key='sub/pd', value=data_df_valid, format='table', data_columns=True)
hf.close()
