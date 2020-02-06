""" script to load all dataset and save to hdf5 """

import os
import numpy as np
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import re                   # regular expression
import time                 # time code execution
import pickle
import warnings

import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import PyNeuroData as pnd   # in this package: align neural data according to task
import misc_tools           # in this package: misc
import store_hdf5           # save/load data using hdf5

import data_load_DLSH       # package specific for DLSH lab data

""" ========== set parameters ========== """
scenario = 'Thor'    # one of 'Thor', 'Dante', 'Dexter'
block_type = 'srv_mask'
# block_type = 'matchnot'


if scenario=='Thor':

    # ----- data directory
    dir_data_save = '/shared/homes/sguan/Coding_Projects/support_data'
    dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT/'
    dir_dg = '/shared/lab/projects/analysis/shaobo/data_dg'

    # ----- tank names to use
    list_name_tanks = os.listdir(dir_tdt_tank)
    keyword_tank = '.*Thor.*U16'
    list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
    list_name_tanks = sorted(list_name_tanks)

    # ----- filename (blockname) to use
    if block_type == 'srv_mask' :
        t_plot = [-0.200, 0.600]
    elif block_type == 'matchnot':
        t_plot = [-0.200, 0.600]
    else:
        t_plot = [-0.200, 0.600]
    block_name_filter = 'h_.*{}.*'.format(block_type)

    h5_filepath = '{}/data_neuro_Thor_U16_all.hdf5'.format(dir_data_save, block_type)


elif scenario=='Dante':

    dir_data_save = '/shared/homes/sguan/Coding_Projects/support_data'
    dir_tdt_tank = '/shared/homes/sguan/neuro_data/tdt_tank'
    dir_dg = '/shared/homes/sguan/neuro_data/stim_dg'

    list_name_tanks = os.listdir(dir_tdt_tank)
    keyword_tank = '.*GM32.*U16'
    list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
    list_name_tanks_0 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is None]
    list_name_tanks_1 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is not None]
    list_name_tanks = sorted(list_name_tanks_0) + sorted(list_name_tanks_1)

    # ----- filename (blockname) to use
    if block_type == 'matchnot':
        t_plot = [-0.200, 1.200]
    else:
        t_plot = [-0.200, 0.600]
    block_name_filter = 'd_.*{}.*'.format(block_type)

    h5_filepath = '{}/data_neuro_Dante_GM32_U16_all.hdf5'.format(dir_data_save, block_type)


""" ========== define functions ========== """

def LoadDataOneDay(tankname, block_name_filter=block_name_filter):

    date_code = re.match('.*?-(\d{6}).*', tankname).group(1)

    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(block_name_filter, tankname,
                                                               tf_interactive=False,
                                                               dir_tdt_tank=dir_tdt_tank,
                                                               dir_dg=dir_dg)

    filename_common = misc_tools.str_common(name_tdt_blocks)
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    """ Get StimOn time stamps in neo time frame """
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')

    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)

    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk = data_load_DLSH.standardize_blk(blk)

    data_neuro_lfp = pnd.blk_align_to_evt(blk, ts_StimOn, t_plot,
                                          type_filter='ana.*', name_filter='LFPs.*')

    data_neuro_spk = pnd.blk_align_to_evt(blk, ts_StimOn, t_plot,
                                          type_filter='spiketrains.*', name_filter='.*Code[1-9]$',
                                          spike_bin_rate=data_neuro_lfp['signal_info'][0]['sampling_rate'])

    pnd.include_trial_info(data_neuro_lfp, data_df=data_df)
    pnd.include_trial_info(data_neuro_spk, data_df=data_df)

    return [date_code, data_neuro_spk, data_neuro_lfp, data_df]


def SaveDataOneDay(date_code, data_neuro_spk, data_neuro_lfp, h5_filepath=h5_filepath):

    store_hdf5.SaveToH5(data_neuro_spk, h5_filepath=h5_filepath, h5_groups=[date_code, block_type, 'spk'])
    store_hdf5.SaveToH5(data_neuro_lfp, h5_filepath=h5_filepath, h5_groups=[date_code, block_type, 'lfp'])


for tankname in list_name_tanks:
    try:
        [date_code, data_neuro_spk, data_neuro_lfp, data_df] = LoadDataOneDay(tankname)
        SaveDataOneDay(date_code, data_neuro_spk, data_neuro_lfp, h5_filepath)
    except:
        warnings.warn('tank {} cannot be processed'.format(tankname))

store_hdf5.ShowH5(h5_filepath=h5_filepath)

