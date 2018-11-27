""" script to generate the final result figures """

import importlib
import os
import warnings
import numpy as np
import scipy as sp
import pandas as pd
import h5py
import matplotlib as mlp
import matplotlib.pyplot as plt
import pickle

import PyNeuroData as pnd
import PyNeuroAna  as pna
import PyNeuroPlot as pnp
import store_hdf5
import df_ana
import misc_tools


# set path to data
path_to_hdf5_data_Dante = '../support_data/data_neuro_Dante_GM32_U16_all.hdf5'
path_to_hdf5_data_Thor = '../support_data/data_neuro_Thor_U16_all.hdf5'
path_to_fig = './temp_figs/final_Dante_Thor'


path_to_U_probe_loc = './script/U16_location_Dante_Thor.csv'

##
""" get U-probe location """
# granular layer location is zero-indexed
loc_U16 = pd.read_csv(path_to_U_probe_loc)
loc_U16['date'] = loc_U16['date'].astype('str')

def sinal_extend_loc_info(signal_all):
    signal_extend = pd.merge(signal_all, loc_U16, 'left', on='date')

    signal_extend['channel_index_U16'] = signal_extend['channel_index'] - 32*(signal_extend['animal']=='Dante')
    signal_extend['area'] = np.where((signal_extend['animal']=='Dante') & (signal_extend['channel_index']<=32),
                                     ['V4']*len(signal_extend), signal_extend['area'])

    signal_extend['depth'] = 0
    signal_extend['depth'] = (signal_extend['channel_index_U16']-1-signal_extend['granular']) * (signal_extend['area']=='TEd') \
                             +(16-signal_extend['channel_index_U16']-signal_extend['granular']) * (signal_extend['area']=='TEm')
    signal_extend['depth'] = np.where(signal_extend['area']=='V4',
                                     np.zeros(len(signal_extend))*np.nan, signal_extend['depth'])

    return signal_extend


def load_data_of_day(datecode):
    if datecode <= '180000':
        animal = 'Dante'
        path_to_hdf5_data = path_to_hdf5_data_Dante
    else:
        animal = 'Thor'
        path_to_hdf5_data = path_to_hdf5_data_Thor
    data_neuro = store_hdf5.LoadFromH5(path_to_hdf5_data, [datecode, 'srv_mask', 'spk'])
    return data_neuro


##
""" ========== example day and neuron ========== """
datecode = '161125'

data_neuro = load_data_of_day(datecode)
df_ana.GroupDataNeuro(data_neuro, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                      groupby=['stim_familiarized', 'mask_opacity_int'])

# pnp.PsthPlotMultiPanel(data_neuro, index_signal=3, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
#                        groupby_subplots=['stim_familiarized', 'mask_opacity_int'],
#                        groupby_panel='stim_categories', color_style='continuous',
#                        sk_std=0.010, tf_legend=False, figsize=(9, 6),
#                        )
# plt.xlim([-0.050, 0.450])
#
#
# pnp.PsthPlotMultiPanel(data_neuro, index_signal=24, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
#                        groupby_subplots=['stim_familiarized', 'mask_opacity_int'],
#                        groupby_panel='stim_categories',
#                        sk_std=0.010, tf_legend=False, figsize=(9, 6),
#                        )
# plt.xlim([-0.050, 0.450])

index_signal = 3
pnp.PsthPlotMultiPanel(data_neuro, index_signal=index_signal, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                       groupby_subplots=['', 'stim_familiarized'], aggregate_subplots=False,
                       groupby_panel='mask_opacity_int' , color_style='continuous',
                       sk_std=0.010, tf_legend=False, figsize=(6, 3),
                       )
plt.xlim([-0.050, 0.450])
plt.savefig(os.path.join(path_to_fig, 'PSTH_example_V4_{}_{}.pdf'.format(datecode, data_neuro['signal_info']['name'][index_signal])))
plt.savefig(os.path.join(path_to_fig, 'PSTH_example_V4_{}_{}.png'.format(datecode, data_neuro['signal_info']['name'][index_signal])))

index_signal = 24
pnp.PsthPlotMultiPanel(data_neuro, index_signal=index_signal, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                       groupby_subplots=['', 'stim_familiarized'], aggregate_subplots=False,
                       groupby_panel='mask_opacity_int' , color_style='continuous',
                       sk_std=0.010, tf_legend=False, figsize=(6, 3),
                       )
plt.xlim([-0.050, 0.450])
plt.savefig(os.path.join(path_to_fig, 'PSTH_example_IT_{}_{}.pdf'.format(datecode, data_neuro['signal_info']['name'][index_signal])))
plt.savefig(os.path.join(path_to_fig, 'PSTH_example_IT_{}_{}.png'.format(datecode, data_neuro['signal_info']['name'][index_signal])))


##
""" save figure for every neuron """
tree_struct_hdf5_Dante = store_hdf5.ShowH5(path_to_hdf5_data_Dante, yn_print=False)
tree_struct_hdf5_Thor  = store_hdf5.ShowH5(path_to_hdf5_data_Thor , yn_print=False)
all_dates = list(tree_struct_hdf5_Dante.keys()) + list(tree_struct_hdf5_Thor.keys())

""" all neurons """
plt.ioff()
if False:
    for datecode in all_dates:
        data_neuro = load_data_of_day(datecode)
        df_ana.GroupDataNeuro(data_neuro, limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                              groupby=['stim_familiarized', 'mask_opacity_int'])

        for index_signal in range(data_neuro['data'].shape[2]):
            try:
                pnp.PsthPlotMultiPanel(data_neuro, index_signal=index_signal,
                                       limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                                       groupby_subplots=['', 'stim_familiarized'], aggregate_subplots=False,
                                       groupby_panel='mask_opacity_int', color_style='continuous',
                                       sk_std=0.010, tf_legend=False, figsize=(6, 3),
                                       )
                plt.xlim([-0.050, 0.450])
                plt.savefig(os.path.join(path_to_fig, 'psth_by_neuron', 'PSTH_{}_{}.pdf'.format(datecode,
                                                                                     data_neuro['signal_info']['name'][
                                                                                         index_signal])))

                pnp.PsthPlotMultiPanel(data_neuro, index_signal=index_signal,
                                       limit=data_neuro['trial_info']['mask_opacity_int'] < 80,
                                       groupby_subplots=['stim_sname'],
                                       aggregate_subplots=False, linearize_subplots=True, ncol=4,
                                       groupby_panel='mask_opacity_int', color_style='continuous',
                                       sk_std=0.010, tf_legend=False, figsize=(10, 8),
                                       )
                plt.xlim([-0.050, 0.450])
                plt.savefig(os.path.join(path_to_fig, 'psth_by_neuron', 'PSTH_image_{}_{}.pdf'.format(datecode,
                                                                                                data_neuro[
                                                                                                    'signal_info'][
                                                                                                    'name'][
                                                                                                    index_signal])))

                plt.close('all')
            except:
                pass



