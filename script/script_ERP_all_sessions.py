""" Import modules """
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


# get tank names
dir_tdt_tank='/shared/lab/projects/encounter/data/TDT/'
list_name_tanks = os.listdir(dir_tdt_tank)
keyword_tank = '.*GM32.*'
# sort by date recorded
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_str_date = [re.match('.*-(\d{6})-.*', name_tank).group(1) for name_tank in list_name_tanks]
list_name_tanks = [y for x,y in sorted(zip(list_str_date, list_name_tanks))]

tf_first_stim = True


def GetERP(tankname='GM32.*U16.*161125', tf_first_stim=tf_first_stim):
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv_mask.*', tankname, tf_interactive=False,
                                                              dir_tdt_tank='/shared/homes/sguan/neuro_data/tdt_tank/',
                                                              dir_dg = '/shared/homes/sguan/neuro_data/stim_dg')

    """ Get StimOn time stamps in neo time frame """
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')


    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'


    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk     = data_load_DLSH.standardize_blk(blk)

    t_plot = [-0.100, 0.500]

    data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot,
                                             type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(1,48+1))
    if tf_first_stim:
        ERP = np.mean(data_neuro['data'][data_df['order']==0, :, :], axis=0).transpose()
    else:
        ERP = np.mean(data_neuro['data'], axis=0).transpose()

    return ERP


""" ========== read data and load ERP ========== """
ERP_all = []
tankname_all= []
for tankname in list_name_tanks:
    try:
        ERP_single = GetERP(tankname=tankname)
        ERP_all.append(ERP_single)
        tankname_all.append(tankname)
    except:
        print( misc_tools.red_text('loading tank {} error'.format(tankname)))



""" do somthing """

tf_3D_GM32 = False
tf_2D_all  = True

if tf_3D_GM32:
    ERP_all = np.dstack(ERP_all)
    np.save('./temp_data/ERP_all_data', ERP_all)

    ERP_all_info = {'data': ERP_all, 'tankname': tankname_all, }
    with open('./temp_data/ERP_all_info', 'wb') as f:
        pickle.dump(ERP_all_info, f)

    """ ==========  ========== """
    with open('./temp_data/ERP_all_info', 'rb') as f:
        ERP_all_info = pickle.load(f)
    ERP_all = ERP_all_info['data']
    tankname = ERP_all_info['tankname']


    # GM32_depth = pd.read_csv('./temp_data/GM32_depth.csv')
    #
    # for chan in range(1,32+1):
    #     pnp.ErpPlot(ERP_all[chan-1,:,:].transpose(), range(ERP_all.shape[1]), depth_linear=GM32_depth['{}'.format(chan)]*0.125 )
    #     plt.suptitle('ERP GM32 channel {}'.format(chan))
    #     plt.savefig('./temp_figs/GM32_ERP_chan_{}.png'.format(chan))
    #     plt.close()

    """ use the log file to determine electrode depth """
    tankdate = np.array([datetime.datetime.strptime(re.match('.*-(\d{6})-.*', tank).group(1), '%y%m%d')  for tank in tankname ])
    with open('./temp_data/GM32_log_info.pkl', 'rb') as f:
        GM32_log = pickle.load(f)
    date_valid = np.sort(np.intersect1d(tankdate, GM32_log['date'] ))

    def arg_select_sort(label, label_ss):
        array_indx = np.zeros(len(label_ss)).astype(int)
        for i, x in enumerate(label_ss):
            array_indx[i] = np.flatnonzero(label==x)
        return array_indx

    ERP_valid  = ERP_all[:,:, arg_select_sort(tankdate, date_valid) ]
    depth_valid = GM32_log['total_depth'][arg_select_sort( GM32_log['date'], date_valid), :]

    for chan in range(1,32+1):
        pnp.ErpPlot(ERP_valid[chan-1,:,:].transpose(), range(ERP_all.shape[1]), depth_linear=depth_valid[:,chan-1] )
        plt.suptitle('ERP GM32 channel {}'.format(chan))
        plt.savefig('./temp_figs/GM32_ERP_chan_{}.png'.format(chan))
        plt.close()

    plt.show()


if tf_2D_all:
    chan_all = np.concatenate([range(1, 1 + erp.shape[0]) for erp in ERP_all])
    tank_all = np.concatenate([[tankname_all[i]] * erp.shape[0] for i, erp in enumerate(ERP_all)])
    erp_2d_all = np.vstack(ERP_all)
    date_all = np.array([re.match('.*-(\d{6})-.*', tank).group(1) for tank in tank_all])

    erp_all_info = pd.DataFrame({'ERP': list(erp_2d_all), 'tank': tank_all, 'chan': chan_all, 'date': date_all})

    if tf_first_stim:
        erp_all_info.to_pickle('./temp_data/erp_all_info_first_stim')
    else:
        erp_all_info.to_pickle('./temp_data/erp_all_info')