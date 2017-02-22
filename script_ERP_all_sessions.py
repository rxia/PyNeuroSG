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


from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge


dir_tdt_tank='/shared/lab/projects/encounter/data/TDT/'
list_name_tanks = os.listdir(dir_tdt_tank)
keyword_tank = '.*GM32.*U16'
list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]
list_name_tanks_0 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is None]
list_name_tanks_1 = [name_tank for name_tank in list_name_tanks if re.match('Dante.*', name_tank) is not None]
list_name_tanks = sorted(list_name_tanks_0) + sorted(list_name_tanks_1)

def GetERP(tankname='GM32.*U16.*161125'):
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv_mask.*', tankname, tf_interactive=False,
                                                              dir_tdt_tank='/shared/lab/projects/encounter/data/TDT/',
                                                              dir_dg = '/shared/lab/projects/analysis/shaobo/data_dg')

    """ Get StimOn time stamps in neo time frame """
    ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')


    """ some settings for saving figures  """
    filename_common = misc_tools.str_common(name_tdt_blocks)
    dir_temp_fig = './temp_figs'


    """ make sure data field exists """
    data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
    blk     = data_load_DLSH.standardize_blk(blk)

    t_plot = [-0.100, 0.500]

    data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*', name_filter='LFPs.*')
    ERP = np.mean(data_neuro['data'], axis=0).transpose()

    # for U16 array in IT
    # pnp.ErpPlot(ERP[0:32, :], data_neuro['ts'], array_layout=layout_GM32)
    # plt.close()
    GM32_depth = pd.read_csv('./temp_data/GM32_depth.csv', sep='\t')

    return ERP


""" ==========  ========== """
ERP_all = []
tankname_all= []
for tankname in list_name_tanks:
    try:
        ERP_single = GetERP(tankname=tankname)
        ERP_all.append(ERP_single)
        tankname_all.append(tankname)
    except:
        print( misc_tools.red_text('loading tank {} error'.format(tankname)))

ERP_all = np.dstack(ERP_all)
np.save('./temp_data/ERP_all_data', ERP_all)

ERP_all_info = {'data': ERP_all, 'tankname': tankname_all, }
with open('./temp_data/ERP_all_info', 'wb') as f:
    pickle.dump(ERP_all_info, f)

""" ==========  ========== """
with open('./temp_data/ERP_all_info', 'rb') as f:
    ERP_all_info = pickle.load(f)
ERP_all = ERP_all_info['data']

GM32_depth = pd.read_csv('./temp_data/GM32_depth.csv')

for chan in range(1,32+1):
    pnp.ErpPlot(ERP_all[chan-1,:,:].transpose(), range(ERP_all.shape[1]), depth_linear=GM32_depth['{}'.format(chan)]*0.125 )
    plt.suptitle('ERP GM32 channel {}'.format(chan))
    plt.savefig('./temp_figs/GM32_ERP_chan_{}.png'.format(chan))
    plt.close()

plt.show()