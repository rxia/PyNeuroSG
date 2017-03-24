""" characterize channel using ERP """

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


def GetERP(tankname='GM32.*U16.*161125'):
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
    ERP = np.mean(data_neuro['data'], axis=0).transpose()

    return ERP

ERP = GetERP(tankname='GM32.*U16.*161125')


""" embed ERP of one day """
reload(pna)
result_embedding =  pna.LowDimEmbedding(ERP, type='MDS')
plt.plot(result_embedding[:,0], result_embedding[:,1],'o')
scale_ERP = np.nanmax(np.abs(ERP))
scale_embedding = np.max(result_embedding, axis=0)-  np.min(result_embedding, axis=0)
for i, xy in enumerate(result_embedding):
    plt.annotate('{}'.format(i+1), xy=xy)
    plt.plot( result_embedding[i,0]+ np.arange(ERP.shape[1])*scale_embedding[0]/ERP.shape[1]/20,
              result_embedding[i,1]+ERP[i,:]/scale_ERP*scale_embedding[1]/20)
plt.savefig('./temp_figs/ERP_embedding.png')




""" embed ERP of all recording sessions (3D stored) """

with open('./temp_data/ERP_all_info', 'rb') as f:
    ERP_all_info = pickle.load(f)
ERP_all = ERP_all_info['data']
tankname = ERP_all_info['tankname']

ERP = np.vstack( [ERP_all[:,:,i] for i in range(ERP_all.shape[2]) if i%1==0] )
result_embedding =  pna.LowDimEmbedding(ERP, type='PCA')

chan_tank = zip(np.arange(len(result_embedding))%32 +1, sum([[tank]*32 for tank in tankname], []))
for chan in range(1,32+1):
    plt.figure(figsize=[12,12])
    pnp.EmbedTracePlot(result_embedding, traces=ERP, labels_interactive=chan_tank,
                                highlight=np.array(zip(*chan_tank)[0])==chan,
                                color=pnp.gen_distinct_colors(n=len(result_embedding), style='continuous') )
    plt.title('ERP_embedding_all_sessions_chan_{}'.format(chan))
    plt.savefig('./temp_figs/ERP_embedding_all_sessions_chan_{}.png'.format(chan))
    plt.close()
plt.savefig('./temp_figs/ERP_embedding_all_sessions.png')


""" embed ERP using all recording sessions (2D stored) """
erp_df_full = pd.read_pickle('./temp_data/erp_all_info')

erp_df = erp_df_full[erp_df_full['chan']>=33]

ERP= np.array(erp_df['ERP'].tolist())

reload(pna)
# result_embedding =  pna.LowDimEmbedding(ERP, type='PCA', para=2)
result_embedding =  pna.LowDimEmbedding(ERP, type='Isomap', para=20)


chan = erp_df['chan'].tolist()
date = erp_df['date'].tolist()
chan_date = zip(erp_df['chan'].tolist(), erp_df['date'].tolist() )
dict_color_chan = pnp.gen_distinct_colors(n=16+1, style='continuous')
dict_color_date = pnp.gen_distinct_colors(n=len(np.unique(date))+1, style='continuous')
chan_color = [dict_color_chan[i-32] for i in chan]
date_color = [dict_color_date[i/16] for i,_ in enumerate(date)]


for date_i in np.unique(date):
    plt.figure(figsize=[12,12])
    pnp.EmbedTracePlot(result_embedding, traces=ERP, labels_interactive=chan_date,
                   highlight=(np.array(date) == date_i),
                   color=chan_color)
    plt.title('U16_ERP_embedding_all_sessions_date_{}'.format(date_i))
    plt.savefig('./temp_figs/ERP_U16_embedding_all_sessions_date_{}.png'.format(date_i))
    plt.close()

pnp.EmbedTracePlot(result_embedding, traces=ERP, labels_interactive=chan_date,
                   highlight=None,
                   color=date_color)