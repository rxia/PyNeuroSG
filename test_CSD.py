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

# ----- modules of the project PyNeuroSG -----
import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc

# ----- modules for the data location and organization in Sheinberg lab -----
import data_load_DLSH       # package specific for DLSH lab data
from GM32_layout import layout_GM32



""" -----  test using synthesit signal ----- """
x_grid = np.arange(0,16)
target = np.sin(2*np.pi/8*x_grid)
target_noisy = target* (1+np.random.randn(*target.shape)/3)
target_noisy[6] = 0
target_gauss = sp.ndimage.filters.gaussian_filter1d(target_noisy, 1)
reload(pna)
# target_smooth = pna.quad_smooth_d3(target=target_noisy, lambda_der=0.5)
target_smooth, csd_smooth = pna.quad_smooth_der(target=target_noisy, lambda_der=1, degree_der=3, return_CSD=True)

_, h_axes = plt.subplots(3,2)
plt.axes(h_axes[0,0])
plt.title('LFP')
plt.plot(x_grid, target)
plt.plot(x_grid, target_noisy)
plt.axes(h_axes[1,0])
plt.plot(x_grid, target)
plt.plot(x_grid, target_gauss)
plt.axes(h_axes[2,0])
plt.plot(x_grid, target)
plt.plot(x_grid, target_smooth)


# plt.legend(['origianl', 'noisy','gaussian_smooth' , 'smoothed'])


csd = pna.cal_1dCSD(target, tf_edge=True)
csd_noisy = pna.cal_1dCSD(target_noisy, tf_edge=True)
csd_gauss = pna.cal_1dCSD(target_gauss, tf_edge=True)
# csd_gauss_after = sp.ndimage.filters.gaussian_filter1d(csd_noisy, 1)
# csd_smooth = pna.cal_1dCSD(target_smooth, tf_edge=True)


plt.axes(h_axes[0,1])
plt.title('CSD')
plt.plot(x_grid, csd, '-', linewidth=2 )
plt.plot(x_grid, csd_noisy, '-', linewidth=2 )
plt.legend(['origianl', 'native estimator'])
plt.axes(h_axes[1,1])
plt.plot(x_grid, csd, '-', linewidth=2 )
plt.plot(x_grid, csd_gauss, '-', linewidth=2 )
# plt.plot(x_grid, csd_gauss_after, '--', linewidth=2 )
plt.legend(['origianl', 'gaussian smoothed'])
plt.axes(h_axes[2,1])
plt.plot(x_grid, csd, '-', linewidth=2 )
plt.plot(x_grid, csd_smooth, '-', linewidth=2 )
plt.legend(['origianl', 'derevative smoothed'])



dir_tdt_tank = '/shared/homes/sguan/neuro_data/tdt_tank'
dir_dg='/shared/homes/sguan/neuro_data/stim_dg'

def GetERP(tankname='GM32.*U16.*161125'):
    [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv_mask.*', tankname, tf_interactive=False,
                                                               dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg)

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

    return ERP, data_neuro['ts']

ERP, ts = GetERP(tankname='GM32.*U16.*161206')

pnp.ErpPlot(ERP[32:, :], ts)

lfp = ERP[32:, :]

reload(pna); csd = pna.cal_1dCSD(lfp, axis_ch=0); pnp.ErpPlot(pna.SmoothTrace(csd, sk_std=2.0, fs=1.0, axis=0), ts)



""" test quadratic smoothing """
lfp = ERP[32:48, :]
# lfp *= np.expand_dims(np.arange(len(lfp))!=8, axis=1)
reload(pna)
# lfp = lfp/np.std(lfp, axis=1, keepdims=True)
lambda_dev = np.ones(16)
lambda_dev[2] = 0
lfp_sm_gau = pna.lfp_cross_chan_smooth(lfp, method='gaussian', sigma_chan=1.5, sigma_t=0.5)
lfp_sm_der = pna.lfp_cross_chan_smooth(lfp, method='der', lambda_dev=lambda_dev, lambda_der=1, sigma_t=0.5)

csd_nai        = pna.cal_1dCSD(lfp, axis_ch=0, tf_edge=True)
csd_gau_before = pna.cal_1dCSD(lfp_sm_gau, axis_ch=0, tf_edge=True)
csd_gau_after  = pna.lfp_cross_chan_smooth(csd_nai, method='gaussian', sigma_chan=0.5, sigma_t=0)
csd_der        = pna.cal_1dCSD(lfp_sm_der, axis_ch=0, tf_edge=True)

pnp.ErpPlot(lfp, ts); plt.suptitle('original lfp')

pnp.ErpPlot(csd_nai,        ts, title='CSD naive'); plt.suptitle('csd from original lfp')
pnp.ErpPlot(csd_gau_before, ts, title='CSD'); plt.suptitle('CSD, gaussian smoothed on lfp')
pnp.ErpPlot(csd_gau_after,  ts, title='CSD'); plt.suptitle('CSD, gaussian smoothed on csd')
pnp.ErpPlot(csd_der,        ts, title='CSD'); plt.suptitle('CSD, derivative smoothed')







