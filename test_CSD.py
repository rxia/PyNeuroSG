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

ERP, ts = GetERP(tankname='GM32.*U16.*161125')

pnp.ErpPlot(ERP[32:, :], ts)

lfp = ERP[32:, :]

reload(pna); csd = pna.cal_1dCSD(lfp, axis_ch=0); pnp.ErpPlot(pna.SmoothTrace(csd, sk_std=2.0, fs=1.0, axis=0), ts)


""" test gaussian process smoothen: does not work well """
lfp = ERP[32:, range(0,500,10)]
reload(pna)
plt.figure()
plt.subplot(121)
plt.pcolormesh(lfp)
plt.colorbar()
plt.subplot(122)
lfp_sm = pna.GP_ERP_smooth(lfp*10**5)
plt.pcolormesh(lfp_sm)
plt.colorbar()
csd = pna.cal_1dCSD(lfp, axis_ch=0); pnp.ErpPlot(csd)


""" test quadratic smoothing """
lfp = ERP[32:48, np.arange(1,500,1)]
# lfp *= np.expand_dims(np.arange(len(lfp))!=8, axis=1)
reload(pna)
# lfp = lfp/np.std(lfp, axis=1, keepdims=True)
lfp_sm = pna.quad_smooth(lfp, 0,0, 0, 5, 1)
plt.figure()
plt.subplot(121)
plt.pcolormesh(lfp)
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(lfp_sm)
plt.colorbar()
pnp.ErpPlot(lfp); plt.suptitle('original lfp')
pnp.ErpPlot(lfp_sm); plt.suptitle('smoothed lfp')
csd = pna.cal_1dCSD(lfp, axis_ch=0); pnp.ErpPlot(csd); plt.suptitle('csd from original lfp')
csd = pna.cal_1dCSD(lfp_sm, axis_ch=0); pnp.ErpPlot(csd); plt.suptitle('csd from smoothed lfp')
csd = pna.cal_1dCSD(lfp, axis_ch=0); pnp.ErpPlot(pna.SmoothTrace(csd, sk_std=1.5, fs=1.0, axis=0)); plt.suptitle('csd from original lfp, gaussian smoothed')

"""  test using synthesit signal """
x_grid = np.arange(0,32)
target = np.sin(2*np.pi/16*x_grid)
target_noisy = target* (1+np.random.randn(*target.shape)/10)
target_gauss = sp.ndimage.filters.gaussian_filter1d(target_noisy, 1)
# target_noisy[20] = 0
reload(pna)
target_smooth = pna.quad_smooth_d3(target=target_noisy, lambda_d3=1)
_, h_axes = plt.subplots(1,2)
plt.axes(h_axes[0])
plt.plot(x_grid, target)
plt.plot(x_grid, target_noisy)
plt.plot(x_grid, target_gauss)
plt.plot(x_grid, target_smooth)

plt.title('LFP')
plt.legend(['origianl', 'noisy','gaussian_smooth' , 'smoothed'])

plt.axes(h_axes[1])
csd = pna.cal_1dCSD(target, tf_edge=True)
csd_noisy = pna.cal_1dCSD(target_noisy, tf_edge=True)
csd_gauss = pna.cal_1dCSD(target_gauss, tf_edge=True)
csd_gauss_after = sp.ndimage.filters.gaussian_filter1d(csd_noisy, 1)
csd_smooth = pna.cal_1dCSD(target_smooth, tf_edge=True)
plt.plot(x_grid, csd, '-', linewidth=2 )
plt.plot(x_grid, csd_noisy, '-', linewidth=2 )
plt.plot(x_grid, csd_gauss, '-', linewidth=2 )
plt.plot(x_grid, csd_gauss_after, '--', linewidth=2 )
plt.plot(x_grid, csd_smooth, '-', linewidth=2 )
plt.ylim([-np.min(csd)*1.5, -np.max(csd)*1.5])
plt.title('CSD')
plt.legend(['origianl', 'noisy', 'gaussian_smooth', 'smoothed'])




