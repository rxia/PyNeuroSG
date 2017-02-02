"""
This is a show case for the project PyNeuroSG
"""

""" Import modules """
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


plt.ioff()

""" load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
# [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*spot.*', '.*GM32.*U16.*161228.*', tf_interactive=True,)
# [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv.*', '.*GM32.*U16.*161228.*', tf_interactive=True,)
[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*match.*', '.*GM32.*U16.*161125.*', tf_interactive=True,)
# [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('x_.*detection_opto_011317.*', '.*Dexter_.*U16.*170113.*', tf_interactive=True,dir_dg='/Volumes/Labfiles/projects/analysis/ruobing')
# [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv_mask.*', '.*GM32.*U16.*170117.*', tf_interactive=True,)
# [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*matchnot.*', '.*GM32.*U16.*170113.*', tf_interactive=True,)

""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')


""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'


""" make sure data field exists """
data_df = data_load_DLSH.standardize_data_df(data_df, filename_common)
blk     = data_load_DLSH.standardize_blk(blk)



""" ========== Baisc overview of the neural data quality ========== """

""" glance spk waveforms """
pnp.SpkWfPlot(blk.segments[0])
plt.savefig('{}/{}_spk_waveform.png'.format(dir_temp_fig, filename_common))


""" ERP plot """
t_plot = [-0.100, 0.500]

data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*', name_filter='LFPs.*')
ERP = np.mean(data_neuro['data'], axis=0).transpose()
pnp.ErpPlot(ERP, data_neuro['ts'])
plt.savefig('{}/{}_ERP_all.png'.format(dir_temp_fig, filename_common))

try:
    # for GM32 array in V4
    pnp.ErpPlot(ERP[32:,:], data_neuro['ts'])
    plt.savefig('{}/{}_ERP_U16.png'.format(dir_temp_fig, filename_common))

    # for U16 array in IT
    pnp.ErpPlot(ERP[0:32, :], data_neuro['ts'], array_layout=layout_GM32)
    plt.savefig('{}/{}_ERP_GM32.png'.format(dir_temp_fig, filename_common))
except:
    pass

plt.close('all')


""" ========== PSTH or spike raster and average LFP of different experimental conditions ========== """

window_offset = [-0.100, 0.6]
""" spike by condition  """
# align
data_neuro_spk = signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000, chan_filter=range(1,48+1))
# group
data_neuro_spk = signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro_spk)
# plot GM32
pnp.DataNeuroSummaryPlot(signal_align.select_signal(data_neuro_spk, chan_filter=range(1,32+1)), sk_std=0.01, signal_type='auto', suptitle='spk_GM32  {}'.format(filename_common))
plt.savefig('{}/{}_spk_GM32.png'.format(dir_temp_fig, filename_common))
# plot U16
pnp.DataNeuroSummaryPlot(signal_align.select_signal(data_neuro_spk, chan_filter=range(33,48+1)), sk_std=0.01, signal_type='auto', suptitle='spk_U16  {}'.format(filename_common))
plt.savefig('{}/{}_spk_U16.png'.format(dir_temp_fig, filename_common))


""" LFP by condition  """
# align
data_neuro_LFP = signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(1,48+1))
# group
data_neuro_LFP = signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro_LFP)
# plot GM32
pnp.DataNeuroSummaryPlot(signal_align.select_signal(data_neuro_LFP, chan_filter=range(1,32+1)), sk_std=0.01, signal_type='auto', suptitle='LFP_GM32  {}'.format(filename_common))
plt.savefig('{}/{}_LFP_GM32.png'.format(dir_temp_fig, filename_common))
# plot U16
pnp.DataNeuroSummaryPlot(signal_align.select_signal(data_neuro_LFP, chan_filter=range(33,48+1)), sk_std=0.01, signal_type='auto', suptitle='LFP_U16  {}'.format(filename_common))
plt.savefig('{}/{}_LFP_U16.png'.format(dir_temp_fig, filename_common))

plt.close('all')

""" ===== psth plot, spk, by channel ===== """

window_offset = [-0.100, 0.6]
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000, chan_filter=range(1,48+1))
data_neuro=signal_align.neuro_sort(data_df, ['stim_sname'], [], data_neuro)
ts = data_neuro['ts']

for i_neuron in range(len(data_neuro['signal_info'] )):
    name_signal = data_neuro['signal_info'][i_neuron]['name']
    # def functionPlot(x):
    #     pnp.PsthPlot(x, cdtn=np.array(data_df['mask_opacity_int']), ts=ts, sk_std=0.020)
    # pnp.SmartSubplot(data_neuro, functionPlot=functionPlot, dataPlot=data_neuro['data'][:,:,i_neuron])
    pnp.PsthPlotCdtn(data_neuro['data'][:, :, i_neuron], data_df, data_neuro['ts'], cdtn_l_name='mask_opacity_int',
                     cdtn0_name='', cdtn1_name='stim_familiarized', subpanel='auto', sk_std=0.010)
    plt.suptitle('file {},   signal {}'.format(filename_common, name_signal, fontsize=20))
    plt.gcf().set_size_inches(8,4)
    plt.savefig('{}/{} spk PSTH by stim {}.png'.format(dir_temp_fig, filename_common, name_signal))
    plt.close()


# psth plot, LFPs
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='ana.*', name_filter='LFPs.*', spike_bin_rate=1000, chan_filter=range(33,48+1))
data_neuro=signal_align.neuro_sort(data_df, ['stim_sname'], [], data_neuro)
ts = data_neuro['ts']

for i_neuron in range(len(data_neuro['signal_info'] )):
    name_signal = data_neuro['signal_info'][i_neuron]['name']
    pnp.PsthPlotCdtn(data_neuro['data'][:, :, i_neuron], data_df, data_neuro['ts'], cdtn_l_name = 'mask_opacity_int', cdtn0_name = '', cdtn1_name = 'stim_familiarized', subpanel='auto')
    plt.suptitle('file {},   signal {}'.format(filename_common, name_signal, fontsize=20))
    plt.gcf().set_size_inches(8, 4)
    plt.savefig('{}/{} LFPs by stim {}.png'.format(dir_temp_fig, filename_common, name_signal))
    plt.close()


""" LFP spectrum """
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, [-0.100, 1.000], type_filter='ana.*', name_filter='LFPs.*', spike_bin_rate=50)
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro)

[spcg, spcg_t, spcg_f] = pna.ComputeSpectrogram(data_neuro['data'], fs=data_neuro['signal_info'][0][2], t_ini=np.array( data_neuro['ts'][0] ), t_bin=0.1, t_step=None, t_axis=1)
time_baseline = [-0.05, 0.05]
tf_baseline = True
N_sgnl = len(data_neuro['signal_info'])

for i_neuron in range(len(data_neuro['signal_info'] )):
    name_signal = data_neuro['signal_info'][i_neuron]['name']
    functionPlot = lambda x: pnp.SpectrogramPlot(x, spcg_t, spcg_f, tf_log=True, tf_phase=True, f_lim=[0, 100], time_baseline=None,
                                  rate_interp=8)
    pnp.SmartSubplot(data_neuro, functionPlot, spcg[:, :, i_neuron, :])
    plt.suptitle('file {},   LFP power spectrum {}'.format(filename_common, name_signal, fontsize=20))
    plt.savefig('{}/{} LFPs power spectrum by condition {}.png'.format(dir_temp_fig, filename_common, name_signal))
    plt.close()


""" coherence of all pairs """
pnp.SpectrogramAllPairPlot(data_neuro, limit_gap=4, t_bin=0.15, f_lim=[0,100])
plt.savefig('{}/{} LFPs power spectrum and coherence of all pairs.png'.format(dir_temp_fig, filename_common))
plt.close()


""" cohrence plot of one pair """
window_offset = [-0.100, 0.7]
data_neuro_LFP = signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(1,48+1))
# data_neuro_LFP = signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro_LFP)
# data_neuro_LFP = signal_align.neuro_sort(data_df, ['stim_sname'], [], data_neuro_LFP)
data_neuro_LFP = signal_align.neuro_sort(data_df, [''], [], data_neuro_LFP)
# list_ch0 = [1,3,5,10,11,14,16,19,21,28]
# list_ch1 = [33,40,48]
# list_ch0 = [1,5,11,16,19,28]
# list_ch1 = [3,10,14,21]
# list_ch0 = [1,5,11,16,19,28]
# list_ch1 = [33,37,41,45,48]
list_ch0 = [5]
list_ch1 = [37]
# list_ch0 = [2,5,11,16,21 ,28]
# list_ch1 = [34,36,40,43,45,48]

for ch0 in list_ch0:
    for ch1 in list_ch1:
        def functionPlot(x):
            [cohg, spcg_t, spcg_f] = pna.ComputeCoherogram(x, data1=None, tf_phase=True, tf_vs_shuffle=False, fs=data_neuro_LFP['signal_info'][0][2],
                                                           t_ini=np.array( data_neuro_LFP['ts'][0] ), t_bin=0.2, t_step=None, f_lim=[0, 70])
            pnp.SpectrogramPlot(cohg, spcg_t, spcg_f, tf_log=False, f_lim=[0, 70], tf_phase=True, time_baseline=None, rate_interp=8, c_lim_style='from_zero', name_cmap='inferno', tf_colorbar=False)
            del(cohg)
        pnp.SmartSubplot(data_neuro_LFP, functionPlot, data_neuro_LFP['data'][:,:,[ch0-1,ch1-1]], suptitle='coherence {}_{},    {}'.format(ch0, ch1, filename_common), tf_colorbar=True)
        plt.gcf().set_size_inches(6,6)
        plt.savefig('{}/{} LFPs coherence by condition {}-{}.png'.format(dir_temp_fig, filename_common, ch0, ch1))
        plt.close()


""" spkike-field coupling of one pair """
window_offset = [-0.100, 0.7]
data_neuro_LFP = signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(1,48+1))
fs = data_neuro_LFP['signal_info'][0][2]
data_neuro_spk = signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='spi.*', name_filter='.*Code[1-9]$', chan_filter=range(1,48+1), spike_bin_rate=fs)
# data_neuro_LFP = signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro_LFP)
# data_neuro_LFP = signal_align.neuro_sort(data_df, ['stim_sname'], [], data_neuro_LFP)
# data_neuro_LFP = signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro_LFP)
# data_neuro_spk = signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro_spk)
data_neuro_LFP = signal_align.neuro_sort(data_df, [''], [], data_neuro_LFP)
data_neuro_spk = signal_align.neuro_sort(data_df, [''], [], data_neuro_spk)
# list_ch0 = [1,3,5,10,11,14,16,19,21,28]
# list_ch1 = [33,40,48]
# list_ch0 = [1,5,11,16,19,28]
# list_ch1 = [3,10,14,21]
# list_ch0 = [1,5,11,16,19,28]
# list_ch1 = [33,37,41,45,48]
# list_ch0 = [2,5,11,16,28]
# list_ch1 = [33,36,42,47]
# list_ch0 = [33,36,42,47]
# list_ch1 = [28]
# list_ch0 = [1,5,11,16,19,28]
# list_ch1 = [34,36,40,43,45,48]
list_ch0 = [37]
list_ch1 = [5]
measure = 'PPC'
for ch0 in list_ch0:
    for ch1 in list_ch1:
        def functionPlot(x):
            if measure == 'coherence':
                [cohg, spcg_t, spcg_f] = pna.ComputeCoherogram(x, data1=None, tf_phase=True, fs=fs, t_ini=np.array( data_neuro_LFP['ts'][0] ),
                                                               t_bin=0.2, t_step=None, f_lim=[0, 70])
            elif measure == 'PLV':
                [cohg, spcg_t, spcg_f] = pna.ComputeSpkTrnFieldCoupling(x, data_spk=None, tf_phase=True, tf_vs_shuffle=False, fs=fs, measure=measure,
                                                           t_ini=np.array(data_neuro_LFP['ts'][0]), t_bin=0.2,
                                                           t_step=None, f_lim=[0, 70])
            elif measure == 'PPC':
                [cohg, spcg_t, spcg_f] = pna.ComputeSpkTrnFieldCoupling(x, data_spk=None, tf_phase=True, tf_vs_shuffle=False, fs=fs, measure=measure,
                                                            t_ini=np.array(data_neuro_LFP['ts'][0]), t_bin=0.2,
                                                            t_step=None, f_lim=[0, 70])
            pnp.SpectrogramPlot(cohg, spcg_t, spcg_f, tf_log=False, f_lim=[0, 70], tf_phase=True, time_baseline=None, rate_interp=8,
                                c_lim_style='from_zero', name_cmap='viridis', tf_colorbar=False)
            del(cohg)
        data_cur = np.dstack([signal_align.select_signal(data_neuro_LFP, chan_filter=[ch0])['data'][:, :, 0],
                              signal_align.select_signal(data_neuro_spk, chan_filter=[ch1])['data'][:, :, -1]])
        pnp.SmartSubplot(data_neuro_LFP, functionPlot, data_cur, suptitle='spike LFP phase consistency  spk {}-LFP {},    {}'.format(ch1, ch0, filename_common), tf_colorbar=True)
        plt.gcf().set_size_inches(6, 6)
        plt.savefig('{}/{} spike LFP phase consistency, spk {}-LFP {}.png'.format(dir_temp_fig, filename_common, ch1, ch0))
        plt.close()




""" decoding using spkikes """
from sklearn import svm
from sklearn import cross_validation
from sklearn.preprocessing import normalize


data_neuro = signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='spi.*', name_filter='.*Code[1-9]$', chan_filter=range(33,48+1), spike_bin_rate=50)
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized', 'mask_opacity_int'], [], data_neuro); pnp.NeuroPlot(data_neuro)

N_window_smooth = 7
data_neuro['data'] = sp.signal.convolve(data_neuro['data'], np.ones([1,N_window_smooth,1]), 'same')

X_normalize = ( data_neuro['data'] - np.mean( data_neuro['data'] , axis=(0,1), keepdims=True ) ) /np.std( data_neuro['data'] , axis=(0,1), keepdims=True )

C=1
clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=C)
time_tic = time.time()
[N_tr, N_ts, N_sg] = data_neuro['data'].shape
N_cd = len(data_neuro['cdtn'])
clf_score = np.zeros([N_cd, N_ts])
clf_score_std = np.zeros([N_cd, N_ts])
for i in range(N_cd):
    cdtn = data_neuro['cdtn'][i]
    indx = np.array(data_neuro['cdtn_indx'][cdtn])
    print(cdtn)
    for t in range(N_ts):
        cfl_scores = cross_validation.cross_val_score(clf, X_normalize[indx, t, :], data_df['stim_names'][indx].tolist(), cv=5)
        clf_score[i, t] = np.mean(cfl_scores)
        clf_score_std[i, t] = np.std(cfl_scores)
        # clf_score[i, t] = np.mean(cross_validation.cross_val_score(clf, normalize(data_neuro['data'][indx, t, :]), data_df['mask_names'][indx].tolist(), cv=5))
print(time.time()-time_tic)

[fig, ax] = plt.subplots(1,2,figsize=(8,4), sharex=True, sharey=True)
for i in range(N_cd):
    if i<3:
        plt.axes(ax[0])
    else:
        plt.axes(ax[1])
    plt.fill_between(data_neuro['ts'], clf_score[i, :] - clf_score[i, :] / 10, clf_score[i, :] + clf_score[i, :] / 10,
                     alpha=0.5, color=pnp.gen_distinct_colors(3, alpha=0.3)[i%3])
    plt.plot(data_neuro['ts'],clf_score[i,:], color=pnp.gen_distinct_colors(3)[i%3], linewidth=2)
    # plt.subplot(2,3,i+1)
    # plt.fill_between(data_neuro['ts'], clf_score[i,:]-clf_score[i,:]/5, clf_score[i,:]+clf_score[i,:]/5, alpha=0.5)
    # plt.plot(data_neuro['ts'],clf_score[i,:])
    # plt.title(data_neuro['cdtn'][i])
    # plt.ylim([0,1])
plt.ylim([0,1])
plt.savefig('{}/{} decoding IT.png'.format(dir_temp_fig, filename_common))
plt.show()



# if rf mapping
if False:
    # align, RF plot
    import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.020, 0.200], type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=100); print(time.time()-t)
    # import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.0200, 0.0700], type_filter='ana.*', name_filter='LFPs .*$', spike_bin_rate=1000); print(time.time()-t)
    # group
    import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.neuro_sort(data_df, ['stim_pos_x','stim_pos_y'], [], data_neuro); print(time.time()-t)
    # plot
    import PyNeuroPlot as pnp; reload(pnp); t=time.time(); pnp.RfPlot(data_neuro, indx_sgnl=0, x_scale=0.2); print(time.time()-t)
    for i in range(len(data_neuro['signal_info'])):
        pnp.RfPlot(data_neuro, indx_sgnl=i, x_scale=0.2, y_scale=100)
        try:
            plt.savefig('{}/{}_RF_{}.png'.format(dir_temp_fig, filename_common, data_neuro['signal_info'][i]['name']))
        except:
            plt.savefig('./temp_figs/RF_plot_' + misc_tools.get_time_string() + '.png')
        plt.close()
