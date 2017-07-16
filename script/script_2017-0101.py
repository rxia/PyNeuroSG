

import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
pd.set_option('display.max_columns', None)
import dg2df
import neo
from neo.core import (Block, Segment, ChannelIndex, AnalogSignal, Unit)
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ioff()
import standardize_TDT_blk
from standardize_TDT_blk import select_obj_by_attr
import quantities as pq
import signal_align
import re
import PyNeuroPlot as pnp
import PyNeuroAna as pna
import time
from scipy import signal
from scipy.signal import spectral
from PyNeuroPlot import center2edge
import misc_tools
import data_load_DLSH
from GM32_layout import layout_GM32


""" load data: (1) neural data: TDT blocks -> neo format; (2)behaverial data: stim dg -> pandas DataFrame """
# [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*spot.*', '.*GM32.*U16.*161228.*', tf_interactive=True,)
# [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*srv.*', '.*GM32.*U16.*161228.*', tf_interactive=True,)
# [blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('d_.*match.*', '.*GM32.*U16.*161125.*', tf_interactive=True,)
[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data('x_.*detection_opto_011317.*', '.*Dexter_.*U16.*170113.*', tf_interactive=True,dir_dg='/Volumes/Labfiles/projects/analysis/ruobing')

""" Get StimOn time stamps in neo time frame """
ts_StimOn = data_load_DLSH.get_ts_align(blk, data_df, dg_tos_align='stimon')


""" some settings for saving figures  """
filename_common = misc_tools.str_common(name_tdt_blocks)
dir_temp_fig = './temp_figs'


""" make sure data field exists """
data_df['']=['']*len(data_df)    # empty column, make some default condition easy

if re.match('.*matchnot.*', filename_common) is not None:
    data_df['stim_names'] = data_df.SampleFilename
    data_df['stim_familiarized'] = data_df.SampleFamiliarized
    data_df['mask_opacity'] = data_df['MaskOpacity']
if re.match('.*matchnot.*', filename_common) is not None or re.match('.*masked.*', filename_common) is not None:
    data_df['mask_opacity_int'] = np.round(data_df['mask_opacity']*100).astype(int)



""" ==================== """
plt.ioff()

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
    data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(33,48+1))
    # data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='spiketrains.*', name_filter='.*Code[1-9].*', chan_filter=range(33,48+1), spike_bin_rate=100)
    ERP = np.mean(data_neuro['data'], axis=0).transpose()
    pnp.ErpPlot(ERP, data_neuro['ts'])
    plt.savefig('{}/{}_ERP_U16.png'.format(dir_temp_fig, filename_common))


    data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, t_plot, type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(1,32+1))
    ERP = np.mean(data_neuro['data'], axis=0).transpose()
    [h_fig, h_axes] = pnp.create_array_layout_subplots(layout_GM32)
    plt.tight_layout()
    for ch in range(32):
        plt.axes(h_axes[layout_GM32[ch+1]])
        plt.plot(data_neuro['ts'], ERP[ch,:]*10**6, linewidth=2)
        plt.xlabel(t_plot)
        plt.title( 'Channel {}'.format(ch+1) )
    plt.savefig('{}/{}_ERP_GM32.png'.format(dir_temp_fig, filename_common))
except:
    pass


""" ==================== """


window_offset = [-0.100, 0.6]
""" GM32 spike by condition  """
# align
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000, chan_filter=range(1,32+1))
# group
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro)
# plot
pnp.NeuroPlot(data_neuro, sk_std=0.010, tf_legend=True, tf_seperate_window=False)
plt.suptitle('spk_GM32    {}'.format(filename_common), fontsize=20)
plt.savefig('{}/{}_spk_GM32.png'.format(dir_temp_fig, filename_common))

""" U16 spike by condition  """
# align
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000, chan_filter=range(33,48+1))
# group
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro)
# plot
pnp.NeuroPlot(data_neuro, sk_std=0.010,tf_legend=True, tf_seperate_window=False)
plt.suptitle('spk_U16    {}'.format(filename_common), fontsize=20)
plt.savefig('{}/{}_spk_U16.png'.format(dir_temp_fig, filename_common))
# import signal_align; reload(signal_align); t=time.time(); data_neuro=signal_align.blk_align_to_evt(blk, blk_StimOn, [-0.100, 1.000], type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(1,48+1), spike_bin_rate=1000); print(time.time()-t)

""" GM32 LFP by condition  """
# align
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(1,32+1))
# group
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro)
# plot
pnp.NeuroPlot(data_neuro, tf_legend=True, tf_seperate_window=False)
plt.suptitle('LFP_GM32    {}'.format(filename_common), fontsize=20)
plt.savefig('{}/{}_LFP_GM32.png'.format(dir_temp_fig, filename_common))

""" U16 LFP by condition  """
# align
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='ana.*', name_filter='LFPs.*', chan_filter=range(33,48+1))
# group
data_neuro=signal_align.neuro_sort(data_df, ['stim_familiarized','mask_opacity_int'], [], data_neuro)
# plot
pnp.NeuroPlot(data_neuro, tf_legend=True, tf_seperate_window=False)
plt.suptitle('LFP_U16    {}'.format(filename_common), fontsize=20)
plt.savefig('{}/{}_LFP_U16.png'.format(dir_temp_fig, filename_common))



""" ===== psth plot, spk ===== """

window_offset = [-0.100, 0.6]
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='spiketrains.*', name_filter='.*Code[1-9]$', spike_bin_rate=1000, chan_filter=range(1,48+1))
data2D = data_neuro['data'][:,:,0]
ts = data_neuro['ts']

for i_neuron in range(len(data_neuro['signal_info'] )):
    name_signal = data_neuro['signal_info'][i_neuron]['name']
    pnp.PsthPlotCdtn(data_neuro['data'][:, :, i_neuron], data_df, data_neuro['ts'], cdtn_l_name = 'mask_opacity_int', cdtn0_name = 'stim_names', cdtn1_name = '', subpanel='auto', sk_std=0.020, )
    plt.suptitle('file {},   signal {}'.format(filename_common, name_signal, fontsize=20))
    plt.savefig('{}/{} spk PSTH by stim {}.png'.format(dir_temp_fig, filename_common, name_signal))
    plt.close()

# psth plot, LFPs
data_neuro=signal_align.blk_align_to_evt(blk, ts_StimOn, window_offset, type_filter='ana.*', name_filter='LFPs.*', spike_bin_rate=1000, chan_filter=range(1,48+1))
data2D = data_neuro['data'][:,:,0]
ts = data_neuro['ts']

for i_neuron in range(len(data_neuro['signal_info'] )):
    name_signal = data_neuro['signal_info'][i_neuron]['name']
    pnp.PsthPlotCdtn(data_neuro['data'][:, :, i_neuron], data_df, data_neuro['ts'], cdtn_l_name = 'mask_opacity_int', cdtn0_name = 'stim_names', cdtn1_name = '', subpanel='auto')
    plt.suptitle('file {},   signal {}'.format(filename_common, name_signal, fontsize=20))
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
    functionPlot = lambda x: pnp.SpectrogramPlot(x, spcg_t, spcg_f, tf_log=True, f_lim=[0, 100], time_baseline=[-0.05, 0.05],
                                  rate_interp=8)
    pnp.SmartSubplot(data_neuro, functionPlot, spcg[:, :, i_neuron, :])
    plt.suptitle('file {},   LFP power spectrum {}'.format(filename_common, name_signal, fontsize=20))
    plt.savefig('{}/{} LFPs power spectrum by condition {}.png'.format(dir_temp_fig, filename_common, name_signal))
    plt.close()


""" LFP coherence """
[spcg_f,spcg_t,spcg] = signal.spectrogram(data_neuro['data'], data_neuro['data'], window=signal.hann(128), nperseg=128, nfft=256,fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)



""" LFP coherence, all pairs """
fig, axes2d = plt.subplots(nrows=12, ncols=12, sharex=True, sharey=True, figsize=(16,10))
plt.tight_layout()
fig.subplots_adjust(hspace=0.05, wspace=0.05)
chan_gap = 4
for ch_x in range(0,48,chan_gap):
    for ch_y in range(0,48,chan_gap):
        [spcg_f, spcg_t, spcg_xy] = spectral._spectral_helper(data_neuro['data'][:, :, ch_x],
                                                              data_neuro['data'][:, :, ch_y], window=signal.hann(128),
                                                              nperseg=128, nfft=256, fs=data_neuro['signal_info'][0][2],
                                                              axis=1, noverlap=96)
        spcg_t = np.array(spcg_t) + np.array(data_neuro['ts'][0])
        spcg_xy_ave = np.mean(spcg_xy, axis=0)
        plt.sca(axes2d[ch_x/chan_gap, ch_y/chan_gap])
        if ch_x is 0:
            axes2d[ch_x/chan_gap, ch_y/chan_gap].set_title('ch{}'.format(ch_y+1))
        if ch_y is 0:
            axes2d[ch_x/chan_gap, ch_y/chan_gap].set_ylabel('ch{}'.format(ch_x+1))

        # plt.subplot2grid([16, 16], [ch_x, ch_y])

        if ch_x == ch_y:
            spcg_xy_ave = np.abs(spcg_xy_ave)
            plt.pcolormesh(center2edge(spcg_t), center2edge(spcg_f), 10*np.log10(spcg_xy_ave), cmap=plt.get_cmap('inferno'))
            # plt.title('ch{}'.format(ch_x + 1))
        else:
            [_, _, spcg_xx] = spectral._spectral_helper(data_neuro['data'][:, :, ch_x], data_neuro['data'][:, :, ch_x],
                                                        window=signal.hann(128), nperseg=128, nfft=256,
                                                        fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
            [_, _, spcg_yy] = spectral._spectral_helper(data_neuro['data'][:, :, ch_y], data_neuro['data'][:, :, ch_y],
                                                        window=signal.hann(128), nperseg=128, nfft=256,
                                                        fs=data_neuro['signal_info'][0][2], axis=1, noverlap=96)
            spcg_xx_ave = np.mean(spcg_xx, axis=0)
            spcg_yy_ave = np.mean(spcg_yy, axis=0)
            spcg_coh = np.abs(np.abs(spcg_xy_ave) ** 2 / (spcg_xx_ave * spcg_yy_ave))

            plt.pcolormesh( center2edge(spcg_t), center2edge(spcg_f), spcg_coh, cmap=plt.get_cmap('viridis'))
            plt.clim(0, 0.5)
            # plt.title( 'ch{} vs ch {}'.format(ch_x+1, ch_y+1) )
        plt.ylim(0, 80)



# psth by six conditions
if False:
    cdtn0_name = 'stim_familiarized'
    cdtn1_name = 'mask_opacity_int'
    cdtn_l_name = 'stim_names'
    N_cdtn0 = len(data_df[cdtn0_name].unique())
    N_cdtn1 = len(data_df[cdtn1_name].unique())

    for i_neuron in range(len(data_neuro['signal_info'] )):
        data2D = data_neuro['data'][:, :, i_neuron]
        ts = data_neuro['ts']
        [h_fig, h_ax] = plt.subplots( N_cdtn0, N_cdtn1, figsize=[12,9], sharex=True, sharey=True )
        for i_cdtn0, cdtn0 in enumerate(sorted(data_df[cdtn0_name].unique())) :
            for i_cdtn1, cdtn1 in enumerate(sorted(data_df[cdtn1_name].unique())):
                plt.axes(h_ax[i_cdtn0,i_cdtn1])
                pnp.PsthPlot(data2D, ts, data_df[cdtn_l_name],
                             np.logical_and(data_df[cdtn0_name] == cdtn0, data_df[cdtn1_name] == cdtn1),
                             tf_legend=False, sk_std=0.020, subpanel='raster')
                plt.title( [cdtn0, cdtn1] )

        plt.suptitle(data_neuro['signal_info'][i_neuron]['name'])
        try:
            plt.savefig('{}/{} PSTH {}.png'.format(dir_temp_fig, filename_common, data_neuro['signal_info'][i_neuron]['name']))
        except:
            plt.savefig('./temp_figs/' + misc_tools.get_time_string() + '.png' )
        plt.close(h_fig)


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
