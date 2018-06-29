import os     # for getting file paths
import neo    # for reading neural data (TDT format)
import dg2df  # for reading behavioral data
import pandas as pd
import re     # use regular expression to find file names
import numpy as np
import scipy as sp
from standardize_TDT_blk import select_obj_by_attr
import data_load_DLSH
import matplotlib as mpl
import matplotlib.pyplot as plt
import PyNeuroAna as pna
mpl.style.use('ggplot')

dir_tdt_tank = '/shared/lab/projects/encounter/data/TDT/'
keyword_tank = 'Murphy-170526-110333'
keyword_blk = 'retract'
speed_retract = 10  # um/sec
binsize_dist = 100      # bin for distance
ch_index = 1
tf_reverse_dist = True
tf_spectragram = True
tf_spectragram_interactive = True
total_distance = None
range_ch=None
range_ch_pos_rel=None
t_gauss_sigma_spcg = 5

path_save_fig = './temp_figs'

def load_blk(keyword_blk=keyword_blk, keyword_tank=keyword_tank, dir_tdt_tank=dir_tdt_tank,
             tf_verbose=True, tf_interactive=False, sortname=''):
    [name_tdt_blocks, path_tdt_tank] = \
        data_load_DLSH.get_file_name(keyword_blk, keyword_tank, mode='tdt',
                                     tf_interactive=tf_interactive, dir_tdt_tank=dir_tdt_tank)

    name_datafiles = name_tdt_blocks
    if True:
        print('')
        print('the data files to be loaded are: {}'.format(name_datafiles))

    """ ----- load neural data ----- """
    blk = neo.core.Block()  # block object containing multiple segments, each represents data form one file
    reader = neo.io.TdtIO(dirname=path_tdt_tank)  # reader for loading data
    for name_tdt_block in name_tdt_blocks:  # for every data file
        if tf_verbose:
            print('loading TDT block: {}'.format(name_tdt_block))
        seg = reader.read_segment(blockname=name_tdt_block,
                                  sortname=sortname)  # read one TDT file as a segment of block
        blk.segments.append(seg)  # append to blk object
    if tf_verbose:
        print('finish loading tdt blocks')

    blk = data_load_DLSH.standardize_blk(blk)
    return blk.segments[0]

def plot_electrode_depth_profile(keyword_blk=keyword_blk, keyword_tank=keyword_tank, dir_tdt_tank=dir_tdt_tank,
                                 ch_index = 1, range_ch= None,
                                 speed_retract=speed_retract, binsize_dist=binsize_dist, range_ch_pos_rel=None,
                                 tf_reverse_dist=tf_reverse_dist, total_distance=None, t_gauss_sigma_spcg = t_gauss_sigma_spcg,
                                 tf_spectragram=tf_spectragram, tf_spectragram_interactive=tf_spectragram_interactive):
    """ generate plot of depth profile of spiking pattern and lfp spectrogram

    assuming data is collected when electrode is retracted at a constant speed """

    """ ----- load data ----- """
    seg = load_blk(keyword_blk=keyword_blk, keyword_tank=keyword_tank, dir_tdt_tank=dir_tdt_tank)

    """ ----- use time and speed to infer depth ----- """
    t_range = np.array([seg.t_start, seg.t_stop])
    if total_distance is not None:
        t_range[1] = total_distance/speed_retract
    dist_range = t_range*speed_retract

    """ ----- get spiking pattern ----- """
    spk_time = np.sort(np.concatenate([np.array(spktrn.times) for spktrn in seg.spiketrains if spktrn.annotations['channel_index']==ch_index]))
    spk_dist = spk_time * speed_retract
    spk_bin, dist_bin_edge = np.histogram(spk_dist, np.arange(dist_range[0], dist_range[1], binsize_dist))
    dist_bin_center = (dist_bin_edge[:-1] + dist_bin_edge[1:]) / 2
    if range_ch is not None:
        list_spk_bin = []
        for ch_cur in range_ch:
            try:
                spk_time_cur = np.sort(np.concatenate([np.array(spktrn.times) for spktrn in seg.spiketrains if
                                               spktrn.annotations['channel_index'] == ch_cur]))
            except:
                spk_time_cur = np.array([])
            spk_dist_cur = spk_time_cur * speed_retract
            spk_bin_cur, _ = np.histogram(spk_dist_cur, np.arange(dist_range[0], dist_range[1], binsize_dist))
            list_spk_bin.append(spk_bin_cur)
        spk_bin_array = np.vstack(list_spk_bin)

    """ ----- compute spectrogram ----- """
    if tf_spectragram:
        scale_factor = 10**6
        lfp_sig = [anasig for anasig in seg.analogsignals if re.search('LFPs.*', anasig.name) and anasig.annotations['channel_index']==ch_index][0]
        lfp_trace = np.ravel(lfp_sig) * scale_factor
        lfp_fs = np.array(lfp_sig.sampling_rate)
        lfp_ts = range(len(lfp_trace))/lfp_fs + np.array(lfp_sig.t_start)
        if total_distance is not None:
            lfp_ts = lfp_ts[lfp_ts<t_range[1]]
            lfp_trace = lfp_trace[:len(lfp_ts)]
        spcg, spcg_t, spcg_f = pna.ComputeSpectrogram(data=np.expand_dims(lfp_trace, axis=0),
                               fs=lfp_fs, t_ini=lfp_ts[0], t_bin=1.0, t_step=1.0, t_axis=1, f_lim=[0,100])
        spcg = spcg[0,:,:]
        if t_gauss_sigma_spcg is not None:
            spcg = sp.ndimage.filters.gaussian_filter1d(spcg, sigma=t_gauss_sigma_spcg, axis=1)
        spcg_dist = spcg_t * speed_retract

    """ ----- direction of distance axis ----- """
    if tf_reverse_dist:   # top to bottom
        dist_bin = np.flip(dist_bin_center, axis=0)
        if tf_spectragram:
            dist_spcg = np.flip(spcg_dist, axis=0)
    else:                 # bottom to top
        dist_bin = dist_bin_center
        if tf_spectragram:
            dist_spcg = spcg_dist

    """ ----- plot ----- """
    if tf_spectragram:
        h_fig, h_ax = plt.subplots(2, 1, figsize=[8, 8], sharex=True, squeeze=False)
    else:
        h_fig, h_ax = plt.subplots(1, 1, figsize=[8, 4], sharex=True, squeeze=False)

    """ upper panel: spiking pattern """
    plt.axes(h_ax[0,0])
    if range_ch is not None:
        list_color = plt.cm.rainbow(np.linspace(0,1,len(range_ch)))
        for i, ch_cur in enumerate(range_ch):
            if range_ch_pos_rel is None:
                dist_shift = 0
            else:
                dist_shift = range_ch_pos_rel[i]
            plt.plot(dist_bin+dist_shift, spk_bin_array[i,:], color=list_color[i], linewidth=1, alpha=0.5)
    plt.plot(dist_bin, spk_bin, color='k', linewidth=2)
    plt.xlabel('distance, um')
    plt.ylabel('spk count in {} um window'.format(binsize_dist))
    plt.title('spiking pattern')

    """ lower panel: spectrogram """
    if tf_spectragram:
        plt.axes(h_ax[1,0])
        plt.pcolormesh(dist_spcg, spcg_f, np.log(spcg), cmap='inferno')
        plt.xlabel('distance, um')
        plt.ylabel('frequency, Hz')
        plt.title('LFP spectragram')

        """ colorbar """
        ax_loc = h_ax[1, 0].get_position()
        h_colorbar = plt.colorbar( cax = h_fig.add_axes([ax_loc.x1+0.02, ax_loc.y0, 0.02, ax_loc.y1-ax_loc.y0]))

        """ colorbar, interactive control widgets """
        if tf_spectragram_interactive:
            clim_org_min, clim_org_max = h_colorbar.get_clim()
            clim_range = clim_org_max - clim_org_min
            h_slider_clim_min = mpl.widgets.Slider(ax=h_fig.add_axes([ax_loc.x0, 0.02, (ax_loc.x1 - ax_loc.x0) * 0.3, 0.02]),
                                                  label='c_min', valmin=clim_org_min-0.5*clim_range, valmax=clim_org_min+0.5*clim_range, valinit=clim_org_min)
            h_slider_clim_max = mpl.widgets.Slider(ax=h_fig.add_axes([(ax_loc.x0+ax_loc.x1)*0.5, 0.02, (ax_loc.x1-ax_loc.x0)*0.3, 0.02]),
                                                  label='c_max', valmin=clim_org_max-0.5*clim_range, valmax=clim_org_max+0.5*clim_range, valinit=clim_org_max)
            def update_clim_min(val):
                plt.axes(h_ax[1,0])
                plt.clim(vmin=val)
            def update_clim_max(val):
                plt.axes(h_ax[1, 0])
                plt.clim(vmax=val)
            h_slider_clim_min.on_changed(update_clim_min)
            h_slider_clim_max.on_changed(update_clim_max)

            h_button_savefig = mpl.widgets.Button(ax=h_fig.add_axes([ax_loc.x1, 0.02, 0.05, 0.03]), label='save')
            def savefig(event):
                plt.savefig('{}/electrode_depth_profile_tank_{}.png'.format(path_save_fig, keyword_tank))
            h_button_savefig.on_clicked(savefig)
    h_ax[0,0].set_xlim([dist_bin.min(), dist_bin.max()])
    plt.suptitle('electrode depth profile, tank {}'.format(keyword_tank))

    """ return results """
    return_dict = {'spk': (dist_bin, spk_bin), 'spcg': None, 'slider': None}
    if tf_spectragram:
        return_dict['spcg'] = (dist_spcg, spcg_f, spcg)
    if tf_spectragram_interactive and tf_spectragram:    # to return the widegets object is important, otherwise widgets do not work whtin functions
        return_dict['widgets'] = (h_slider_clim_min, h_slider_clim_max, h_button_savefig)

    return return_dict

""" ----- process data day by day ----- """
plot_electrode_depth_profile(keyword_tank='Thor_S4-180303', keyword_blk='retract', tf_spectragram=True)

plot_electrode_depth_profile(keyword_tank='Thor_S4-180227', keyword_blk='retract', tf_spectragram=True)

plot_electrode_depth_profile(keyword_tank='Thor_U16-180224', keyword_blk='retract', tf_spectragram=True,
                             ch_index=16, range_ch=np.arange(1, 16 + 1), range_ch_pos_rel=np.arange(-1500, 1, 100))

plot_electrode_depth_profile(keyword_tank='Thor_S4-180222', keyword_blk='retract', tf_spectragram=True)

plot_electrode_depth_profile(keyword_tank='Thor_S4-180217', keyword_blk='retreat', tf_spectragram=True)

plot_electrode_depth_profile(keyword_tank='Thor_S4-180214', keyword_blk='retreat', tf_spectragram=True)


