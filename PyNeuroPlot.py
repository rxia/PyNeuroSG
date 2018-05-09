import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits
import pandas as pd
import numpy as np
import scipy as sp
from scipy import signal as sgn
import re
import warnings

import misc_tools
import signal_align
import df_ana
import PyNeuroAna as pna

mpl.style.use('ggplot')



""" df (pandas DataFrame) related opeartions """
# from df_ana import GroupPlot, DfPlot


def SpkWfPlot(seg, sortcode_min =1, sortcode_max =100, ncols=None):
    """
    Plot spk waveforms of a segment, one channel per axes, different sort codes are color coded

    :param seg:          neo blk.segment
    :param sortcode_min: the min sortcode included in the plot, default to 1
    :param sortcode_max: the max sortcode included in the plot, default to 1
    :param ncols:        number of columns in the subplot
    :return:             figure handle
    """
    
    N_chan = max([item.annotations['channel_index'] for item in seg.spiketrains])   # ! depend on the frame !
    if ncols is None:
        nrows, ncols = AutoRowCol(N_chan)
    else:
        nrows  = int(np.ceil(1.0 * N_chan / ncols))
    # spike waveforms:
    fig, axes2d = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', sharey='all',
                               figsize=(8,6), squeeze=False)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    axes1d = axes2d.ravel()
    sortcode_color = {0: np.array([225, 217, 111]) / 255.0,
                      1: np.array([149, 196, 128]) / 255.0,
                      2: np.array([112, 160, 234]) / 255.0,
                      3: np.array([142, 126, 194]) / 255.0,
                      4: np.array([202,  66,  45]) / 255.0,
                      5: np.array([229, 145,  66]) / 255.0}
    for i, axes in enumerate(axes1d):
        plt.sca(axes)
        plt.xticks([])
        plt.yticks([])
        plt.text(0.1, 0.8, 'C{}'.format(i + 1), transform=axes.transAxes)
        try:  # for differenet versions of matploblib
            try:
                h_axes_top.set_facecolor([0.98, 0.98, 0.98])
            except:
                h_axes_top.set_axis_bgcolor([0.98, 0.98, 0.98])
        except:
            pass
    for i in range(len(seg.spiketrains)):
        try:
            cur_chan = int(re.match('Chan(\d*) .*', seg.spiketrains[i].name).group(1))  # ! depend on the naming !
            cur_code = int(re.match('.* Code(\d*)', seg.spiketrains[i].name).group(1))  # ! depend on the naming !
        except:
            cur_chan = i
            cur_code = 0
            print (misc_tools.red_text('the segment does not contain name like "Chan1 Code2"'))
        if sortcode_min <= cur_code < sortcode_max:
            axes_cur = axes1d[cur_chan - 1]
            plt.sca(axes_cur)
            h_text = plt.text(0.4, 0.12 * cur_code, 'N={}'.format(len(seg.spiketrains[i])), transform=axes_cur.transAxes, fontsize='smaller')
            h_waveform = plt.plot(np.squeeze(np.mean(seg.spiketrains[i].waveforms, axis=0)))
            if cur_code >= 0 and cur_code <= 5:
                h_waveform[0].set_color(sortcode_color[cur_code])
                h_text.set_color(sortcode_color[cur_code])
    axes.set_ylim( np.max(np.abs(np.array(axes.get_ylim())))*np.array([-1,1]) )    # make y_lim symmetrical
    fig.suptitle('spike waveforms, y_range={} uV'.format(np.round(np.diff(np.array(axes.get_ylim())) * 1000000)[0] ))

    return fig


def ErpPlot_singlePanel(erp, ts=None, tf_inverse_color=False, cmap='coolwarm', c_lim_style='diverge', trace_scale=1):
    """
    ERP plot in a single panel, where trace and color plot are superimposed. ideal for ERP recorded with linear probe

    :param erp:   erp traces, [N_chan, N_ts]
    :param ts:    timestapes
    :param tf_inverse_color:  if inverse sign for color plot.  Useful for CSD plot since minus "sink" are commonly plot as red
    :return:      None
    """
    N,T = erp.shape
    if ts is None:
        ts= np.arange(T)


    if c_lim_style == 'diverge':
        scale_signal = np.nanmax(np.abs(erp))
        center_signal = 0
        c_min = -scale_signal
        c_max = +scale_signal
    else:
        scale_signal = np.nanmax(erp)-np.nanmin(erp)
        center_signal = np.nanmean(erp)
        c_min = np.nanmin(erp)
        c_max = np.nanmax(erp)


    if tf_inverse_color:
        erp_plot = -erp
    else:
        erp_plot = erp
    plt.pcolormesh(center2edge(ts), center2edge(range(N)), erp_plot, cmap=cmap, vmin=c_min, vmax=c_max)
    plt.plot(ts, ( -(erp-center_signal)/scale_signal/2*trace_scale+np.expand_dims(np.arange(N), axis=1)).transpose(), 'k', alpha=0.2)  # add "-" because we later invert y axis
    ax = plt.gca()
    # ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_ylim([N-0.5, -0.5])


def ErpPlot(array_erp, ts=None, array_layout=None, depth_linear=None, title="ERP"):
    """
    ERP (event-evoked potential) plot

    :param array_erp:     a 2d numpy array ( N_chan * N_timepoints )
    :param ts:            a 1d numpy array ( N_timepoints )
    :param array_layout:  electrode array layout:

                          * if None, assume linear layout,
                          * otherwise, use the the give array_layout in the format: {chan: (row, col)}
    :param depth_linear:   a list of depth
    :return:              figure handle
    """

    [N_chan, N_ts] = array_erp.shape
    if ts is None:
        ts = np.arange(N_ts)
    if array_layout is None:                # if None, assumes linear layout
        offset_plot_chan = (array_erp.max()-array_erp.min())/5

        if depth_linear is None:
            depth_linear = np.array(np.arange(array_erp.shape[0]), ndmin=2).transpose() * offset_plot_chan
            array_erp_offset = (array_erp - depth_linear).transpose()
        else:
            depth_linear = np.array(depth_linear)
            depth_scale = depth_linear.max()-depth_linear.min()
            if depth_scale < 10**(-9):
                depth_scale = 1
            depth_linear = np.expand_dims(depth_linear, axis=1)
            array_erp_offset = (array_erp / offset_plot_chan /15.0 * depth_scale - depth_linear).transpose()

        name_colormap = 'rainbow'
        cycle_color = plt.cm.get_cmap(name_colormap)(np.linspace(0, 1, N_chan))

        h_fig, h_axes = plt.subplots(1,2, figsize=(9,6))
        plt.axes(h_axes[0])
        for i in range(N_chan):
            plt.plot(ts, array_erp_offset[:,i], c=cycle_color[i]*0.9, lw=2)
        try:  # for differenet versions of matploblib
            try:
                h_axes_top.set_facecolor([0.95, 0.95, 0.95])
            except:
                h_axes_top.set_axis_bgcolor([0.95, 0.95, 0.95])
        except:
            pass
        plt.xlim(ts[0],ts[-1])
        # plt.ylim( -(N_chan+2)*offset_plot_chan, -(0-3)*offset_plot_chan,  )
        plt.title(title)
        plt.xlabel('time from event onset (s)')
        plt.ylabel('Voltage (V)')

        plt.axes(h_axes[1])
        plt.pcolormesh(center2edge(ts), center2edge(np.arange(N_chan)+1) , np.array(array_erp), cmap=plt.get_cmap('coolwarm'))
        color_max = np.max(np.abs(np.array(array_erp)))
        plt.clim(-color_max, color_max)
        plt.xlim(ts[0], ts[-1])
        plt.ylim( center2edge(np.arange(N_chan)+1)[0], center2edge(np.arange(N_chan)+1)[-1]  )
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel('time from event onset (s)')
        plt.ylabel('channel index')
    else:                                 # use customized 2D layout, e.g. GM32 array
        text_props = dict(boxstyle='round', facecolor='w', alpha=0.5)
        [h_fig, h_axes] = create_array_layout_subplots(array_layout, tf_linear_indx=False)
        plt.tight_layout()
        h_fig.subplots_adjust(hspace=0.02, wspace=0.02)
        h_fig.set_size_inches([8, 8],forward=True)

        for ch in range(N_chan):
            plt.axes(h_axes[array_layout[ch + 1]])
            plt.plot(ts, array_erp[ch, :], linewidth=2, color='dimgray')
            plt.text(0.1, 0.8, 'C{}'.format(ch + 1), transform=plt.gca().transAxes, fontsize=10, bbox=text_props)
        plt.xlim(ts[0], ts[-1])

        # axis appearance
        if False:
            for ax in h_axes.flatten():
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if len(ax.lines)==0:
                    try:  # for differenet versions of matploblib
                        try:
                            h_axes_top.set_facecolor([1,1,1,0])
                        except:
                            h_axes_top.set_axis_bgcolor([1,1,1,0])
                    except:
                        pass
            ax_bottomleft = h_axes[-1,0]
            plt.axes(ax_bottomleft)
            ax_bottomleft.get_xaxis().set_visible(True)
            ax_bottomleft.get_yaxis().set_visible(True)
            ax_bottomleft.set_xlabel('time')
            ax_bottomleft.set_ylabel('Voltage')
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            ylim_max = np.max(np.abs(ax_bottomleft.get_ylim()))
            ax_bottomleft.set_ylim([-ylim_max, ylim_max])

        plt.suptitle(title, fontsize=18)

    return h_fig, h_axes


def RfPlot(data_neuro, indx_sgnl=0, data=None, t_focus=None, tf_scr_ctr=False,
           psth_overlay=True, t_scale=None, fr_scale=None, sk_std=None ):
    """
    plot RF using one single plot

    :param data_neuro:  the data_neuro['cdtn'] contains 'x' and 'y', which represents the location of stimulus
    :param indx_sgnl:   index of signal, i.e. the data_neuro['cdtn'][:,:,indx_sgnl] would be used for plotting
    :param data:        if not None, use the data instead of data_neuro['data'][:,:,indx_sgnl], a 2D array of shape [N_trials, N_ts]
    :param t_focus:     duration of time used to plot heatmap, e.g. [0.050, 0.200]
    :param tf_scr_ctr:  True/False, plot [0.0] point of screen
    :param psth_overlay:True/False overlay psth
    :param t_scale:     for psth overlay, duration of time (ts) to be mapped to unit 1 of space
    :param fr_scale:    for psth overlay, scale for firing rate
    :param sk_std:      smooth kernel standard deviation, e.g. 0.005, for 5ms
    :return:
    """

    if data is None:
        data = data_neuro['data'][:,:,indx_sgnl]
    else:
        data = data[:, :, indx_sgnl]
    ts = np.array(data_neuro['ts'])

    if sk_std is not None:
        data = pna.SmoothTrace(data=data, sk_std=sk_std, ts=ts)

    if t_focus is None:
        t_focus = ts[[0,-1]]

    # get x and y values
    x_grid = np.unique(np.array(data_neuro['cdtn'])[:,0])
    y_grid = np.unique(np.array(data_neuro['cdtn'])[:,1])
    x_spacing = np.mean(np.diff(x_grid))
    y_spacing = np.mean(np.diff(y_grid))
    if t_scale is None:
        t_scale = ( data_neuro['ts'][-1] - data_neuro['ts'][0] )/x_spacing*1.1

    fr_2D = np.zeros([len(x_grid),len(y_grid)])
    for i, xy in enumerate(data_neuro['cdtn']):
        x,y = xy
        i_x = np.flatnonzero(x_grid == x)[0]
        i_y = np.flatnonzero(y_grid == y)[0]
        fr_2D[i_x, i_y] = np.mean( data[ data_neuro['cdtn_indx'][data_neuro['cdtn'][i]] , np.argwhere(np.logical_and(ts>=t_focus[0], ts<t_focus[1])) ])

    if fr_scale is None:
        fr_scale = np.nanmax(fr_2D)*2

    if psth_overlay:
        # plt.figure()
        plt.title(data_neuro['signal_info'][indx_sgnl]['name'])
        plt.pcolormesh( center2edge(x_grid), center2edge(y_grid), fr_2D.transpose(), cmap='inferno')

        for i, xy in enumerate(data_neuro['cdtn']):
            plt.fill_between(xy[0] - x_spacing/2 + (ts-ts[0]) / t_scale, xy[1] - y_spacing/2,
                             xy[1] - y_spacing/2 + np.mean( data[data_neuro['cdtn_indx'][data_neuro['cdtn'][i]], :], axis=0) / fr_scale,
                             color='deepskyblue', alpha=0.5)
            # plt.fill_between( xy[0]+np.array(data_neuro['ts'])/t_scale, xy[1], xy[1]+np.mean( data_neuro['data'][ data_neuro['cdtn_indx'][data_neuro['cdtn'][i]] ,:,indx_sgnl], axis=0 )/fr_scale, color=[0,0,0,0.5] )
            # plt.plot(xy[0],xy[1],'r+', linewidth=1)
        plt.xlim(center2edge(x_grid)[[0, -1]])
        plt.ylim(center2edge(y_grid)[[0, -1]])
        plt.axis('equal')
    else:
        plt.pcolormesh(center2edge(x_grid), center2edge(y_grid), fr_2D.transpose(), cmap='inferno')
    if tf_scr_ctr:
        plt.plot(x_grid[[0,-1]], [0,0], 'w-', Linewidth=0.5)
        plt.plot([0,0], y_grid[[0,-1]], 'w-', Linewidth=0.5)


def CreateSubplotFromGroupby(df_groupby_ord, figsize=None, tf_title=True):
    """ todo: description """

    if len(df_groupby_ord) == 0:
        raise Exception('input dictionary can not be empty')
    elif isinstance(list(df_groupby_ord.values())[0], int):  # input is {str/num: int}, e.g. {'a': 0, 'b': 1}
        N = max(df_groupby_ord.values())+1
        num_r, num_c = AutoRowCol(N)
        h_fig, h_axes = plt.subplots(num_r, num_c, sharex='all', sharey='all', squeeze=False, figsize=figsize)
        h_axes = np.ravel(h_axes)
        h_axes = {key: h_axes[val] for key, val in df_groupby_ord.items()}
    else:      # input is {tuple_of_cdtn: tuple_of_order}, e.g. {('a',1): (0,0), ('b',1): (1,1)}
        Ns = list(map(lambda a: max(a)+1, zip(*df_groupby_ord.values())))
        if len(Ns)==1:
            h_fig, h_axes = plt.subplots(Ns[0], 1, sharex='all', sharey='all', squeeze=False, figsize=figsize)
            h_axes = {key: h_axes[val][0] for key, val in df_groupby_ord.items()}
        elif len(Ns)==2:
            h_fig, h_axes = plt.subplots(Ns[0], Ns[1], sharex='all', sharey='all', squeeze=False, figsize=figsize)
            h_axes = {key: h_axes[val] for key, val in df_groupby_ord.items()}
        else:
            raise Exception('val can not be more than two dimensions in input {key: val}')

    if tf_title:
        for key in h_axes:
            plt.axes(h_axes[key])
            plt.title(str(key))

    return h_fig, h_axes


def SmartSubplot(data_neuro, functionPlot=None, dataPlot=None, suptitle='', tf_colorbar=False):
    """
    Smart subplots based on the data_neuro['cdtn']

    in each panel, plot using function 'functionPlot', on data 'dataPlot';
    if cdtn is 1D, automatically decide row and column; if 2D, use dim0 as rows and dim1 as columns
    :param data_neuro:    dictionary containing field 'cdtn' and 'cdtn_indx', which directing the subplot layout
    :param functionPlot:  the plot function in each panel
    :param dataPlot:      the data that plot function applies on; in each panel, its first dim is sliced using data_neuro['cdtn_indx'], if None, use data_neuro[data]
    :return:              [h_fig, h_ax]
    """
    if 'cdtn' not in data_neuro.keys():            # if the input data does not contain the field for sorting
        raise Exception('data does not contain fields "cdtn" for subplot')
    if ('data' in data_neuro.keys()) and dataPlot is None:  # if dataPlot is None, use data_neuro[data]
        dataPlot = data_neuro['data']
    if isSingle(data_neuro['cdtn'][0]):            # if cdtn is 1D, automatically decide row and column
        N_cdtn = len(data_neuro['cdtn'])
        [n_rows, n_cols] = AutoRowCol(N_cdtn)
        [h_fig, h_ax]= plt.subplots(n_rows, n_cols, sharex=True, sharey=True)      # creates subplots
        h_ax = np.array(h_ax).flatten()
        for i, cdtn in enumerate(data_neuro['cdtn']):
            plt.axes(h_ax[i])
            if (functionPlot is not None) and (dataPlot is not None):    # in each panel, plot
                functionPlot(dataPlot.take(data_neuro['cdtn_indx'][cdtn], axis=0))
            plt.title(cdtn)
    elif len(data_neuro['cdtn'][0])==2:            # if cdtn is 2D,  use dim10 as rows and dim1 as columns
        [cdtn0, cdtn1] = zip( *data_neuro['cdtn'] )
        cdtn0 = list(set(cdtn0))
        cdtn1 = list(set(cdtn1))
        n_rows = len(cdtn0)
        n_cols = len(cdtn1)
        [h_fig, h_ax] = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)    # creates subplots
        for cdtn in data_neuro['cdtn']:
            i = cdtn0.index(cdtn[0])
            j = cdtn1.index(cdtn[1])
            plt.axes(h_ax[i,j])
            if (functionPlot is not None) and (dataPlot is not None):    # in each panel, plot
                functionPlot(dataPlot.take(data_neuro['cdtn_indx'][cdtn], axis=0))
            plt.title(cdtn)
    else:                                          # if cdtn is more than 2D, raise exception
        raise Exception('cdtn structure does not meet the requirement of SmartSubplot')

    h_fig.set_size_inches([12,9], forward=True)
    try:
        plt.suptitle(suptitle.__str__() + '    ' + data_neuro['grpby'].__str__(), fontsize=16)
    except:
        pass
    # share clim across axes
    try:
        c_lim = share_clim(h_ax)
    except:
        warnings.warn('share clim was not successful')

    if tf_colorbar:
        try:
            h_fig.colorbar(plt.gci(), ax=h_ax.flatten().tolist())
        except:
            warnings.warn('can not create colorbar')

    return [h_fig, h_ax]


def PsthPlot(data, ts=None, cdtn=None, limit=None, sk_std=None, subpanel='auto', color_style='discrete', tf_legend=False, xlabel=None, ylabel=None):
    """
    funciton to plot psth with a raster panel on top of PSTH, works for both spike data and LFP data

    :param data:        neuro data, np array of various size and dtpye:
                            size: 2D [N_trials * N_ts] or 3D [N_trials * N_ts * N_signals]
                            dtype: boolean (spike exist or not) or float (LFP continuous values)
    :param ts:          1D array containing timestamps for data (length is N_ts)
    :param cdtn:        conditions used to group
                            if data is 2D, represent the type of trials,  len(cdtn)=N_ts
                            if data is 3D, represent the type of signals, len(cdtn)=N_signals
    :param limit:       index array to select a subset of the trials of data, i.e., data=data[limit,:]
    :param sk_std:      std of gaussian smoothness kernel, applied along time axis, default to None
    :param subpanel:    types of sub-panel on tops of PSTH, default to 'auto':

                        * if 'spk'  : data2D is boolean, where every True value represents a spike, plot line raster
                        * if 'LFP'  : data2D is continuous float, plot pcolormesh
                        * if 'auto' : use data format to decide which plot to use
                        * if ''     : does not create subpanel
    :param color_style: 'discrete' or 'continuous'
    :param tf_legend:   boolean, true/false to plot legend
    :param x_label:     string
    :param y_label:     string
    :return:            axes of plot: [ax_psth, ax_raster]
    """

    """ ----- process the input, to work with various inputs ----- """
    if limit is not None:       # select trials of interest
        limit = np.array(limit)
        data = data[limit]

    cdtn_unq = None
    if len(data.shape) == 3:     # if data is 3D [N_trials * N_ts * N_signals]
        [N,T,S] = data.shape
        data2D = signal_align.data3Dto2D(data)   # re-organize as 2D [(N_trials* N_signals) * N_ts ]
        if cdtn is None:                         # condition is the tag of the last dimension (e.g. signal name)
            cdtn = np.array([[i]*N for i in range(S)]).ravel()
        elif len(cdtn) == S:
            cdtn_unq = np.array(cdtn)
            cdtn = np.array([[i]*N for i in cdtn]).ravel()
        else:
            cdtn = ['']*(N*S)
            warnings.warn('cdtn length does not match the number of signals in data')

    elif len(data.shape) == 2:   # if data is 2D [N_trials * N_ts]
        data2D = data
        [N, T] = np.shape(data2D)
        if cdtn is None:                         # condition is the tag of the trials
            cdtn = np.array(['']*N)
        elif limit is not None:
            cdtn = cdtn[limit]
    else:
        raise(Exception('input data does not have the right dimension'))

    cdtn = np.array(cdtn)
    if cdtn_unq is None:
        cdtn_unq = get_unique_elements(cdtn)        # unique conditions
    M = len(cdtn_unq)

    if ts is None:
        ts = np.arange( np.size(data2D, axis=1) )
    else:
        ts = np.array(ts)


    """ ----- calculate PSTH for every condition ----- """

    ax_psth = plt.gca()
    psth_cdtn = np.zeros([M, T])
    N_cdtn     = np.zeros(M).astype(int)
    N_cdtn_cum = np.zeros(M+1).astype(int)
    for k, cdtn_k in enumerate(cdtn_unq):
        N_cdtn[k] = np.sum(cdtn==cdtn_k)
        if N_cdtn[k] == 0:
            psth_cdtn[k, :] = psth_cdtn[k, :]
        else:
            psth_cdtn[k, :] = np.mean(data2D[cdtn==cdtn_k], axis=0)
    N_cdtn_cum[1:] = np.cumsum(N_cdtn)

    if sk_std is not None:  # condition for using smoothness kernel
        ts_interval = np.diff(np.array(ts)).mean()  # get sampling interval
        kernel_std = sk_std / ts_interval  # std in frames
        kernel_len = int(np.ceil(kernel_std) * 3 * 2 + 1)  # num of frames, 3*std on each side, an odd number
        smooth_kernel = sgn.gaussian(kernel_len, kernel_std)
        smooth_kernel = smooth_kernel / smooth_kernel.sum()  # normalized smooth kernel
        for k in range(M):
            psth_cdtn[k, :] = np.convolve(psth_cdtn[k, :], smooth_kernel, 'same')

    colors = gen_distinct_colors(M, luminance=0.7, alpha=0.8, style=color_style)

    """ ========== plot psth ========== """
    hl_lines = []
    for k, cdtn_k in enumerate(cdtn_unq):
        hl_line, = plt.plot(ts, psth_cdtn[k, :], c=colors[k])
        hl_lines.append(hl_line)

    textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.9, 'N={}'.format(N), transform=ax_psth.transAxes,
             verticalalignment='top', fontsize='x-small', bbox=textprops)

    ax_psth.set_xlim([ts[0], ts[-1]])
    if tf_legend:
        plt.legend(cdtn_unq, labelspacing=0.1, prop={'size': 8},
                   fancybox=True, framealpha=0.5)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    """ ========== plot a subpanel, either rasters or LFP pcolor ========== """
    if subpanel is not '':
        # ========== sub-panel for raster or p-color  ==========
        ax_raster = add_axes_on_top(ax_psth, r=0.4)
        plt.axes(ax_raster)

        if subpanel == 'auto':      # if auto, use the data features to determine raster (spk) plot or pcolor (LFP) plot
            if len(np.unique(data2D)) <=20:
                subpanel = 'spk'
            else:
                subpanel = 'LFP'

        if subpanel is 'spk':       # ---------- plot raster (spike trains) ----------
            RasterPlot(data2D, ts=ts, cdtn=cdtn, colors=colors, max_rows=None, RasterType=subpanel)

        elif subpanel is 'LFP':     # ---------- plot_pcolor (LFP) ----------
            RasterPlot(data2D, ts=ts, cdtn=cdtn, colors=colors, max_rows=None, RasterType=subpanel)
        plt.setp(ax_raster.get_xticklabels(), visible=False)
    else:
        ax_raster = None

    return [ax_psth, ax_raster]


def RasterPlot(data2D, ts=None, cdtn=None, colors=None, RasterType='auto', max_rows=None):
    """
    Spike/LFP raster Plot, where evary row presresent one trial, sorted by cdtn

    :param data2D:   2D np.array of boolean/float values, [N_trial * N_ts]
    :param ts:       1D np.array of timestamps, 1D np.array of length N_ts,    used as x axis for plot
    :param cdtn:     condition of every trial,  1D np.array of length N_trial, used to sort trials
    :param colors:   a list of colors for every unique condition
    :param RasterType:    string, 'spk', 'LFP' or 'auto', default to 'auto':

                     * if 'spk'  : data2D is boolean, where every True value represents a spike, plot line raster
                     * if 'LFP'  : data2D is continuous float, plot pcolormesh
                     * if 'auto' : use data2D format to decide which plot to use
    :return:         handle of raster plot
    """
    data2D = np.array(data2D)
    [N, T] = data2D.shape

    if cdtn is None:        # if cdtn is none, fill with blank string
        cdtn = ['']*N

    # reorder index according to cdtn, so that trials of the same condition sits together in rows
    cdtn_unq = get_unique_elements(cdtn)
    M = len(cdtn_unq)
    cdtn = np.array(cdtn)
    index_reorder = np.concatenate([np.flatnonzero(cdtn == cdtn_k) for cdtn_k in cdtn_unq])
    data2D = data2D[index_reorder,:]
    cdtn = cdtn[index_reorder]

    if colors is None:      # if no color is given
        colors = gen_distinct_colors(M, luminance=0.7)

    if ts is None:
        ts = np.arange(T)

    # if N is too large, to speed up plot, randomly select a fraction to plot
    if max_rows is not None:
        if max_rows < N:
            indx_subselect = np.sort(np.random.choice(N, max_rows, replace=False))
            data2D = data2D[indx_subselect,:]
            cdtn   = cdtn[indx_subselect]
            N = max_rows

    if RasterType == 'auto':
        if len(np.unique(data2D)) <= 20:
            RasterType = 'spk'
        else:
            RasterType = 'LFP'

    # color for every raster line
    cdtn_indx_of_trial = np.zeros(N).astype(int)
    N_cdtn = np.zeros(M).astype(int)
    for k, cdtn_k in enumerate(cdtn_unq):
        N_cdtn[k] = np.sum(cdtn == cdtn_k)
        cdtn_indx_of_trial[cdtn == cdtn_k] = k
    N_cdtn_cum = np.cumsum(N_cdtn)

    if RasterType == 'spk':
        if np.sum(data2D > 0) > 0 :
            [y,x] = zip(*np.argwhere(data2D>0))
            x_ts  = ts[np.array(x)]   # x loc of raster lines
            y     = np.array(y)       # y loc of raster lines
            y_min = y - 0.0005*N
            y_max = y+1 +0.0005*N

            c = np.array(colors)[ cdtn_indx_of_trial[np.array(y)] ]

            # ----- major plot commmand -----
            h_raster = plt.vlines(x_ts, y_min, y_max, colors=c, linewidth=2)
        else:
            h_raster = plt.vlines([], [], [])

    elif RasterType == 'LFP':
        # -----  major plot command -----
        # c_axis_abs = np.max(np.abs(data2D))
        c_axis_abs = np.percentile(np.abs(data2D), 98)
        h_raster = plt.pcolormesh(center2edge(ts), range(N + 1), data2D, vmin=-c_axis_abs, vmax=c_axis_abs,
                       cmap=plt.get_cmap('coolwarm'))

        # a colored line segment illustrating the conditions
        if len(np.unique(cdtn))!=1:
            plt.vlines(ts[[0] * M], np.insert(N_cdtn_cum, 0, 0)[0:M], N_cdtn_cum, colors=colors, linewidth=10)
    else:
        raise Exception('wrong RasterType, must by either "spk" or "LFP" ')

    # horizontal lines deviding difference conditions
    plt.hlines(N_cdtn_cum, ts[0], ts[-1], color='k', linewidth=1)

    plt.xlim(ts[0],ts[-1])
    plt.ylim(0, N)
    plt.gca().invert_yaxis()
    plt.yticks(keep_less_than([0]+N_cdtn_cum.tolist(), 8), fontsize='x-small')
    return h_raster


def DataNeuroSummaryPlot(data_neuro, sk_std=None, signal_type='auto', suptitle='', xlabel='', ylabel='', tf_legend=False):
    """
    Summary plot for data_neuro, uses SmartSubplot and PsthPlot

    Use data_neuro['cdtn'] to sort and group trials in to sub-panels, plot all signals in every sub-panel

    :param data_neuro:  data_neuro structure, see moduel signal_align.blk_align_to_evt
    :param sk_std:      smoothness kernel, if None, set automatically based on data type
    :param signal_type: 'spk', 'LFP' or 'auto'
    :param suptitle:    title of figure
    :param xlabel:      xlabel
    :param ylabel:      ylabel
    :return:            figure handle
    """

    ts = data_neuro['ts']
    if signal_type == 'auto':
        if len(np.unique(data_neuro['data'])) <=20:   # signal is spk
            signal_type = 'spk'
        else:
            signal_type = 'LFP'           # signal is LFP

    if sk_std is None:
        if signal_type == 'spk':
            sk_std = (ts[-1]-ts[0])/100
    if xlabel =='':
        xlabel = 't (s)'
    if ylabel =='':
        if signal_type == 'spk':
            ylabel = 'firing rate (spk/s)'
        elif signal_type == 'LFP':
            ylabel = 'LFP (V)'

    try:
        name_signals = [x['name'] for x in data_neuro['signal_info']]
    except:
        name_signals = np.arange(data_neuro['data'].shape[2])

    if 'cdtn' in data_neuro:
        # plot function for every panel
        functionPlot = lambda x: PsthPlot(x, ts=ts, cdtn=name_signals, sk_std=sk_std, tf_legend=tf_legend,
                                          xlabel=xlabel, ylabel=ylabel, color_style='continuous')

        # plot every condition in every panel
        h = SmartSubplot(data_neuro, functionPlot, suptitle=suptitle)
    else:
        h = PsthPlot(data_neuro['data'], ts=ts, cdtn=name_signals, sk_std=sk_std, tf_legend=tf_legend,
                 xlabel=xlabel, ylabel=ylabel, color_style='continuous')
    return h


def PsthPlotCdtn(data_neuro, data_df, i_signal=0, grpby='', fltr=[], sk_std=None, subpanel='', tf_legend=False):
    """ Obsolete funciton """

    data_neuro = signal_align.neuro_sort(data_df, grpby=grpby, fltr=fltr, neuro=data_neuro)

    ts = data_neuro['ts']
    # plot function for every panel
    functionPlot = lambda x: PsthPlot(x, ts=ts, sk_std=sk_std, tf_legend=tf_legend, color_style='continuous')

    # plot every condition in every panel
    h = SmartSubplot(data_neuro, functionPlot, dataPlot=data_neuro['data'][:,:,i_signal])

    return h


def SpectrogramPlot(spcg, spcg_t=None, spcg_f=None, limit_trial = None,
                 tf_phase=False, tf_mesh_t=False, tf_mesh_f=False,
                 tf_log=False, time_baseline=None,
                 t_lim = None, f_lim = None, c_lim = None, c_lim_style=None, name_cmap=None,
                 rate_interp=None, tf_colorbar= False, quiver_scale=None, max_quiver=None):
    """
    plot power spectrogram or coherence-gram, input spcg could be [ N_t * N*f ] or [ N_trial * N_t * N*f ], real or complex

    :param spcg:          2D numpy array, [ N_t * N*f ] or 3D numpy array [ N_trial * N_t * N*f ], either real or complex:

                            * if [ N_trial * N_t * N*f ], average over trial and get [ N_t * N*f ]
                            * if complex, plot spectrogram using its abs value, and plot quiver using the complex value
    :param spcg_t:        tick of time, length N_t
    :param spcg_f:        tick of frequency, length N_f
    :param limit_trial:   index array to specify which trials to use
    :param tf_phase:      true/false, plot phase using quiver (spcg has to be complex): points down if negative phase (singal1 lags signal0 for coherence)
    :param tf_mesh_t:     true/false, plot LinemeshFlatPlot, focuse on t, (horizontal lines)
    :param tf_mesh_f:     true/false, plot LinemeshFlatPlot, focuse on f, (vertical lines)
    :param tf_log:        true/false, use log scale
    :param c_lim_style:   'basic' (min, max), 'from_zero' (0, max), or 'diverge' (-max, max); default to None, select automatically based on tf_log and time_baseline
    :param time_baseline: baseline time period to be subtracted, eg. [-0.1, 0.05], default to None
    :param name_cmap:     name of color map to use, default to 'inferno', if use diverge c_map, automatically change to 'coolwarm'; another suggested one is 'viridis'
    :param rate_interp:   rate of interpolation for plotting, if None, do not interpolate, suggested value is 8
    :param tf_colorbar:   true/false, plot colorbar
    :return:              figure handle
    """

    if tf_log:                   # if use log scale
        spcg = np.log(spcg * 10**6 + 10**(-32)) - 6         # prevent log(0) error

    if limit_trial is not None:
        spcg = spcg[limit_trial, :, :]

    if spcg.ndim == 3:           # if contains many trials, calculate average
        spcg = np.mean(spcg, axis=0)

    if np.any(np.iscomplex(spcg)):     # if imaginary, keep the origianl and calculate abs
        spcg_complex = spcg
        spcg = np.abs(spcg)
    else:
        spcg_complex = None

    if spcg_t is None:
        spcg_t = np.arange(spcg.shape[1]).astype(float)

    if spcg_f is None:
        spcg_f = np.arange(spcg.shape[0]).astype(float)

    # if use reference period for baseline, subtract baseline
    if time_baseline is not None:
        spcg_baseline = np.mean( spcg[:,np.logical_and(spcg_t >= time_baseline[0], spcg_t < time_baseline[1])],
                                 axis=1, keepdims=True)
        spcg = spcg - spcg_baseline

    # set x, y limit
    if t_lim is None:
        t_lim = spcg_t[[0, -1]]
    else:
        tf_temp = np.logical_and(spcg_t>=t_lim[0], spcg_t<=t_lim[1])
        spcg_t = spcg_t[tf_temp]
        spcg   = spcg[:, tf_temp]
    if f_lim is None:
        f_lim = spcg_f[[0, -1]]
    else:
        tf_temp = np.logical_and(spcg_f >= f_lim[0], spcg_f <= f_lim[1])
        spcg_f = spcg_f[tf_temp]
        spcg = spcg[tf_temp, :]

    # set color limit
    if c_lim is None:
        if c_lim_style is None:
            if time_baseline is not None:  # if use reference period, symmetric about zero
                c_lim_style = 'diverge'
            else:
                if (not tf_log) and (not np.any(spcg<0)):  # if not log scale and no nagative values
                    c_lim_style = 'from_zero'
                else:                                      # otherwise, use basic
                    c_lim_style = 'basic'

        if c_lim_style == 'diverge':       # if 'deverge', set to [-a,a]
            c_lim = [-np.max(np.abs(spcg)), np.max(np.abs(spcg))]
        elif c_lim_style == 'from_zero':   # if 'from zero', set to [0,a]
            c_lim = [0, np.max(spcg)]
        elif c_lim_style == 'basic':       # otherwise, use 'basic', set to [a,b]
            c_lim = [np.min(spcg), np.max(spcg)]
        else:
            c_lim = [np.min(spcg), np.max(spcg)]
            warnings.warn('c_lim_style not recognized, must be "basic", "from_zero", "diverge", or None ')

    if name_cmap is None:
        if c_lim_style == 'diverge':
            name_cmap = 'coolwarm'  # use divergence colormap
        else:
            name_cmap = 'inferno'   # default to inferno

    if rate_interp is not None:
        f_interp = sp.interpolate.interp2d(spcg_t, spcg_f, spcg, kind='linear')
        spcg_t_plot = np.linspace(spcg_t[0], spcg_t[-1], (len(spcg_t)-1)*rate_interp+1 )
        spcg_f_plot = np.linspace(spcg_f[0], spcg_f[-1], (len(spcg_f)-1)*rate_interp+1 )
        spcg_plot   = f_interp(spcg_t_plot, spcg_f_plot)
    else:
        spcg_t_plot = spcg_t
        spcg_f_plot = spcg_f
        spcg_plot = spcg

    # plot using pcolormesh
    h_plot = plt.pcolormesh(center2edge(spcg_t_plot), center2edge(spcg_f_plot),
                   spcg_plot, cmap=plt.get_cmap(name_cmap), shading= 'flat')

    plt.xlim(t_lim)
    plt.ylim(f_lim)

    if c_lim is not None:
        plt.clim(c_lim)
    # color bar
    if tf_colorbar:
        plt.colorbar()

    # quiver plot of phase: pointing down if negative phase, singal1 lags signal0 for coherence
    if tf_phase is True and spcg_complex is not None:
        try:                   # plot a subset of quivers, to prevent them from filling the whole plot
            if max_quiver is None:
                max_quiver = 32    # max number of quivers every dimension (in both axis)
            if quiver_scale is None:
                quiver_scale = 32  # about 1/32 of axis length
            [N_fs, N_ts] = spcg.shape
            indx_fs = np.array(keep_less_than(range(N_fs), max_quiver*1.0*(spcg_f.max()-spcg_f.min())/(f_lim[1]-f_lim[0])))
            indx_ts = np.array(keep_less_than(range(N_ts), max_quiver))
            plt.quiver(spcg_t[indx_ts], spcg_f[indx_fs],
                       spcg_complex[indx_fs, :][:, indx_ts].real, spcg_complex[indx_fs, :][:, indx_ts].imag,
                       color='r', units='height', pivot='mid', headwidth=5, width=0.005,
                       scale=np.percentile(spcg, 99.5) * quiver_scale)
        except:
            plt.quiver(spcg_t, spcg_f, spcg_complex.real, spcg_complex.imag,
                       color='r', units='height', pivot='mid', scale=np.percentile(spcg, 99.8) * quiver_scale)

    if tf_mesh_t:
        LinemeshFlatPlot(spcg_plot, spcg_t_plot, spcg_f_plot, N_mesh=16, axis_mesh='x')
    if tf_mesh_f:
        LinemeshFlatPlot(spcg_plot, spcg_t_plot, spcg_f_plot, N_mesh=16, axis_mesh='y')

    plt.sci(h_plot) # set the current color-mappable object to be the specrogram plot but not the quiver

    return h_plot


def LinemeshFlatPlot(data, x_grid=None, y_grid=None, axis_mesh='x',
                     N_mesh=16, scale_mesh=1.0/8, color='dimgrey'):
    """
    an alternative to pcolormesh, focuse on change along one dimension (x, or y)

    :param data:        2D numpy array (num_y * num_x), like the imput to imshow
    :param x_grid:      1D numpy array, num_x
    :param y_grid:      1D numpy array, num_x
    :param axis_mesh:   'x', or 'y', the change on which axis you are interested in
    :param N_mesh:      number of lines to plot (controls thinning)
    :param scale_mesh:  the scale of individual lines relative the axis limit
    :param color:       color of lines
    :return:            handles of the lines
    """
    if x_grid is None:
        x_grid = np.arange(data.shape[1])
    if y_grid is None:
        y_grid = np.arange(data.shape[0])
    x_grid_2D = np.expand_dims(x_grid, 0)
    y_grid_2D = np.expand_dims(y_grid, 1)

    if axis_mesh == 'x':
        factor_scale = 1.0*scale_mesh * np.abs(y_grid[-1]-y_grid[0]) / (np.nanmax(np.abs(data))-np.nanmin(np.abs(data)))
        data_plot = (data-np.nanmean(data)) *factor_scale + y_grid_2D
        if N_mesh is None:
            N_mesh = len(y_grid)
        indx_thin = keep_less_than( np.arange(len(y_grid)), N_mesh)
        h_lines = plt.plot(x_grid, data_plot[indx_thin,:].transpose(), color=color)
    elif axis_mesh == 'y':
        factor_scale = 1.0 * scale_mesh * np.abs(x_grid[-1] - x_grid[0]) / (np.nanmax(np.abs(data))-np.nanmin(np.abs(data)))
        data_plot = (data-np.nanmean(data)) *factor_scale + x_grid_2D
        if N_mesh is None:
            N_mesh = len(x_grid)
        indx_thin = keep_less_than(np.arange(len(x_grid)), N_mesh)
        h_lines = plt.plot(data_plot[:,indx_thin], y_grid, color=color)
    return h_lines



def SpectrogramAllPairPlot(data_neuro, indx_chan=None, max_trial=None, limit_gap=1, t_bin=0.2, t_step=None, f_lim = None, coh_lim=None, t_axis=1, batchsize=100, verbose=False):
    """
    Plot all LFP power specgtrogram (diagonal panels) and all pairwise coherence (off-diagonal panels)

    :param data_neuro: standard data input
    :param indx_chan:  index of channels to plot (from zero)
    :param max_trial:  a interger, use only max_trial from all trials to speedup the calculation
    :param limit_gap:  the gap between channels to plot, used when indx_chan is None. e.g. if limit_gap=4, the indx_chan=[0,4,8,...]
    :param t_bin:      time bin size for spectrogram
    :param t_step:     time step size for spectrogram
    :param f_lim:      frequency limit for plot (ylim)
    :param coh_lim:    lim for coherence plot, control clim
    :param t_axis:     axis index of time in data_neuro['data']
    :param batchsize:  batch size for ComputeSpectrogram, used to prevent memory overloading
    :param verbose:    if print some intermediate results
    :return:           [h_fig, h_ax], handles of figure and axes
    """

    import PyNeuroAna as pna

    # calculate the channel index used for plotting
    N_chan = len(data_neuro['signal_info'])
    if indx_chan is None:
        indx_chan = range(0, N_chan, limit_gap)
    data = np.take( data_neuro['data'], indx_chan,  axis=2)
    signal_info = data_neuro['signal_info'][indx_chan]
    N_plot = len(indx_chan)

    # use only a subset of trials to speedup the function
    if type(max_trial) is int:
        if max_trial < data.shape[0]:
            data = np.take( data, np.random.choice(range(data.shape[0]),max_trial, replace=False),  axis=0)

    fs = data_neuro['signal_info'][0][2]
    t_ini = np.array(data_neuro['ts'][0])

    # compute spectrogram
    spcg_all = []
    for i_plot, i_chan in enumerate(indx_chan):
        [spcg_cur, spcg_t, spcg_f] = pna.ComputeSpectrogram(data[:,:,i_plot], data1=None, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, t_axis=t_axis, f_lim=f_lim, batchsize=batchsize)
        spcg_all.append( np.mean(spcg_cur, axis=0) )

    # plot
    text_props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    [h_fig, h_ax] = plt.subplots(nrows=N_plot, ncols=N_plot, sharex=True, sharey=True, figsize=[20,20])
    h_fig.set_size_inches([12,9])
    h_fig.subplots_adjust(hspace=0, wspace=0)
    for indx_row in range(N_plot):
        for indx_col in range(N_plot):
            if indx_row == indx_col:            # power spectrogram on diagonal panels
                plt.axes(h_ax[indx_row, indx_col])
                h_plot_pow = SpectrogramPlot(spcg_all[indx_row], spcg_t, spcg_f, tf_log=True, f_lim=f_lim, time_baseline=None,
                                    rate_interp=8)
                plt.text(0.03, 0.85, '{}'.format(signal_info[indx_row][0]), transform=plt.gca().transAxes, bbox=text_props)
            elif indx_row<indx_col:             # coherence on off diagonal panels
                # compute coherence
                [cohg, _, _] = pna.ComputeCoherogram(data[:, :, indx_row], data[:, :, indx_col], fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, f_lim=f_lim,
                                       batchsize=batchsize, data0_spcg_ave=spcg_all[indx_row], data1_spcg_ave=spcg_all[indx_col])

                # plot on two symmetric panels
                plt.axes(h_ax[indx_row, indx_col])
                h_plot_coh = SpectrogramPlot(cohg, spcg_t, spcg_f, tf_log=False, f_lim=f_lim,
                                time_baseline=None, rate_interp=8, name_cmap='viridis', c_lim=coh_lim)
                # plt.text(0.03, 0.85, '{}-{}'.format(signal_info[indx_row][0],signal_info[indx_col][0]), transform=plt.gca().transAxes,
                #          bbox=text_props)
                plt.axes(h_ax[indx_col, indx_row])
                h_plot_coh = SpectrogramPlot(cohg, spcg_t, spcg_f, tf_log=False, f_lim=f_lim,
                                time_baseline=None, rate_interp=8, name_cmap='viridis', c_lim=coh_lim)
                # plt.text(0.03, 0.85, '{}-{}'.format(signal_info[indx_col][0], signal_info[indx_row][0]),
                #          transform=plt.gca().transAxes,
                #          bbox=text_props)

    # share clim for all diagonal panels
    ax_diag = h_ax.flatten()[range(0,N_plot*N_plot,N_plot+1)]
    share_clim( ax_diag )
    # share clim for all off-diagonal panels
    ax_nondiag = h_ax.flatten()[list(set(range(N_plot * N_plot)) - set(range(0, N_plot * N_plot, N_plot + 1)))]
    share_clim( ax_nondiag )

    # add colorbar
    h_fig.colorbar(h_plot_pow, ax=h_ax[0:N_plot//2,:].flatten().tolist())
    h_fig.colorbar(h_plot_coh, ax=h_ax[N_plot//2:N_plot,:].flatten().tolist())
    # plt.suptitle('Power Spectrogram and Cohrence plot', fontsize=12)

    return [h_fig, h_ax]



def EmbedTracePlot(loc_embed, traces=None, labels=None, labels_interactive=None, color=None, highlight=None):
    """
    Plot trances (e.g. ERPs) in the embeded 2D space, if labels_interactive is not None, the plot is interactive

    :param loc_embedding: [N*2] array, where each row stores the [x, y] of every data point in the embedded space
    :param traces:        if None, do not plot original
    :param labels:        labels of every point
    :param labels_interactive: labels of every point, shown as annotation when clicked by mouse
    :return:
    """

    """" scatter plot for every data point """
    if color is None:
        color=np.zeros([len(loc_embed), 3])+0.5
    h_scatter = plt.scatter(loc_embed[:, 0], loc_embed[:, 1], color='k', alpha=0.3, picker=True)

    """ label shown in figure """
    if labels is not None:
        for i, xy in enumerate(loc_embed):
            plt.annotate('{}'.format(labels[i]), xy=xy)

    """ plot traces """
    if traces is not None:
        scale_trace = np.nanmax(np.abs(traces))
        scale_embedding = np.max(loc_embed, axis=0) - np.min(loc_embed, axis=0)
        if highlight is None:
            for i, xy in enumerate(loc_embed):
                plt.plot(loc_embed[i, 0] + np.arange(traces.shape[1]) * scale_embedding[0] / traces.shape[1] / 20,
                         loc_embed[i, 1] + traces[i, :] / scale_trace * scale_embedding[1] / 20,
                         color = color[i])
        else:
            for i, xy in enumerate(loc_embed):
                if highlight[i] == True:
                    plt.plot(loc_embed[i, 0] + np.arange(traces.shape[1]) * scale_embedding[0] / traces.shape[1] / 20,
                             loc_embed[i, 1] + traces[i, :] / scale_trace * scale_embedding[1] / 20,
                             color=color[i], linewidth=2)
        h_scatter.set_sizes(h_scatter.get_sizes()/2)

    if highlight is not None:
        h_scatter.set_sizes(h_scatter.get_sizes()+ 4*h_scatter.get_sizes()*highlight)


    """ label shown by mouse clicking """
    if labels_interactive is not None:
        h_text = plt.annotate('', xy=loc_embed[0])

        def onpick(event, h_text=h_text):
            ind = event.ind
            for i in ind:
                h_text.set_text('{}'.format(labels_interactive[i]))
                h_text.set_x(loc_embed[i][0])
                h_text.set_y(loc_embed[i][1])
            plt.show()

        fig = plt.gcf()
        fig.canvas.mpl_connect('pick_event', onpick)



""" ========== ========= tool functions ========== ========== """



def get_unique_elements(labels):
    return sorted(list(set(labels)))



def add_axes_on_top(h_axes, r=0.25):
    """
    tool funciton to add an axes on the top of the existing axis, used by funciton PsthPlot

    :param h_axes: the curret axex handle
    :param r:      ratio of the height of the newly added axes
    :return:       axis handle
    """
    # get axes position
    axes_rect = h_axes.get_position()
    # make the curretn axes smaller
    h_axes.set_position([axes_rect.x0, axes_rect.y0, axes_rect.width, axes_rect.height * (1-r)])
    # add a new axes
    h_axes_top = h_axes.figure.add_axes([axes_rect.x0, axes_rect.y0+axes_rect.height*(1-r), axes_rect.width, axes_rect.height*r], sharex=h_axes)
    h_axes_top.invert_yaxis()
    # h_axes_top.set_xticklabels({})
    try:    # for differenet versions of matploblib
        try:
            h_axes_top.set_facecolor([0.95, 0.95, 0.95])
        except:
            h_axes_top.set_axis_bgcolor([0.95, 0.95, 0.95])
    except:
        pass

    return h_axes_top


def add_sub_axes(h_axes=None, loc='top', size=0.25, gap=0.02, sub_rect = None):
    """
    tool funciton to add an axes around the existing axis

    :param h_axes: the current axes handle, default to None, use the gca
    :param loc:    location of the newly added sub-axes: one of ['top', 'bottom', 'left', 'right', 'custom'], default to 'top'

                    - if one of ['top', 'bottom', 'left', 'right'], the size of sub axes is determined by size and gap parameter;
                    - if set to 'custom', the location and size if specifited by sub_rect parameter
    :param size:   size of the sub-axes, with respect to the origial axes, default to 0.25
    :param gap:    gap between the original axes and and the newly added sub-axes
    :param sub_rect: the rect of custom sub-axes, rect = [x_left, y_bottom, ]
    :return:       handle of sub axes
    """
    if h_axes is None:
        h_axes = plt.gca()
    # get axes position
    axes_rect = h_axes.get_position()
    x0, y0, width, height = axes_rect.x0, axes_rect.y0, axes_rect.width, axes_rect.height


    # set modefied axes and new sub-axes position
    if sub_rect is not None:
        loc = 'custom'
    if loc == 'top':
        x0_new, y0_new, width_new, height_new = x0, y0, width, height * (1 - size - gap)
        x0_sub, y0_sub, width_sub, height_sub = x0, y0+height * (1 - size), width, height * size
        sharex, sharey = h_axes, None
    elif loc == 'bottom':
        x0_new, y0_new, width_new, height_new = x0, y0 + height * (size + gap), width, height * (1 - size - gap)
        x0_sub, y0_sub, width_sub, height_sub = x0, y0, width, height * size
        sharex, sharey = h_axes, None
    elif loc == 'left':
        x0_new, y0_new, width_new, height_new = x0 + width * (size + gap), y0, width * (1 - size - gap), height
        x0_sub, y0_sub, width_sub, height_sub = x0, y0, width * size, height
        sharex, sharey = None, h_axes
    elif loc == 'right':
        x0_new, y0_new, width_new, height_new = x0, y0, width * (1 - size - gap), height
        x0_sub, y0_sub, width_sub, height_sub = x0 + width * (1 - size), y0, width * size, height
        sharex, sharey = None, h_axes
    elif loc == 'custom':
        x0_rel, y0_rel, width_rel, height_rel = sub_rect
        x0_new, y0_new, width_new, height_new = x0, y0, width, height
        x0_sub, y0_sub, width_sub, height_sub = x0 + x0_rel * width, y0 + y0_rel * height, width * width_rel, height * height_rel
        sharex, sharey = None, None
    else:
        warnings.warn('loc has to be one of "top", "bottom", "left", "right", or "custom"')
        return None

    # make the curretn axes smaller
    h_axes.set_position([x0_new, y0_new, width_new, height_new])
    # add a new axes
    h_subaxes = h_axes.figure.add_axes([x0_sub, y0_sub, width_sub, height_sub], sharex=sharex, sharey=sharey)

    # adjust tick labels
    if loc == 'top':
        plt.setp(h_subaxes.get_xticklabels(), visible=False)
    elif loc == 'bottom':
        plt.setp(h_axes.get_xticklabels(), visible=False)
    elif loc == 'left':
        plt.setp(h_axes.get_yticklabels(), visible=False)
    elif loc == 'right':
        plt.setp(h_subaxes.get_yticklabels(), visible=False)

    return h_subaxes






def create_array_layout_subplots(array_layout, tf_linear_indx=True, tf_text_ch=False):
    """
    create the subplots based on the electrode array's spatial layout

    :param array_layout:   electrode array's spatial layout, a dict, {chan: (row, column)}
    :param tf_linear_indx: True/False the returned subplot axis is a 1D array, indexed in the channel orders, default to True
    :param tf_text_ch:     True/False show the channel index for every axes
    :return: [h_fig, h_axes],as the plt.subplots
    """
    [ch, r, c] = zip(*sorted([[ch, r, c] for ch, (r, c) in array_layout.items()]))
    max_r = max(r)
    max_c = max(c)

    # create subplots
    [h_fig, h_axes] = plt.subplots(max_r+1, max_c+1, sharex=True, sharey=True)
    h_axes = np.array(h_axes, ndmin=2)

    # set all axes off
    for r_cur in r:
        for c_cur in c:
            h_axes[r_cur, c_cur].set_axis_off()

    # for the valid axes, set them on, and create a list of axes according to the order of channels
    h_axes_linear = []
    for ch_cur, ch_loc  in array_layout.items():
        h_axes_linear.append(h_axes[ch_loc])
        h_axes[ch_loc].set_axis_on()

        if tf_text_ch:
            plt.axes(h_axes[ch_loc])
            plt.text(0.02, 0.98, 'Ch {}'.format(ch_cur), transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # if true, return the axes according to the channel, otherwise, returns 2D axes array
    if tf_linear_indx:
        h_axes = h_axes_linear

    return [h_fig, h_axes]


def AutoRowCol(N, nrow=None, ncol=None, aspect=1.0):
    """
    tool function to automatically calculate number of row and col based on total number of panels

    :param N:      total number of panels
    :param nrow:   optional, the desired number of rows, if given, fix the number of rows
    :param ncol:   optional, the desired number of columns, if given, fix the number of columns
    :param aspect: desired aspect ratio: ncol/nrow, default to 1/1
    :return:       (nrow, ncol)
    """

    if nrow is not None:
        ncol, rem = divmod(int(N), int(nrow))
        if rem > 0:
            ncol += 1
    elif ncol is not None:
        nrow, rem = divmod(int(N), int(ncol))
        if rem > 0:
            nrow += 1
    else:
        ncol = int(np.ceil(np.sqrt(N * 1.0 * aspect)))
        nrow, rem = divmod(int(N), int(ncol))
        if rem > 0:
            nrow += 1
    return nrow, ncol


def SubplotsAutoRowCol(num_panels, nrow=None, ncol=None, aspect=1.0, **kwargs):
    """
    tool function to create subplots using the number of panels, a wrapper of plt.subplots() and AutoRowCol()

    the function will automatically decide the number of rows and columns, and the returned h_axes is a linear array .

    :param num_panels: total number of subplot panels
    :param nrow:       optional, the desired number of rows,
    :param ncol:       optional, the desired number of columns,
    :param aspect:     optional, the desired ratio: ncol/nrow, default to 1/1
    :param kwargs:     other arguments to pass to plt.subplots()
    :return:           h_fig, h_axes,  where h_axes is a linear array of axes
    """

    # get the number of rows and columns
    nrow, ncol = AutoRowCol(num_panels, nrow=nrow, ncol=ncol, aspect=aspect)

    kwargs['squeeze'] = False
    h_fig, h_axes = plt.subplots(nrow, ncol, **kwargs)
    h_axes = np.ravel(h_axes)    # make it a 1D array
    return h_fig, h_axes


def DataFastSubplot(data_list, layout=None, data_type=None, gap = 0.05, tf_axis=True, tf_label=True,
                    tf_nmlz = True, xx=None, yy=None):

    N_data = len(data_list)

    if data_type is None:
        data_type = 'line'

    if layout is None:             # if not give, set layout to be the number of data
        layout = N_data
    if np.array(layout).size==1:   # if is a single number, turn to [num_row, num_col]
        ncol = int(np.ceil(np.sqrt(layout)))
        nrow = int(np.ceil(1.0*layout/ncol))
        layout = (nrow, ncol)

    plt.axes([0.05, 0.02, 0.94, 0.90])

    if data_type == 'mesh':
        """ get the sizes """
        ny_mesh, nx_mesh = data_list[0].shape    # size of every panel
        nx_gap = int(np.ceil(nx_mesh * gap))     # size of gap between panels
        ny_gap = int(np.ceil(ny_mesh * gap))
        nx_cell = nx_mesh+nx_gap                 # a cell is panel with gap
        ny_cell = ny_mesh+ny_gap
        nx_shift = nx_gap//2                     # shift the starting point of panel to be half the gap
        ny_shift = ny_gap // 2
        nx_canvas = (nx_mesh + nx_gap) * ncol    # size of canvas
        ny_canvas = (ny_mesh + ny_gap) * nrow

        # initialize the data for canvas
        mesh_canvas = np.zeros([ny_canvas, nx_canvas])*np.nan
        # contains the data for mask (create frames between panels)
        mask_canvas = np.zeros([ny_canvas, nx_canvas])

        if xx is None:
            xx = np.arange(nx_mesh)
        if yy is None:
            yy = np.arange(ny_mesh)


        """ normalize individual mesh plot to range [0,1] before putting them together """
        if tf_nmlz:
            for i in range(N_data):
                cur_min = np.nanmin(data_list[i])
                cur_max = np.nanmax(data_list[i])
                data_list[i] = (data_list[i] - cur_min) / (cur_max - cur_min)

        """ put mesh plot together into the big canvas matrix """
        def indx_in_canvas(indx, rowcol='row', startend='start'):
            # function to compute the index on cavas
            if rowcol == 'row':
                n_cell = ny_cell
                n_shift  = ny_shift
                n_mesh = ny_mesh
            else:
                n_cell = nx_cell
                n_shift = nx_shift
                n_mesh = nx_mesh
            return indx * n_cell + n_shift + n_mesh * (startend=='end')

        def map_value_in_cavas(values, row=0, col=0, xy='x'):
            if xy=='x':
                range_value = (np.min(xx), np.max(xx))
                n_mesh = nx_mesh
                indx = col
                rowcol = 'col'
            elif xy=='y':
                range_value = (np.min(yy), np.max(yy))
                n_mesh = ny_mesh
                indx = row
                rowcol = 'row'
            return 1.0*(np.array(values) - range_value[0]) / (range_value[1]-range_value[0]) \
                   * (n_mesh-1) + indx_in_canvas(indx, rowcol)

        for i in range(N_data):
            row = i // ncol    # row index of panel
            col = i %  ncol    # col index of panel
            # fill the mesh data of the panel in to the right location of canvas matrix for mesh plot
            mesh_canvas[indx_in_canvas(row, 'row','start') : indx_in_canvas(row, 'row','end'),
                        indx_in_canvas(col, 'col', 'start') : indx_in_canvas(col, 'col','end')] = data_list[i]
            # fill value 1.0 to the pixels that contains mesh data in the canvas matrix for mask
            mask_canvas[indx_in_canvas(row, 'row','start') : indx_in_canvas(row, 'row','end'),
                        indx_in_canvas(col, 'col', 'start') : indx_in_canvas(col, 'col','end')] = 1

        # create colormap for mask (transparent if 0.0, opaque if 1.0)
        cmap_mask = mpl.colors.LinearSegmentedColormap.from_list('cmap_mask', [(0.9, 0.9, 0.9, 1.0), (0.9, 0.9, 0.9, 0.1)], N=2)

        """ plot big matrix containing all mesh plots """
        # # mesh data
        plt.imshow(mesh_canvas, vmin=np.nanmin(mesh_canvas), vmax=np.nanmax(mesh_canvas), cmap='inferno', aspect='auto')
        # # maks that forms the frames that seperates data panels
        plt.imshow(mask_canvas, vmin=0, vmax=1, cmap=cmap_mask, aspect='auto')

        """ make plot look better """
        # set y axis direction in the imshwow format
        ylim =np.array(plt.gca().get_ylim())
        plt.gca().set_ylim( ylim.max(), ylim.min() )
        plt.axis('off')

        """ plot panel axis and ticks """
        if tf_axis:
            for i in range(N_data):
                row = i // ncol    # row index of panel
                col = i %  ncol    # col index of panel
                # axis line
                plt.plot([indx_in_canvas(col, 'col', 'start'), indx_in_canvas(col, 'col', 'end') - 1],
                         [indx_in_canvas(row, 'row', 'end') - 0.5, indx_in_canvas(row, 'row', 'end') - 0.5],
                         'k-', alpha=0.5)
                plt.plot([indx_in_canvas(col, 'col', 'start') -0.5, indx_in_canvas(col, 'col', 'start') -0.5],
                         [indx_in_canvas(row, 'row', 'start') , indx_in_canvas(row, 'row', 'end') - 1],
                         'k-', alpha=0.5)
                # axis tick
                plt.vlines(map_value_in_cavas(auto_tick(xx), row, col, xy='x'), indx_in_canvas(row, 'row', 'end') - 0.5,
                           indx_in_canvas(row, 'row', 'end') - 0.5 + ny_gap / 4.0, alpha=0.5)
                plt.hlines(map_value_in_cavas(auto_tick(yy), row, col, xy='y'), indx_in_canvas(col, 'col', 'start') - 0.5,
                           indx_in_canvas(col, 'col', 'start') - 0.5 - nx_gap / 4.0, alpha=0.5)

        """ plot labels """
        if tf_label:
            for i in range(N_data):
                row = i // ncol    # row index of panel
                col = i %  ncol    # col index of panel
                plt.text(indx_in_canvas(col, 'col', 'start'), indx_in_canvas(row, 'row', 'start')-0.5, '')




def center2edge(centers):
    # tool function to get edges from centers for plt.pcolormesh
    centers = np.array(centers,dtype='float')
    edges = np.zeros(len(centers)+1)
    if False:     # assumes even spacing
        if len(centers) is 1:
            dx = 1.0
        else:
            dx = centers[-1]-centers[-2]
        edges[0:-1] = centers - dx/2
        edges[-1] = centers[-1]+dx/2
    if True:      # more universal
        if len(centers) is 1:
            dx = 1.0
            edges[0] = centers - dx / 2
            edges[1] = centers + dx / 2
        else:
            edges[1:-1] = (centers[0:-1] + centers[1:])/2.0
            edges[0]  = centers[0]  - (edges[1]- centers[0])
            edges[-1] = centers[-1] + (centers[-1]-edges[-2])
    return edges


def prettyfloat(input, precision=2):
    """
    function to cut the long float numbers for print
    """
    if hasattr(input, '__iter__'):
        output = []
        for x in input:
             output.append( prettyfloat(x, precision) )  # through recursion
    elif isinstance(input, float):
        output = round(input, precision)
    else:
        output = input
    return output


def gen_distinct_colors(n, luminance=0.9, alpha=0.8, style='discrete'):
    """
    tool funciton to generate n distinct colors for plotting

    :param n:          num of colors
    :param luminance:  num between [0,1]
    :param alhpa:      num between [0,1]
    :param style:      sting, 'discrete', or 'continuous'
    :return:           n*4 rgba color matrix
    """

    if style == 'discrete':
        magic_number = 0.618  # from the golden ratio, to make colors evely distributed
        initial_number = 0.25
        colors_ini = plt.cm.rainbow((initial_number + np.arange(n)) * magic_number % 1)
    elif style == 'continuous':
        colors_ini = plt.cm.rainbow( 1.0*np.arange(n)/n )
    else:
        magic_number = 0.618  # from the golden ratio, to make colors evely distributed
        initial_number = 0.25
        colors_ini = plt.cm.rainbow((initial_number + np.arange(n)) * magic_number % 1)
    return colors_ini* np.array([[luminance, luminance, luminance, alpha]])


def keep_less_than(list_in, n=6):
    """
    keep less than n element, through recursion

    :param list_in: input list, 1D
    :param n:       criterion
    :return:        list of a subset of the original list with elemetns smaller than n
    """
    m = len(list_in)
    if m<=n:
        return list_in
    else:
        list_in = [ list_in[i] for i in range(0, m, 2) ]
        return keep_less_than(list_in, n)


def cal_rc(N):
    """
    calculate n_row, n_column automatically, for subplot layout, obsolete, should switch to AutoRowCol

    :return: [n_rows, n_cols]
    """
    n_rows = int(np.ceil(np.sqrt(N)))
    n_cols = int(np.ceil(1.0 * N / n_rows))
    return [n_rows, n_cols]


def isSingle(x):
    """
    Check whether input is does not contain multiple items, works for lists and tuples

    e.g. 1, 3.0, 'a string', [1.0], or ('afe') returns True, [1,2] returns false
    :param x:  list, tuple, string or number
    :return:   Ture of False
    """
    if isinstance(x, (list, tuple)):
        if len(x) > 1:
            return False
        else:
            return isSingle(x[0])
    else:
        return True

def share_clim(h_ax, c_lim=None):
    """
    tool funciton to share clim (make sure c_lim of given axes are the same), call after plotting all images

    :param h_ax: list of axes to reset clim
    :param c_lim: if None, calculate automatically, otherwise, use the given clim, e.g. [-1, 5]
    :return:     c_lim
    """
    h_ax_all = np.array(h_ax).flatten()
    if c_lim is None:    # if not given, calculate the clim to be the smallest possible range to accommodate all axes
        c_lim = [+np.Inf, -np.Inf]
        for ax in h_ax_all:  # get clim
            plt.axes(ax)
            if plt.gci() is not None:
                c_lim_new = plt.gci().get_clim()
                c_lim = [np.min([c_lim[0], c_lim_new[0]]), np.max([c_lim[1], c_lim_new[1]])]
    for ax in h_ax_all:  # set clim
        plt.axes(ax)
        if plt.gci() is not None:
            plt.clim(c_lim)
    return c_lim



def auto_tick(data_range, max_tick=10, tf_inside=False):
    """
    tool function that automatically calculate optimal ticks based on range and the max number of ticks
    :param data_range:   range of data, e.g. [-0.1, 0.5]
    :param max_tick:     max number of ticks, an interger, default to 10
    :param tf_inside:    True/False if only allow ticks to be inside
    :return:             list of ticks
    """
    data_range = np.array(data_range, dtype=float)
    if len(data_range)>2:
        data_range = [data_range.min(), data_range.max()]
    data_span = data_range[1] - data_range[0]
    scale = 10.0**np.floor(np.log10(data_span))    # scale of data as the order of 10, e.g. 1, 10, 100, 0.1, 0.01, ...
    list_tick_size_nmlz = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]   # possible tick sizes for normalized data in range [1, 10]
    tick_size_nmlz = 1.0     # initial tick size for normalized data
    for i in range(len(list_tick_size_nmlz)):                 # every loop reduces tick size thus increases tick number
        num_tick = data_span/scale/list_tick_size_nmlz[i]     # number of ticks for the current tick size
        if num_tick > max_tick:                               # if too many ticks, break loop
            tick_size_nmlz = list_tick_size_nmlz[i-1]
            break
    tick_size = tick_size_nmlz * scale             # tick sizse for the original data
    ticks = np.unique(np.arange(data_range[0]/tick_size, data_range[1]/tick_size).round())*tick_size    # list of ticks

    if tf_inside:     # if only allow ticks within the given range
        ticks = ticks[ (ticks>=data_range[0]) * (ticks<=data_range[1])]

    return ticks



""" ========== ========== obsolete function ========== ========== """

def SignalPlot(ts, data3D, sk_std=np.nan):
    """
    function to generate plot using data3D, (N_trial * N_ts * N_signal)

    :param ts:      timestamps (in sec)
    :param data3D:  data
    :param sk_std:  smooth kernel std (in sec); default is nan, do not smooth data
    :return:        plot handle
    """

    tf_smooth_every_trial = False    # very slow if set to true

    if data3D.ndim==2:
        data2D = data3D
        (N_tr, N_ts) = data2D.shape
        data3D = np.ones([N_tr,N_ts,1])
        data3D[:,:,0] = data2D


    (_, N_ts, N_sign) = data3D.shape

    if sk_std is not np.nan:   # condition for using smoothness kernel
        ts_interval = np.diff(np.array(ts)).mean()        # get sampling interval
        kernel_std  = sk_std/ts_interval                  # std in frames
        kernel_len  = int(np.ceil(kernel_std)*3*2+1)      # num of frames, 3*std on each side, an odd number
        smooth_kernel = sgn.gaussian(kernel_len,kernel_std)
        smooth_kernel = smooth_kernel/smooth_kernel.sum() # normalized smooth kernel

        if tf_smooth_every_trial:
            smooth_kernel_3D = np.zeros([1, kernel_len, 1])
            smooth_kernel_3D[0,:,0] = smooth_kernel
            data3D_smooth = sgn.convolve(data3D, smooth_kernel_3D, 'same')

            data_plot = np.mean(data3D_smooth, axis=0)
        else:
            data_plot = np.zeros([N_ts, N_sign])
            for i in range(N_sign):
                data_plot[:,i] = np.convolve(np.mean(data3D, axis=0)[:, i], smooth_kernel, 'same')
    else:
        data_plot = np.mean(data3D, axis=0)


    name_colormap = 'rainbow'
    cycle_color = plt.cm.get_cmap(name_colormap)(np.linspace(0, 1, N_sign))
    cycle_linestyle = ['-']

    hl_plot = []   # handle list
    for i in range(N_sign):
        plt.plot()
        h_plot, = plt.plot(ts, data_plot[:,i], c=cycle_color[i%len(cycle_color)]*0.9, linestyle=cycle_linestyle[i%len(cycle_linestyle)], linewidth=2 )
        hl_plot.append(h_plot)
        plt.xlim(ts[0], ts[-1])

    return hl_plot


def NeuroPlot(data_neuro, layout=[], sk_std=np.nan, tf_seperate_window=False, tf_legend=True):

    [_,_,N_sgnl] = data_neuro['data'].shape

    if tf_seperate_window:
        lh_fig = []
        for i in range(N_sgnl):
            lh_fig.append(plt.figure(figsize=(16,9)))
    else:
        fig = plt.figure(figsize=(16, 9))

    if 'cdtn' in data_neuro.keys():
        N_cndt = len(data_neuro['cdtn'])
        if len(layout) == 0:
            N_col = np.ceil(np.sqrt(N_cndt)).astype(int)
            N_row = np.ceil(1.0*N_cndt/N_col).astype(int)
        else:
            [N_row, N_col] = layout

        axes1 = plt.subplot(N_row, N_col, 1)
        for i in range(N_cndt):
            cdtn = data_neuro['cdtn'][i]
            if tf_seperate_window:
                for j in range(N_sgnl):
                    plt.figure(lh_fig[j].number)
                    plt.subplot(N_row, N_col, i + 1, sharey=axes1)
                    hl_plot = SignalPlot(data_neuro['ts'], data_neuro['data'][data_neuro['cdtn_indx'][cdtn], :, [j]], sk_std)
                    plt.title(prettyfloat(cdtn))
                    if i==0:
                        lh_fig[j].suptitle(data_neuro['signal_info']['name'][j])
            else:
                plt.figure(fig.number)
                plt.subplot(N_row, N_col, i + 1, sharey=axes1)
                hl_plot = SignalPlot(data_neuro['ts'], data_neuro['data'][data_neuro['cdtn_indx'][cdtn], :, :], sk_std)
                if tf_legend:
                    plt.legend(hl_plot, data_neuro['signal_info']['name'].tolist(), labelspacing=0.1, prop={'size':8}, fancybox=True, framealpha=0.5)
                    plt.title(prettyfloat(cdtn))
    else:
        hl_plot = SignalPlot(data_neuro['ts'], data_neuro['data'], sk_std )
        # plt.legend(hl_plot, data_neuro['signal_info']['name'].tolist())
        plt.title(prettyfloat(cdtn))

    return 1