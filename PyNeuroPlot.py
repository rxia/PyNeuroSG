import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
import mpl_toolkits
import pandas as pd
import numpy as np
import scipy as sp
from scipy import signal as sgn
import re
import warnings
import misc_tools
import signal_align

def PyNeuroPlot(df, y, x, c=[], p=[]):
    df_plot = pd.DataFrame()

    if len(c)==0:
        df_plot = df.groupby(x) [y].agg(np.mean)
        df_plot.plot(kind='bar',title= y )
    else:
        catg = sorted(df[c].unique())
        for i in range(len(catg)):
            df_plot[catg[i]] = df [df[c]==catg[i]].groupby(x) [y].agg(np.mean)
        df_plot.plot(kind='bar',title= y )
    plt.legend(bbox_to_anchor=(1.1, 1.1), fancybox=True, framealpha=0.5)
    plt.gca().get_legend().set_title(c)

    return 1


def SpkWfPlot(seg, sortcode_min =1, sortcode_max =100, ncols=8):
    """
    Plot spk waveforms of a segment, one channel per axes, different sort codes are color coded

    :param seg:          neo blk.segment
    :param sortcode_min: the min sortcode included in the plot, default to 1
    :param sortcode_max: the max sortcode included in the plot, default to 1
    :param ncols:        number of columns in the subplot
    :return:             figure handle
    """
    
    N_chan = max([item.annotations['channel_index'] for item in seg.spiketrains])   # ! depend on the frame !
    nrows  = int(np.ceil(1.0 * N_chan / ncols))
    # spike waveforms:
    fig, axes2d = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    axes1d = [item for sublist in axes2d for item in sublist]  # flatten axis
    sortcode_color = {0: np.array([225, 217, 111]) / 255.0,
                      1: np.array([149, 196, 128]) / 255.0,
                      2: np.array([112, 160, 234]) / 255.0,
                      3: np.array([142, 126, 194]) / 255.0,
                      4: np.array([202, 066, 045]) / 255.0,
                      5: np.array([229, 145, 066]) / 255.0}
    for i, axes in enumerate(axes1d):
        plt.sca(axes)
        plt.xticks([])
        plt.yticks([])
        plt.text(0.1, 0.8, 'C{}'.format(i + 1), transform=axes.transAxes)
        axes.set_axis_bgcolor([0.98,0.98,0.98])
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
            h_text = plt.text(0.5, 0.12 * cur_code, 'N={}'.format(len(seg.spiketrains[i])), transform=axes_cur.transAxes, fontsize='smaller')
            h_waveform = plt.plot(np.squeeze(np.mean(seg.spiketrains[i].waveforms, axis=0)))
            if cur_code >= 0 and cur_code <= 5:
                h_waveform[0].set_color(sortcode_color[cur_code])
                h_text.set_color(sortcode_color[cur_code])
    axes.set_ylim( np.max(np.abs(np.array(axes.get_ylim())))*np.array([-1,1]) )    # make y_lim symmetrical
    fig.suptitle('spike waveforms, y_range={} uV'.format(np.round(np.diff(np.array(axes.get_ylim())) * 1000000)[0] ))

    return fig

def ErpPlot(array_erp, ts, array_layout=None, depth_start=0, depth_incr=0.1):
    """
    ERP (event-evoked potential) plot

    :param array_erp:     a 2d numpy array ( N_chan * N_timepoints )
    :param ts:            a 1d numpy array ( N_timepoints )
    :param array_layout:  electrode array layout:

                          * if None, assume linear layout,
                          * otherwise, use the the give array_layout in the format: {chan: (row, col)}
    :param depth_start:   starting depth of channel 1
    :param depth_incr:    depth increment
    :return:              figure handle
    """

    [N_chan, N_ts] = array_erp.shape
    if array_layout is None:                # if None, assumes linear layout
        offset_plot_chan = (array_erp.max()-array_erp.min())/5

        array_erp_offset = (array_erp - np.array(np.arange(array_erp.shape[0]), ndmin=2).transpose() * offset_plot_chan).transpose()

        name_colormap = 'rainbow'
        cycle_color = plt.cm.get_cmap(name_colormap)(np.linspace(0, 1, N_chan))

        h_fig = plt.figure(figsize=(12,8))
        plt.subplot(1,2,1)
        for i in range(N_chan):
            plt.plot(ts, array_erp_offset[:,i], c=cycle_color[i]*0.9, lw=2)
        plt.gca().set_axis_bgcolor([0.95, 0.95, 0.95])
        plt.xlim(ts[0],ts[-1])
        plt.ylim( -(N_chan+2)*offset_plot_chan, -(0-3)*offset_plot_chan,  )
        plt.title('ERPs')
        plt.xlabel('time from event onset (s)')
        plt.ylabel('Voltage (V)')

        plt.subplot(1, 2, 2)
        plt.pcolormesh(center2edge(ts), center2edge(np.arange(N_chan)+1) , np.array(array_erp), cmap=plt.get_cmap('coolwarm'))
        color_max = np.max(np.abs(np.array(array_erp)))
        plt.clim(-color_max, color_max)
        plt.xlim(ts[0], ts[-1])
        plt.ylim( center2edge(np.arange(N_chan)+1)[0], center2edge(np.arange(N_chan)+1)[-1]  )
        plt.gca().invert_yaxis()
        plt.title('ERPs')
        plt.xlabel('time from event onset (s)')
        plt.ylabel('channel index')
    else:                                 # use customized 2D layout
        text_props = dict(boxstyle='round', facecolor='w', alpha=0.5)
        [h_fig, h_axes] = create_array_layout_subplots(array_layout)
        plt.tight_layout()
        h_fig.subplots_adjust(hspace=0.02, wspace=0.02)
        h_fig.set_size_inches([8, 8],forward=True)

        for ch in range(N_chan):
            plt.axes(h_axes[array_layout[ch + 1]])
            plt.plot(ts, array_erp[ch, :], linewidth=2, color='dimgray')
            plt.text(0.1, 0.8, 'C{}'.format(ch + 1), transform=plt.gca().transAxes, fontsize=10, bbox=text_props)
        plt.xlim(ts[0], ts[-1])

        # axis appearance
        for ax in h_axes.flatten():
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if len(ax.lines)==0:
                ax.set_axis_bgcolor([1,1,1,0])
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

        plt.suptitle('ERPs', fontsize=18)

    return h_fig


def RfPlot(data_neuro, indx_sgnl=0, x_scale=0.1, y_scale=50):
    """
    plot RF using one single plot, the data_neuro['cdtn'] contains 'x' and 'y', which represents the location of stimulus
    """
    plt.figure()
    plt.suptitle(data_neuro['signal_info'][indx_sgnl]['name'])
    for i, xy in enumerate(data_neuro['cdtn']):
        plt.fill_between( xy[0]+np.array(data_neuro['ts'])/x_scale, xy[1], xy[1]+np.mean( data_neuro['data'][ data_neuro['cdtn_indx'][data_neuro['cdtn'][i]] ,:,indx_sgnl], axis=0 )/y_scale, color=[0,0,0,0.5] )
        plt.plot(xy[0],xy[1],'ro', linewidth=10)



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
        [n_rows, n_cols] = cal_rc(N_cdtn)
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
        data = np.take(np.array(data), limit, axis=0)

    cdtn_unq = None
    if len(data.shape) ==3:     # if data is 3D [N_trials * N_ts * N_signals]
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

    elif len(data.shape) ==2:   # if data is 2D [N_trials * N_ts]
        data2D = data
        [N, T] = np.shape(data2D)
        if cdtn is None:                         # condition is the tag of the trials
            cdtn = ['']*N
        if limit is not None:
            cdtn   = cdtn[limit]
    else:
        raise( Exception('input data does not have the right dimension') )

    cdtn = np.array(cdtn)
    if cdtn_unq is None:
        cdtn_unq  = get_unique_elements(cdtn)        # unique conditions
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
    plt.text(0.05, 0.9, 'N={}'.format(N), transform=ax_psth.transAxes, verticalalignment='top',bbox=textprops)

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
        [y,x] = zip(*np.argwhere(data2D>0))
        x_ts  = ts[np.array(x)]   # x loc of raster lines
        y     = np.array(y)       # y loc of raster lines
        y_min = y - 0.0005*N
        y_max = y+1 +0.0005*N

        c     = np.array(colors)[ cdtn_indx_of_trial[np.array(y)] ]

        # ----- major plot commmand -----
        h_raster = plt.vlines(x_ts, y_min, y_max, colors=c, linewidth=2)

    elif RasterType == 'LFP':
        # -----  major plot command -----
        # c_axis_abs = np.max(np.abs(data2D))
        c_axis_abs = np.percentile(np.abs(data2D), 98)
        h_raster = plt.pcolormesh(center2edge(ts), range(N + 1), data2D, vmin=-c_axis_abs, vmax=c_axis_abs,
                       cmap=plt.get_cmap('coolwarm'))

        # a colored line segment illustrating the conditions
        plt.vlines(ts[[0] * M], np.insert(N_cdtn_cum, 0, 0)[0:M], N_cdtn_cum, colors=colors, linewidth=10)
    else:
        raise Exception('wrong RasterType, must by either "spk" or "LFP" ')

    # horizontal lines deviding difference conditions
    plt.hlines(N_cdtn_cum, ts[0], ts[-1], color='k', linewidth=1)

    plt.xlim(ts[0],ts[-1])
    plt.ylim(0, N)
    plt.gca().invert_yaxis()
    plt.gca().yaxis.set_ticks(keep_less_than([0]+N_cdtn_cum.tolist(), 8))

    return h_raster

def DataNeuroSummaryPlot(data_neuro, sk_std=None, signal_type='auto', suptitle='', xlabel='', ylabel=''):
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

    name_signals = [x['name'] for x in data_neuro['signal_info']]

    # plot function for every panel
    functionPlot = lambda x: PsthPlot(x, ts=ts, cdtn=name_signals, sk_std=sk_std, tf_legend=True,
                                      xlabel=xlabel, ylabel=ylabel, color_style='continuous')

    # plot every condition in every panel
    h = SmartSubplot(data_neuro, functionPlot, suptitle=suptitle)
    return h


def PsthPlotCdtn(data2D, data_df, ts=None, cdtn_l_name='', cdtn0_name='', cdtn1_name='', limit=None, sk_std=np.nan, subpanel='', tf_legend=False):
    """ Obsolete funciton """

    N_cdtn0 = len(data_df[cdtn0_name].unique())
    N_cdtn1 = len(data_df[cdtn1_name].unique())
    if N_cdtn1 > 1:
        [h_fig, h_ax] = plt.subplots(N_cdtn0, N_cdtn1, figsize=[12, 9], sharex=True, sharey=True)
    else:
        [N_rows, N_cols] = cal_rc(N_cdtn0)
        [h_fig, h_ax] = plt.subplots(N_rows, N_cols, figsize=[12, 9], sharex=True, sharey=True)
    h_ax = np.array([h_ax]).flatten()
    for i_cdtn0, cdtn0 in enumerate(sorted(data_df[cdtn0_name].unique())):
        for i_cdtn1, cdtn1 in enumerate(sorted(data_df[cdtn1_name].unique())):
            plt.axes(h_ax[i_cdtn0 * N_cdtn1 + i_cdtn1])
            PsthPlot(data2D, ts, data_df[cdtn_l_name],
                         limit = np.logical_and(data_df[cdtn0_name] == cdtn0, data_df[cdtn1_name] == cdtn1),
                         tf_legend=tf_legend, sk_std=sk_std, subpanel=subpanel)
            plt.title([cdtn0, cdtn1])
    # plt.suptitle(data_neuro['signal_info'][i_neuron]['name'])




def SpectrogramPlot(spcg, spcg_t=None, spcg_f=None, limit_trial = None, tf_phase =False, tf_log=False, time_baseline=None,
                 t_lim = None, f_lim = None, c_lim = None, name_cmap='inferno',
                 rate_interp=None, tf_colorbar= False):
    """
    plot power spectrogram or coherence-gram, input spcg could be [ N_t * N*f ] or [ N_trial * N_t * N*f ], real or complex

    :param spcg:          2D numpy array, [ N_t * N*f ] or 3D numpy array [ N_trial * N_t * N*f ], either real or complex:

                            * if [ N_trial * N_t * N*f ], average over trial and get [ N_t * N*f ]
                            * if complex, plot spectrogram using its abs value, and plot quiver using the complex value
    :param spcg_t:        tick of time
    :param spcg_f:        tick of frequency
    :param limit_trial:   index array to specify which trials to use
    :param tf_phase:      true/false, plot phase using quiver (spcg has to be complex): points down if negative phase (singal1 lags signal0 for coherence)
    :param tf_log:        true/false, use log scale
    :param time_baseline: baseline time period to be subtracted
    :param name_cmap:     name of color map to use
    :param rate_interp:   rate of interpolation for plotting
    :param tf_colorbar:   true/false, plot colorbar
    :return:              figure handle
    """

    if tf_log:                   # if use log scale
        spcg = np.log(spcg * 10**6 + 10**(-32)) - 6         # prevent log(0) error

    if spcg.ndim == 3:           # if contains many trials, calculate average
        if limit_trial is not None:
            spcg = spcg[limit_trial,:,:]
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
        name_cmap = 'coolwarm'           # use divergence colormap

    # set color limit
    if c_lim is None:
        if time_baseline is not None:    # if use reference period
            c_lim = [-np.max(np.abs(spcg)), np.max(np.abs(spcg))]
        else:
            if tf_log is True:           # if log scale, set c_lim
                c_lim = [np.min(spcg), np.max(spcg)]
            else:                        # if not log scale, set c_lim with lower boundary to be 0
                c_lim = [0, np.max(spcg)]

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

    # set x, y limit
    if t_lim is None:
        t_lim = spcg_t[[0, -1]]
    if f_lim is None:
        f_lim = spcg_f[[0, -1]]

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
            max_quiver = 32    # max number of quivers every dimension (in both axis)
            quiver_scale = 32  # about 1/32 of axis length
            [N_fs, N_ts] = spcg.shape
            indx_fs = np.array(keep_less_than(range(N_fs), max_quiver*1.0*(spcg_f.max()-spcg_f.min())/(f_lim[1]-f_lim[0])))
            indx_ts = np.array(keep_less_than(range(N_ts), max_quiver))
            plt.quiver(spcg_t[indx_ts], spcg_f[indx_fs],
                       spcg_complex[indx_fs, :][:, indx_ts].real, spcg_complex[indx_fs, :][:, indx_ts].imag,
                       color='r', units='height', pivot='mid',
                       scale=np.percentile(spcg, 99.5) * quiver_scale)
        except:
            plt.quiver(spcg_t, spcg_f, spcg_complex.real, spcg_complex.imag,
                       color='r', units='height', pivot='mid', scale=np.percentile(spcg, 99.8) * quiver_scale)

    plt.sci(h_plot) # set the current color-mappable object to be the specrogram plot but not the quiver

    return h_plot


def SpectrogramAllPairPlot(data_neuro, indx_chan=None, limit_gap=1, t_bin=0.2, t_step=None, f_lim = None, coh_lim=None, t_axis=1, batchsize=100, verbose=False):
    """
    Plot all LFP power specgtrogram (diagonal panels) and all pairwise coherence (off-diagonal panels)

    :param data_neuro: standard data input
    :param indx_chan:  index of channels to plot (from zero)
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

    fs = data_neuro['signal_info'][0][2]
    t_ini = np.array(data_neuro['ts'][0])

    # compute spectrogram
    spcg_all = []
    for i_plot, i_chan in enumerate(indx_chan):
        [spcg_cur, spcg_t, spcg_f] = pna.ComputeSpectrogram(data[:,:,i_plot], data1=None, fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step, t_axis=t_axis, batchsize=batchsize)
        spcg_all.append( np.mean(spcg_cur, axis=0) )

    # plot
    text_props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    [h_fig, h_ax] = plt.subplots(nrows=N_plot, ncols=N_plot, sharex=True, sharey=True, figsize=[16,16])
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
                [cohg, _, _] = pna.ComputeCoherogram(data[:, :, indx_row], data[:, :, indx_col], fs=fs, t_ini=t_ini, t_bin=t_bin, t_step=t_step,
                                       t_axis=t_axis, batchsize=batchsize, data0_spcg_ave=spcg_all[indx_row], data1_spcg_ave=spcg_all[indx_col])

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

def get_unique_elements(labels):
    return sorted(list(set(labels)))

def add_axes_on_top(h_axes, r=0.25):
    """
    tool funciton to add an axes on the top of the existing axis

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
    h_axes_top.set_axis_bgcolor([0.95,0.95,0.95])

    return h_axes_top

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




def create_array_layout_subplots(array_layout):
    """
    create the subplots based on the electrode array's spatial layout

    :param array_layout: electrode array's spatial layout, a dict, {chan: (row, column)}
    :return: as the plt.subplots
    """
    [ch, r, c] = zip(*sorted([[ch, r, c] for ch, (r, c) in array_layout.items()]))
    max_r = max(r)
    max_c = max(c)

    [h_fig, h_axes] = plt.subplots(max_r+1, max_c+1, sharex=True, sharey=True)
    return [h_fig, h_axes]

def center2edge(centers):
    # tool function to get edges from centers for plt.pcolormesh
    centers = np.array(centers,dtype='float')
    edges = np.zeros(len(centers)+1)
    if len(centers) is 1:
        dx = 1.0
    else:
        dx = centers[-1]-centers[-2]
    edges[0:-1] = centers - dx/2
    edges[-1] = centers[-1]+dx/2
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
    calculate n_row, n_column automatically, for subplot layout

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

def share_clim(h_ax):
    """
    tool funciton to share clim (make sure c_lim of given axes are the same), call after plotting all images

    :param h_ax: list of axes
    :return:     c_lim
    """
    h_ax_all = np.array(h_ax).flatten()
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

# to test:
if 0:
    import PyNeuroPlot as pnp; reload(pnp); pnp.NeuroPlot(data_neuro)