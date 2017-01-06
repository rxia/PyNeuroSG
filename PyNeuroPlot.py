import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
import mpl_toolkits
import pandas as pd
import numpy as np
from scipy import signal as sgn
import re
# try:
    # import seaborn as sns
# except:
#     print ('no seaborn module installed for plotting, use matplotlib default')


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


def SpkWfPlot(seg, ncols=8):
    """
    Plot spk waveforms of a segment, one channel per axes
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
        cur_chan = int(re.match('Chan(\d*) .*', seg.spiketrains[i].name).group(1))  # ! depend on the naming !
        cur_code = int(re.match('.* Code(\d*)', seg.spiketrains[i].name).group(1))  # ! depend on the naming !
        if cur_code >= 1:
            axes_cur = axes1d[cur_chan - 1];
            plt.sca(axes_cur)
            h_text = plt.text(0.5, 0.12 * cur_code, 'N={}'.format(len(seg.spiketrains[i])), transform=axes_cur.transAxes, fontsize='smaller')
            h_waveform = plt.plot(np.squeeze(np.mean(seg.spiketrains[i].waveforms, axis=0)))
            if cur_code >= 0 and cur_code <= 5:
                h_waveform[0].set_color(sortcode_color[cur_code])
                h_text.set_color(sortcode_color[cur_code])
    axes.set_ylim( np.max(np.abs(np.array(axes.get_ylim())))*np.array([-1,1]) )    # make y_lim symmetrical
    fig.suptitle('spike waveforms, y_range={} uV'.format(np.round(np.diff(np.array(axes.get_ylim())) * 1000000)[0] ))


def ErpPlot(array_erp, ts, depth_start=0, depth_incr=0.1):
    # array_erp   : a 2d numpy array ( N_chan * N_timepoints ):
    # ts          : a 1d numpy array ( N_timepoints )
    # chan_start  : starting channel
    # depth_start : starting depth of channel 1
    # depth_incr  : depth increment

    offset_plot_chan = (array_erp.max()-array_erp.min())/5

    [N_chan, N_ts] = array_erp.shape
    array_erp_offset = (array_erp - np.array(np.arange(array_erp.shape[0]), ndmin=2).transpose() * offset_plot_chan).transpose()

    name_colormap = 'rainbow'
    cycle_color = plt.cm.get_cmap(name_colormap)(np.linspace(0, 1, N_chan))

    fig = plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    for i in range(N_chan):
        plt.plot(ts, array_erp_offset[:,i], c=cycle_color[i]*0.9, lw=2)
    plt.gca().set_axis_bgcolor([0.95, 0.95, 0.95])
    plt.xlim(ts[0],ts[-1])
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
    # plt.colorbar()

    # plt.get_current_fig_manager().window.raise_()
    fig.canvas.manager.window.raise_()
    fig.show()
    fig.savefig('PyNeuroPlot_temp_fig.png')

    return 1


def RfPlot(data_neuro, indx_sgnl=0, x_scale=0.1, y_scale=50):
    """
    plot RF using one single plot, the data_neuro['cdtn'] contains 'x' and 'y', which represents the location of stimulus
    """
    plt.figure()
    plt.suptitle(data_neuro['signal_info'][indx_sgnl]['name'])
    for i, xy in enumerate(data_neuro['cdtn']):
        plt.fill_between( xy[0]+np.array(data_neuro['ts'])/x_scale, xy[1], xy[1]+np.mean( data_neuro['data'][ data_neuro['cdtn_indx'][data_neuro['cdtn'][i]] ,:,indx_sgnl], axis=0 )/y_scale, color=[0,0,0,0.5] )
        plt.plot(xy[0],xy[1],'ro', linewidth=10)


def PsthPlot(data2D, ts=None, cdtn=None, limit=None, sk_std=np.nan, subpanel='', tf_legend=False):
    """
    funciton to plot psth and raster
    :param data: 2D np array, (N_trail * N_ts)
    :param ts:
    :param cdtn:
    :param limit:   index array for selecting trials
    :return:
    """
    ax_psth = plt.gca()
    data2D = np.array(data2D)
    M = 1
    [N, T] = np.shape(data2D)
    ts = np.array(ts)
    if ts is None:
        ts = np.arange( np.size(data2D, axis=1) )
    if cdtn is None:
        cdtn = ['']*N
    cdtn = np.array(cdtn)
    if limit is not None:
        limit = np.array(limit)
        data2D = data2D[limit, :]
        cdtn   = cdtn[limit]
        [N, T] = np.shape(data2D)
    cdtn_unq  = get_unique_elements(cdtn)
    M = len(cdtn_unq)

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

    if sk_std is not np.nan:  # condition for using smoothness kernel
        ts_interval = np.diff(np.array(ts)).mean()  # get sampling interval
        kernel_std = sk_std / ts_interval  # std in frames
        kernel_len = int(np.ceil(kernel_std) * 3 * 2 + 1)  # num of frames, 3*std on each side, an odd number
        smooth_kernel = sgn.gaussian(kernel_len, kernel_std)
        smooth_kernel = smooth_kernel / smooth_kernel.sum()  # normalized smooth kernel
        for k in range(M):
            psth_cdtn[k, :] = np.convolve(psth_cdtn[k, :], smooth_kernel, 'same')

    colors = gen_distinct_colors(M, luminance=0.7)

    # ========== plot psth ==========
    hl_lines = []
    for k, cdtn_k in enumerate(cdtn_unq):
        hl_line, = plt.plot(ts, psth_cdtn[k, :], c=colors[k])  # temp
        hl_lines.append(hl_line)

    ax_psth.set_xlim([ts[0], ts[-1]])
    if tf_legend:
        plt.legend(cdtn_unq, labelspacing=0.1, prop={'size': 8},
                   fancybox=True, framealpha=0.5)

    if subpanel is not '':
        # ========== sub-panel for raster or p-color  ==========
        ax_raster = add_axes_on_top(ax_psth, r=0.4)
        plt.axes(ax_raster)

        if subpanel == 'auto':      # if auto, use the data features to determine raster (spk) plot or pcolor (LFP) plot
            if len(np.unique(data2D)) <=5:
                subpanel = 'raster'
            else:
                subpanel = 'pcolor'

        if subpanel is 'raster':       # ---------- plot raster (spike trains) ----------
            for k, cdtn_k in enumerate(cdtn_unq):
                try:
                    plt.eventplot([ts[data2D[i, :] > 0] for i in np.flatnonzero(cdtn==cdtn_k)],
                              lineoffsets=range(N_cdtn_cum[k], N_cdtn_cum[k+1]),
                              linewidths=2, color=[colors[k]]*N_cdtn[k])
                except:
                    print('function PsthPlot(), can not plot raster')

        elif subpanel is 'pcolor':    # ---------- plot_pcolor (LFP) ----------
            y_axis_abs = np.abs(np.array(ax_psth.get_ylim())).max()
            index_reorder = np.concatenate([np.flatnonzero(np.array(cdtn) == cdtn_k) for cdtn_k in cdtn_unq])
            data2D_reorder = data2D[index_reorder,:]
            plt.pcolormesh( center2edge(ts), range(N+1), data2D_reorder, vmin=-y_axis_abs, vmax=y_axis_abs, cmap=plt.get_cmap('coolwarm'))
            for k, cdtn_k in enumerate(cdtn_unq):
                plt.plot( ts[[0,0]]+(ts[-1]-ts[0])/20,   N_cdtn_cum[k:k+2], color=colors[k], linewidth=5 )
                # ax_raster.add_patch(plt.Rectangle([ts[0], N_cdtn_cum[k]], 0, 1, facecolor=colors[k]))
                plt.plot( [ts[0], ts[-1]], [N_cdtn_cum[k],N_cdtn_cum[k]], color='k', linewidth=2)
        ax_raster.set_ylim([0,N])
        ax_raster.set_xlim([ts[0], ts[-1]])
        ax_raster.yaxis.set_ticks( keep_less_than(N_cdtn_cum[1:]) )


    # data_plot = np.zeros([N_ts, N_sign])
    # for i in range(N_sign):
    #     data_plot[:, i] = np.convolve(np.mean(data3D, axis=0)[:, i], smooth_kernel, 'same')


def PsthPlotCdtn(data2D, data_df, ts=None, cdtn_l_name='', cdtn0_name='', cdtn1_name='', limit=None, sk_std=np.nan, subpanel='', tf_legend=False):
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


def get_unique_elements(labels):
    return sorted(list(set(labels)))

def add_axes_on_top(h_axes, r=0.25):
    """
    tool funciton to add an axes on the top of the existing axis
    :param h_axes: the curret axex handle
    :param r:      ratio of the height of the newly added axes
    :return:
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
    :return:
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


def gen_distinct_colors(n, luminance=0.9):
    """
    tool funciton to generate n distinct colors for plotting
    :param n:          num of colors
    :param luminance:  num between [0,1]
    :return:           n*4 rgba color matrix
    """
    magic_number = 0.618   # from the golden ratio, to make colors evely distributed
    initial_number = 0.25
    return plt.cm.rainbow( (initial_number+np.arange(n))*magic_number %1 )*luminance


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

# to test:
if 0:
    import PyNeuroPlot as pnp; reload(pnp); pnp.NeuroPlot(data_neuro)