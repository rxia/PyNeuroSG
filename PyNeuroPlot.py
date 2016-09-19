import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import pandas as pd
import numpy as np
from scipy import signal as sgn
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
    plt.legend(bbox_to_anchor=(1.1, 1.1))
    plt.gca().get_legend().set_title(c)

    return 1

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
                    plt.title(cdtn)
                    if i==0:
                        lh_fig[j].suptitle(data_neuro['signal_info']['name'][j])
            else:
                plt.figure(fig.number)
                plt.subplot(N_row, N_col, i + 1, sharey=axes1)
                hl_plot = SignalPlot(data_neuro['ts'], data_neuro['data'][data_neuro['cdtn_indx'][cdtn], :, :], sk_std)
                if tf_legend:
                    plt.legend(hl_plot, data_neuro['signal_info']['name'].tolist(), labelspacing=0.1, prop={'size':8})
                    plt.title(cdtn)
    else:
        hl_plot = SignalPlot(data_neuro['ts'], data_neuro['data'], sk_std )
        # plt.legend(hl_plot, data_neuro['signal_info']['name'].tolist())
        plt.title(cdtn)
    plt.show()

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


# to test:
if 0:
    import PyNeuroPlot as pnp; reload(pnp); pnp.NeuroPlot(data_neuro)