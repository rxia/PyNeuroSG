import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import pandas as pd
import numpy as np
# try:
#     import seaborn as sns
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

    fig = plt.figure(figsize=(6,8))
    plt.plot(ts, ( array_erp - np.array( np.arange(array_erp.shape[0]), ndmin=2).transpose() * offset_plot_chan) .transpose() )
    plt.title('ERPs')
    plt.xlabel('time from event onset (s)')
    plt.ylabel('Voltage (V)')

    # plt.get_current_fig_manager().window.raise_()
    fig.canvas.manager.window.raise_()
    fig.show()
    fig.savefig('PyNeuroPlot_temp_fig.png')

    return 1

def NeuroPlot(data_neuro, layout='ccs'):
    if 'cdtn' in data_neuro.keys():
        fig = plt.figure( figsize=(16,9) )
        plt.plot(data_neuro['ts'], np.mean(data_neuro['data'], axis=0) )
        plt.show()
        # plt.ylim((0,60))
        plt.title(key)
    else:
        fig = plt.figure( figsize=(16,9) )
        plt.plot(data_neuro['ts'], data_neuro['data'] )
        plt.show()
        # plt.ylim((0,60))
        plt.title(key)



# to test:
if 0:
    import PyNeuroPlot as pnp; reload(pnp); pnp.NeuroPlot(data_neuro)