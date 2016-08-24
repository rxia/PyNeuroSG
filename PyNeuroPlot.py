import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import pandas as pd
import numpy as np
try:
    import seaborn as sns
except:
    print ('no seaborn module installed for plotting, use matplotlib default')

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

def RasterPlot(array_raster)
    # array_raster:   1D object array; every entry is a 1D array of spike times
    x = np.zeros(0)
    y = np.zeros(0)
    for i in len(array_raster):
        x = array_raster[i]