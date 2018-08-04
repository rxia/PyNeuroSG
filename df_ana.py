"""
functions related to df (pandas DataFrames): selection, grouping, plotting
"""

import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import misc_tools as misc_tools
import PyNeuroAna as pna
import PyNeuroPlot as pnp


mpl.style.use('ggplot')


def GroupPlot(values, x=None, c=None, p=None, limit=None, plot_type=None, tf_legend=True, tf_count=True,
              values_name='', x_name='', c_name='', p_name='', title_text='', **kwargs):
    """
    function to plot values according to multiple levels of grouping keys: x, c, and p.
    The input values, x, c and p can be viewed as columns from the same table

    Example usage could be found in show_case/GroupPlot_DfPlot.py

    :param values:      values to plot, array of length N
    :param x:           grouping key, plot as x axis, array of length N, default to None.

                        * if x is continuous, plot_type can be one of ['dot','line'], plot values against x
                        * elif x is discrete, plot_type can be one of ['bar','box','violin'], plot values groupped by x
    :param c:           grouping key, plot as separate colors within panel, array of length N, default to None
    :param p:           grouping key, plot as separate panels, array of length N, default to None
    :param limit:       used to select a subset of data, boolean array of length N
    :param plot_type:   the type of plot, one of ['dot', 'line', 'bar', 'box', 'violin'], if None, determined automatically:

                        * 'dot': values against x;
                        * 'line': values against x;
                        * 'bar', mean values with errbar determined by errbar;
                        * 'box': median as colored line, 25%~75% quantile as box, mean as cross, outliers as circles;
                        * 'violin': median and distribution of values
    :param tf_legend:   True/False flag, whether plot legend
    :param tf_count:    Trur/False flag, whether to show count of values for plot type ['bar', 'box', 'violin']
    :param values_name: text on plot, for values
    :param x_name:      text on plot, for x
    :param c_name:      text on plot, for colors
    :param p_name:      text on plot, for panels
    :param title_text:  text for title
    :param errbar:      type of error bar, only used for bar plot, one of ['auto', 'std', 'se', 'binom', ''], default to auto:

                        * 'auto':  if values are binary, use binom, otherwise, use se
                        * 'std':   standard deviation
                        * 'se':    standard error
                        * 'binom': binomial distribution confidence interval based on binom_alpha
                        * '':      do not use error bar
    :param binom_alpha: alpha value for binomial distribution error bar (hpyothesis test for binary values), default = 0.05
    :return:            handles of axes
    """

    """ set default values for inputs """
    N = len(values)
    if x is None:
        x = np.array(['']*N)
    else:
        x = np.array(x)
    if c is None:
        c = np.array(['']*N)
    else:
        c = np.array(c)
    if p is None:
        p = np.array(['']*N)
    else:
        p = np.array(p)
    if limit is None:
        limit = np.ones(N, dtype=bool)
    else:
        values = values[limit]
        x = x[limit]
        c = c[limit]
        p = p[limit]

    errbar = 'auto'
    binom_alpha = 0.05
    for arg_name, arg_value in kwargs.items():
        if arg_name == 'errbar':
            errbar = arg_value
        elif arg_name == 'binom_alpha':
            binom_alpha = arg_value

    """ number of conditions to seperate data with """
    values_unq = np.unique(values)
    x_unq = np.unique(x)
    c_unq = np.unique(c)
    p_unq = np.unique(p)
    Nvalues = len(values_unq)
    Nx = len(x_unq)
    Nc = len(c_unq)
    Np = len(p_unq)

    if errbar == 'auto':
        errbar = 'binom' if Nvalues <=2 else 'se'

    """ determine plot style if not specified """
    if plot_type is None:
        if Nx <= 0.1*N:          # if x is discrete
            if Nvalues <=2:       # if binary data, bar plot, error bar using binomial distribution for errarbar
                plot_type = 'bar'
            else:
                plot_type = 'box' # if continuous data, box plot, error bar using standard error
        else:                 # if x is continuous
            plot_type = 'dot'

    if plot_type in ['bar', 'box', 'violin']:
        plot_supertype = 'discrete'
    elif plot_type in ['dot', 'line']:
        plot_supertype = 'continuous'
    else:
        plot_supertype = ''
        warnings.warn('plot_type not supported, must be one of {}'.format(['dot','line', 'bar', 'box', 'violin']))

    if Nc <=1:
        tf_legend = False

    """ if multipanel, open new fig and use subplot; otherwise, just use current axes """
    tf_multipanel = (Np >1)
    if tf_multipanel:
        nrow, ncol = pnp.AutoRowCol(len(p_unq))
        h_fig, h_ax = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True, squeeze=False)
        h_ax = np.ravel(h_ax)
        string_suptitle = '{}   {} by {}'.format(title_text, values_name, p_name) if p_name!='' else '{}   {} '.format(title_text, values_name)
        plt.suptitle(string_suptitle)
    else:
        h_ax = [plt.gca()]
        plt.title('{}    {}'.format(title_text, values_name))

    plt.xlabel(x_name)
    plt.axes(h_ax[0])
    plt.ylabel(values_name)

    """ ----- loops for plot ----- """
    default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    legend_obj = []
    h_text_count = []
    """ for every panel """
    for p_i, p_c in enumerate(p_unq):
        plt.axes(h_ax[p_i])
        if tf_multipanel:
            plt.title(p_c, style='normal', fontsize=10)
        """ for every condition """
        for c_i, c_c in enumerate(c_unq):
            tf_cur = (p==p_c) * (c==c_c)
            """ if x variable is continuous, plot y against x """
            if plot_supertype == 'continuous':
                if plot_type == 'dot':
                    plt.plot(x[tf_cur], values[tf_cur], 'o')
                elif plot_type == 'line':
                    plt.plot(x[tf_cur], values[tf_cur], '-')
                """ otherwise if x variable is discrete, group y by x """
            elif plot_supertype == 'discrete':
                # values_plot is a length Nx list, where every entry is a array containing values under that x condition
                values_by_x = []
                for x_c in x_unq:
                    values_by_x.append(values[tf_cur * (x==x_c)])
                # numbers of values in every x condition
                values_by_x_num = [len(values_by_x_single) for values_by_x_single in values_by_x]
                # width related thing for bar/box/violin
                cell_width = 1.0/(Nc+1)
                x_loc = np.arange(Nx) + cell_width * (-Nc*0.5 + c_i + 0.5 +0.05)
                bar_width = cell_width*0.90
                values_by_x_median = [np.nanmedian(values_by_x_single) for values_by_x_single in values_by_x]
                if plot_type == 'bar':     # if bar plot
                    values_by_x_mean = [np.nanmean(values_by_x_single) for values_by_x_single in values_by_x]
                    if errbar == 'std':
                        values_by_x_std = [np.nanstd(values_by_x_single) for values_by_x_single in values_by_x]
                        values_by_x_err = values_by_x_std
                    elif errbar == 'se':
                        values_by_x_std = [np.nanstd(values_by_x_single) for values_by_x_single in values_by_x]
                        values_by_x_se  = np.array(values_by_x_std)/np.sqrt(np.array(values_by_x_num))
                        values_by_x_err = values_by_x_se
                    elif errbar == 'binom':
                        values_by_x_err = np.array([np.abs(pna.ErrIntvBinom(x=values_by_x_single, alpha=binom_alpha)) for values_by_x_single in values_by_x]).transpose()
                    else:
                        values_by_x_err = 0
                    plt.bar(x=x_loc, height=values_by_x_mean, yerr=values_by_x_err, width=bar_width, error_kw=dict(capsize=2) )
                elif plot_type == 'box':    # if box plot
                    current_color = default_color_cycle[c_i % len(default_color_cycle)]
                    medianprops = dict(linestyle='-', linewidth=4, color=current_color)
                    meanpointprops = dict(marker='x', markeredgecolor=current_color, markersize=10, markeredgewidth=2)
                    flierprops = dict(marker='.', markersize=5)
                    values_by_x_not_nan = [values[np.isfinite(values)] for values in values_by_x]
                    h_box = plt.boxplot(values_by_x_not_nan, positions=x_loc, widths=bar_width, showmeans=True,
                                medianprops=medianprops, meanprops=meanpointprops, flierprops=flierprops)
                    if tf_legend and (p_i==0):
                        legend_obj.append(h_box['medians'][0])
                elif plot_type == 'violin':  # if violin plot
                    values_by_x = [[np.nan, np.nan] if len(values_by_x_single)==0 else values_by_x_single for values_by_x_single in values_by_x]
                    h_vioinplot = plt.violinplot(values_by_x, positions=x_loc, widths=bar_width, showmedians=True, showextrema=False)
                    if tf_legend and (p_i==0):
                        legend_obj.append(h_vioinplot['bodies'][0])
                """ show count of values for every bar/box/violin """
                if tf_count:
                    for txt_loc_x, txt_loc_y, text_str in zip(x_loc, values_by_x_median, values_by_x_num):
                        h_text_count_cur = plt.text(txt_loc_x, txt_loc_y, text_str, ha='center', va='bottom', fontsize='x-small', rotation='vertical')
                        h_text_count.append(h_text_count_cur)
                """ axes look """
                plt.xlim([-0.5,Nx-0.5])
                plt.xticks(np.arange(Nx), x_unq)
    """ plot legend """
    if tf_legend:
        plt.axes(h_ax[0])
        if len(legend_obj)==0:
            plt.legend(c_unq, title=c_name, fontsize='small')
        else:
            plt.legend(legend_obj, c_unq, title=c_name, fontsize='small')
    """ set text for values count to the bottom of plot  """
    if tf_count and (plot_supertype=='discrete'):
        y_lim_min = h_ax[0].get_ylim()[0]
        h_ax[0].text(-0.5, y_lim_min, 'count', va='bottom', fontsize='x-small', rotation='vertical')
        for h_text_count_cur in h_text_count:
            h_text_count_cur.set_y(y_lim_min)

    return h_ax


def DfPlot(df, values, x='', c='', p='', limit=None, plot_type=None, **kwargs):
    """
    function to plot values according to multiple levels of grouping keys: x, c, and p for Pandas DataFrame df.
    This is a wrapper of  function GroupPlot

    Example usage could be found in show_case/GroupPlot_DfPlot.py

    :param values:      name of the column containing values to plot
    :param x:           name grouping key, plot as x axis, array of length N, default to None:

                        * if x is continuous, plot_type can be one of ['dot','line'], plot values against x
                        * elif x is discrete, plot_type can be one of ['bar','box','violin'], plot values groupped by x
    :param c:           name of grouping key, plot as separate colors within panel, array of length N, default to None
    :param p:           name of grouping key, plot as separate panels, array of length N, default to None
    :param limit:       used to select a subset of data, boolean array of length N
    :param plot_type:   the type of plot, one of ['dot', 'line', 'bar', 'box', 'violin'], if None, determined automatically:

                        * 'dot': values against x;
                        * 'line': values against x;
                        * 'bar', mean values with errbar determined by errbar;
                        * 'box': median as colored line, 25%~75% quantile as box, mean as cross, outliers as circles;
                        * 'violin': median and distribution of values
    :param tf_legend:   True/False flag, whether plot legend
    :param tf_count:    Trur/False flag, whether to show count of values for plot type ['bar', 'box', 'violin']
    :param title_text:  text for title
    :param errbar:      type of error bar, only used for bar plot, one of ['std', 'se', 'binom']:

                        * std:   standard deviation
                        * se:    standard error
                        * binom: binomial distribution confidence interval based on binom_alpha
    :param binom_alpha: alpha value for binomial distribution error bar (hpyothesis test for binary values), default = 0.05
    :param errbar:      type of error bar, only used for bar plot, one of ['auto', 'std', 'se', 'binom', ''], default to auto:

                    * 'auto':  if values are binary, use binom, otherwise, use se
                    * 'std':   standard deviation
                    * 'se':    standard error
                    * 'binom': binomial distribution confidence interval based on binom_alpha
                    * '':      do not use error bar
    :return:            handles of axes
    """

    df = df.reset_index()
    df[''] = [''] * len(df)  # empty column, make some default condition easy

    values_name = values
    x_name = x
    c_name = c
    p_name = p

    values_data = df[values]
    x_data = df[x]
    c_data = df[c]
    p_data = df[p]

    # call funciton GroupPlot, inherit arguments
    h_ax = GroupPlot(values=values_data, x=x_data, c=c_data, p=p_data, limit=limit, plot_type=plot_type,
              values_name=values_name, x_name=x_name, c_name=c_name, p_name=p_name, **kwargs)

    return h_ax


def DfGroupby(data_df, groupby='', limit=None, tf_aggregate=False, tf_linearize=False):
    """
    group a pandas DataFrame by 'groupby', and returns the grouped indexes and order in group

    :param data_df:      pandas DataFrame
    :param groupby:      column name(s) to group the data_df, either a str or a list of strings
    :param limit:        a filter on the indexes, either a boolean array or an index array
    :param tf_aggregate: True/False to add a aggregation group (not grouped) for every column
    :param tf_linearize: True/False to linearize the order of groups, ie., turn (3,0) to 4
    :return: {'idx': {group_key: array of trial indexes within group}, 'order': {group_key: order in plot}}
    """

    # convert limit to index array
    if limit is None:
        limit = np.arange(len(data_df))
    elif len(limit) == 0:
        limit = np.array([])
    elif limit[0] in [True, False]:    # boolean array
        limit = misc_tools.index_bool2int(limit)
    limit = np.array(limit)

    if len(limit) == 0:
        return dict(), dict()

    # make sure groupby is a list
    tf_single_groupby = False
    if isinstance(groupby, str):
        groupby = [groupby]
        tf_single_groupby = True

    # group by every column independently
    idx_by_col = []
    ord_by_col = []
    for col in groupby:
        col_idx = dict()
        col_ord = dict()
        count = 0
        if tf_aggregate:                            # add aggregation group
            col_idx['all'] = limit
            col_ord['all'] = count
            count += 1
        col_idx_grpby = data_df.groupby(col).indices       # index of rows in df that fall into every group
        for col_key in col_idx_grpby:         # select/filter using limit, and delete if empty
            cur_idx = np.intersect1d(col_idx_grpby[col_key], limit)  # select/filter
            if len(cur_idx) > 0:           # delete if empty
                col_idx[col_key] = cur_idx
                col_ord[col_key] = count
                count += 1
        idx_by_col.append(col_idx)
        ord_by_col.append(col_ord)

    # get the intersection of columns
    idx_by_grp = dict()
    ord_by_grp = dict()
    for col_idx, col_ord in zip(idx_by_col, ord_by_col):
        if not idx_by_grp:
            idx_by_grp = {(col, ): col_idx[col] for col in col_idx}
            ord_by_grp = {(col, ): (col_ord[col], ) for col in col_ord}
        else:
            idx_by_grp = {cdtn+(col_cur, ): np.intersect1d(idx_by_grp[cdtn], col_idx[col_cur]) \
                          for cdtn in idx_by_grp for col_cur in col_idx}
            ord_by_grp = {cdtn+(col_cur, ): ord_by_grp[cdtn] + (col_ord[col_cur], ) \
                          for cdtn in ord_by_grp for col_cur in col_ord}

    for key in list(idx_by_grp.keys()):   # delete a group if empty
        if len(idx_by_grp[key]) == 0:
            idx_by_grp.pop(key)
            ord_by_grp.pop(key)

    if tf_linearize:        # linearize ord_by_grp if tf_cross==False
        ord_by_grp = dict(zip([key for key, val in sorted(ord_by_grp.items(), key=lambda a: a[1])],
                              range(len(ord_by_grp))))

    if tf_single_groupby:   # if groupby is simply single str, the key should not be a tuple
        idx_by_grp = {key[0]: val for key, val in idx_by_grp.items()}
        ord_by_grp = {key[0]: val[0] for key, val in ord_by_grp.items()}

    return {'idx': idx_by_grp, 'order': ord_by_grp}


def GroupDataNeuro(data_neuro, data_df=None, groupby='', limit=None, tf_aggregate=False, tf_linearize=False):
    """
    group data_neuro using DfGroupby(), place field ('cdtn', 'cdtn_indx', cdtn_order, 'grpby', 'limit') to data_neuro,
    achieve similar functionality with PyNeuroData.NeuroSort

    :param data_df:      pandas DataFrame
    :param groupby:      column name(s) to group the data_df, either a str or a list of strings
    :param limit:        a filter on the indexes, either a boolean array or an index array
    :param tf_aggregate: True/False to add a aggregation group (not grouped) for every column
    :param tf_linearize: True/False to linearize the order of groups, ie., turn (3,0) to 4
    :return:             data_neuro with fields ('cdtn', 'cdtn_indx', cdtn_order, 'grpby', 'limit')
    """

    if data_df is None:
        if (data_neuro is not None) and ('trial_info' in data_neuro):
            data_df = data_neuro['trial_info']
        else:
            raise Exception('data_neuro (containing data_df in field "trial_info") or data_df has to be given')
    DfGroupbyResult = DfGroupby(data_df, groupby=groupby, limit=limit,
              tf_aggregate=tf_aggregate, tf_linearize=tf_linearize)

    data_neuro['grpby'] = groupby
    data_neuro['fltr'] = limit
    data_neuro['cdtn'] = sorted(list(DfGroupbyResult['idx'].keys()))
    data_neuro['cdtn_indx'] = DfGroupbyResult['idx']
    data_neuro['cdtn_order'] = DfGroupbyResult['order']

    return data_neuro