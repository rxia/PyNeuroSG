""" showcase for pnp.GroupPlot, pnp.DfPlot """

import os
import sys
import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt

# costom packages
sys.path.append('/shared/homes/sguan/Coding_Projects/PyNeuroSG')
import PyNeuroPlot as pnp


""" ===== test code for GroupPlot ===== """

""" synthetic data """
reload(pnp)
N=600
values = np.random.randn(N)
x_continuous = np.random.rand(N)
x_discrete = np.random.randint(0,4, size=N)
x = x_discrete
c = ['a','b','c']*(N/3)
p = ['panel 1','panel 2']*(N/2)

""" distinct plot type comparison, do not use panel """
_, h_ax = plt.subplots(2,2, figsize=[10,8])
plt.suptitle('show case of distinct plot type/style for GroupPlot')
h_ax = np.ravel(h_ax)
# conitnuous x
plt.axes(h_ax[0])
pnp.GroupPlot(values=values, x=x_continuous, c=c)
# descrete x, default box,
plt.axes(h_ax[1])
pnp.GroupPlot(values=values, x=x_discrete, c=c)
# descrete x, bar, more labels
plt.axes(h_ax[2])
pnp.GroupPlot(values=values, x=x_discrete, c=c, plot_type='bar', errbar='se',
              values_name='label_of_value', x_name='label_x', c_name='label_c', title_text='add text')
# descrete x, violin, do not use condition c, no legend, no count
plt.axes(h_ax[3])
pnp.GroupPlot(values=values, x=x_discrete, plot_type='violin', tf_legend=False, tf_count=False)

""" use seprate data by panel """
pnp.GroupPlot(values=values, x=x_discrete, c=c, p=p, plot_type='box', values_name='label_of_value', x_name='label_x', c_name='label_c')


""" ===== test code for DfPlot ===== """

data_df = pd.DataFrame( {'values': values, 'x':x, 'c':c, 'p':p, 'x_continuous':x_continuous, 'x_discrete':x_discrete} )   # create pandas dataframe

""" distinct plot type comparison, do not use panel """
_, h_ax = plt.subplots(2,2, figsize=[10,8])
plt.suptitle('show case of distinct plot type/style for GroupPlot')
h_ax = np.ravel(h_ax)
# conitnuous x
plt.axes(h_ax[0])
pnp.DfPlot(data_df, values='values', x='x_continuous', c='c')
# descrete x, default box,
plt.axes(h_ax[1])
pnp.DfPlot(data_df, values='values', x='x_discrete', c='c')
# descrete x, bar, more labels
plt.axes(h_ax[2])
pnp.DfPlot(data_df, values='values', x='x_discrete', c='c', plot_type='bar', errbar='se', title_text='add text')
# descrete x, violin, do not use condition c, no legend, no count
plt.axes(h_ax[3])
pnp.DfPlot(data_df, values='values', x='x_discrete', plot_type='violin', tf_legend=False, tf_count=False)

""" use seprate data by panel """
pnp.DfPlot(df=data_df, values='values', x='x', c='c', p='p')