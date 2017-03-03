""" this module is to get information fro the google doc and plot the signal quality over time """

""" import modules """
import csv
import re
import numpy as np
import datetime
import warnings
import pickle
from matplotlib import pyplot as plt
import PyNeuroPlot as pnp

path_to_original_csv = '/shared/homes/sguan/neuro_data/' + 'Dante GrayMatter 32 V4 Log - Advance log.csv'
path_to_result_pickle = './temp_data/GM32_log_info.pkl'

item_total = 2
item_count = item_total
list_date = []
list_total_turn = []
list_spikes = []

# use csv reader to gather information
with open(path_to_original_csv, 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if re.match('\d{1,2}/\d{1,2}/\d{2,4}', row[0]):   # get date information
            if item_count != item_total or item_count != item_total:
                warnings.warn('fails to get all {} items before {}'.format(item_total, row[0]) )
            list_date.append(datetime.datetime.strptime(row[0], '%m/%d/%Y'))
            item_count = 0
            tf_spikes = False
        if row[0] == 'Total Turn':
            list_total_turn.append(row[1:])
            item_count += 1
        if (row[0] == 'spikes' or row[0] == 'neuron')  and tf_spikes==False:
            list_spikes.append(row[1:])
            item_count += 1
            tf_spikes = True


date = np.array(list_date)
total_turn = np.array(list_total_turn, dtype=float)
spikes = np.array(list_spikes)
spikes[np.logical_or(spikes=='', spikes==' ')]='0'
spikes = spikes.astype(float)
total_depth = total_turn*0.125

result_log = {'date': date, 'total_turn': total_turn, 'total_depth': total_depth}

plt.figure()
plt.plot(date, total_turn, '-o')

from GM32_layout import layout_GM32
[h_fig, h_axes] = pnp.create_array_layout_subplots(layout_GM32)
plt.tight_layout()
h_fig.subplots_adjust(hspace=0.02, wspace=0.02)
h_fig.set_size_inches([10, 10], forward=True)
spike_color_dict={-2:'grey', -1:'grey', 0:'y',1:'g',2:'c',3:'b'}
import matplotlib.dates as mdates
formatter = mdates.DateFormatter('%b')
text_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
for i in range(32):
    plt.axes(h_axes[layout_GM32[i+1]])
    plt.plot(date, -total_depth[:,i], '-k')
    plt.scatter(date, -total_depth[:,i], c=[spike_color_dict[x] for x in spikes[:,i]])
    plt.gca().xaxis.set_major_formatter(formatter)
    # plt.gca().set_xlabel('date')
    # plt.gca().set_ylabel('depth')
    plt.text(0.05, 0.95, 'Chan {}'.format(i+1), transform=plt.gca().transAxes,fontsize=12,verticalalignment='top',bbox=text_props)
plt.suptitle('depth over recording days')
plt.savefig('./temp_figs/GM32_advance_log.png')

