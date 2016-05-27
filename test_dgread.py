# test DLSH dgread function
# exec(open("./test_dgread.py").read())

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import pandas as pd
pd.set_option('display.max_columns', None)
import Tkinter, tkFileDialog   # for GUI select files
Tkinter.Tk().withdraw()

# import the customize module
# cur_path = os.path.dirname(__file__)  # get the path of this file
# sys.path.append(cur_path)
import dg2df
import PyNeuroPlot; reload(PyNeuroPlot)


# set path of datafile
dir_dg  = '/Volumes/Labfiles/projects/analysis/shaobo/data_dg'

if False:
    file_dg = 'd_MTS_bin_051916007.dg'
    file_dgs = ['d_MTS_bin_051916005.dg','d_MTS_bin_051916006.dg', 'd_MTS_bin_051916007.dg', 'd_MTS_bin_051916008.dg']


    for i in range(0,len(file_dgs)):
        file_dg = file_dgs[i]
        path_dg = os.path.join(dir_dg, file_dg)
        # load file
        if i==0:
            data_df = dg2df.dg2df(path_dg)
        else:
            temp = dg2df.dg2df(path_dg)
            data_df = pd.concat([data_df,temp])
        # path_dgs[i] = path_dg

# use gui to read files
tf_GUI_getfile = True
if tf_GUI_getfile:
    path_dgs = tkFileDialog.askopenfilename(multiple=True, initialdir=dir_dg, title='select dgs to open')

for i in range(len(path_dgs)):
    path_dg = path_dgs[i]
    if i==0:
        data_df = dg2df.dg2df(path_dg)
    else:
        temp = dg2df.dg2df(path_dg)
        data_df = pd.concat([data_df,temp])

# save df to disk
# data_df.to_hdf('test_hdf_store.h5', 'stimdg', mode='w')
# data_df.to_json('test_json_store.json')

# analyze
# data_df['NoiseOpacityInt']=(data_df['NoiseOpacity']*100).astype(int)
# data_df.groupby(['NoiseOpacityInt'])[['status']].agg(np.mean).plot(kind='bar', subplots=True)
# data_df.groupby(['NoiseOpacityInt'])[['resp']].agg(np.mean).plot(kind='bar', subplots=True)
# data_df.groupby(['NoiseOpacityInt','IndexNoise'])[['status']].agg(np.mean).plot(kind='bar', subplots=True)
# plt.gca().get_xaxis().set_major_formatter()

data_df['NoiseOpacityInt']=(data_df['NoiseOpacity']*100).astype(int)
data_df['NoiseFixed']=(data_df['IndexNoise']>=0)
PyNeuroPlot.PyNeuroPlot(data_df, 'status','NoiseOpacityInt')
plt.show()
print("finish")