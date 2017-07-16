"""  script to test copulat model to capture the correlation """

import os
import sys
sys.path.append('/shared/homes/sguan/Coding_Projects/PyNeuroSG')

import numpy as np
import scipy as sp
import pandas as pd         # pandas tabular DataFrame for task/behavioral data
import matplotlib as mpl    # plot
import matplotlib.pyplot as plt
import re                   # regular expression
import time                 # time code execution
import datetime
import pickle

# ----- modules used to read neuro data -----
import dg2df                # for DLSH dynamic group (behavioral data)
import neo                  # data structure for neural data
import quantities as pq

# ----- modules of the project PyNeuroSG -----
import signal_align         # in this package: align neural data according to task
import PyNeuroAna as pna    # in this package: analysis
import PyNeuroPlot as pnp   # in this package: plot
import misc_tools           # in this package: misc

# ----- modules for the data location and organization in Sheinberg lab -----
import data_load_DLSH       # package specific for DLSH lab data
from GM32_layout import layout_GM32


dir_tdt_tank='/shared/lab/projects/encounter/data/TDT'
dir_dg='/shared/lab/projects/analysis/shaobo/data_dg'


keyword_block = 'd_.*srv.*'
keyword_tank = '.*GM32.*U16.*161125.*'
[blk, data_df, name_tdt_blocks] = data_load_DLSH.load_data(keyword=keyword_block, keyword_tank=keyword_tank,
                                                           tf_interactive=True, dir_tdt_tank=dir_tdt_tank, dir_dg=dir_dg)



