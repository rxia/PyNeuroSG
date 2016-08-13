# -*- coding: utf-8 -*-
"""
This is an example for reading files with neo.io
"""


import neo


# data files
# data_path = '/Users/Summit/Dropbox/Coding_Projects_Data/python-neo/tdt/aep_05'
data_path = '/Users/Summit/Dropbox/Coding_Projects_Data/python-neo/Dexter_2016-0408-160419-175957'
output_path = '/Users/Summit/Dropbox/Coding_Projects_Data/python-neo/Dexter_2016-0408-160419-175957.h5'

#create a reader
reader = neo.io.TdtIO(dirname=data_path)

# read the blocks
blk = reader.read_block(cascade=True, lazy=False)


# create writer
writer = neo.io.hdf5io.NeoHdf5IO(output_path)
writer.write(blk)
writer.close()



