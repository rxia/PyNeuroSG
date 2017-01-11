"""
funciton for loading data, designed specifically for the format/naming scheme in the Sheinberg lab at Brown University
"""

import os     # for getting file paths
import neo    # for reading neural data (TDT format)
import dg2df  # for reading behavioral data
import pandas as pd
import re     # use regular expression to find file names
import numpy as np
from standardize_TDT_blk import select_obj_by_attr
import warnings


def get_file_name(keyword = None,           # eg. 'd_.*_122816'
                  keyword_tank = None,      # eg. '.*GM32_U16.*161228.*'
                  tf_interactive = False,
                  dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/',
                  dir_dg  = '/Volumes/Labfiles/projects/analysis/shaobo/data_dg'):
    """
    funciton to get the name of data files
    :param keyword:         key word for tdt blocks
    :param keyword_tank:    key word for tdt tanks
    :param tf_interactive:  flag for interactively selecting file
    :param dir_tdt_tank:    root directory for tdt tanks    (neural data)
    :param dir_dg:          root directory for stimdg files (behavioral data)
    :return:                (name_tdt_blocks, path_tdt_tank)
    """

    name_tdt_blocks = []
    name_tdt_tank = ''
    path_tdt_tank = ''

    """ get the list of tanks that matches the keyword """
    list_name_tanks = os.listdir(dir_tdt_tank)
    if keyword_tank is not None:                            # if keyword_tank is given, filter the range
        list_name_tanks = [name_tank for name_tank in list_name_tanks if re.match(keyword_tank, name_tank) is not None]

    """ search for blocks that matches the keyword """
    for name_tank in list_name_tanks:                       # for every directory
        path_tank = os.path.join(dir_tdt_tank, name_tank)   # get path
        if os.path.isdir(path_tank):                        # if is directory, treat as a tank
            for name_block in os.listdir(path_tank):        # for every block
                if re.match(keyword, name_block) and os.path.isdir(os.path.join(path_tank, name_block)) is not None:  # if matches the keyword
                    name_tdt_blocks.append(name_block)
        if len(name_tdt_blocks) >0:                         # if matching blocks exist, break
            name_tdt_tank = name_tank
            path_tdt_tank = path_tank
            break

    """ sort the names """
    name_tdt_blocks.sort()

    """ if interactive mode, use keyboard to confirm the selection """
    if tf_interactive:           # if interactive, type in y/n to select every file
        name_tdt_blocks_select = []
        print('')
        print('the tank selected is {}'.format(name_tdt_tank))
        print('the blocks selected are {}'.format(name_tdt_blocks))
        action_keyboard = raw_input('please type to confirm: accept all (a) / decline all (d) / select one-by-one (s)')
        if action_keyboard == 'a':
            pass
        elif action_keyboard == 'd':
            name_tdt_blocks = []
            path_tdt_tank   = ''
        elif action_keyboard == 's':
            i=0
            while i<len(name_tdt_blocks):
                name_block = name_tdt_blocks[i]
                yn_keyboard = raw_input('keep file {}: (y/n)'.format(name_block))
                if yn_keyboard == 'y':
                    name_tdt_blocks_select.append(name_block)
                    i = i + 1
                elif yn_keyboard == 'n':
                    i = i + 1
                else:
                    print('please type "y" or "n"')
            name_tdt_blocks = name_tdt_blocks_select

    return (name_tdt_blocks, path_tdt_tank)



def load_data(keyword = None,
              keyword_tank = None,
              dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/',
              dir_dg  = '/Volumes/Labfiles/projects/analysis/shaobo/data_dg',
              sortname = 'PLX',
              tf_interactive = True ,
              tf_verbose = True):
    """
    :param keyword:         key word for tdt blocks
    :param keyword_tank:    key word for tdt tanks
    :param dir_tdt_tank:    root directory for tdt tanks    (neural data)
    :param dir_dg:          root directory for stimdg files (behavioral data)
    :param sortname:        name of sort code in TDT format
    :param tf_interactive:  flag for interactively selecting file
    :param tf_verbose:      flag for print intermediate results
    :return:                (blk, data_df, name_datafiles)
    """


    """ ----- get the name and path of data files ----- """
    [name_tdt_blocks, path_tdt_tank] = \
        get_file_name(keyword, keyword_tank, tf_interactive=tf_interactive, dir_tdt_tank = dir_tdt_tank, dir_dg=dir_dg)

    file_dgs = [name + '.dg' for name in name_tdt_blocks]   # add '.dg' to every name
    name_datafiles = name_tdt_blocks
    if tf_verbose:
        print('')
        print('the data files to be loaded are: {}'.format(name_datafiles))

    """ ----- load neural data ----- """
    blk = neo.core.Block()                       # block object containing multiple segments, each represents data form one file
    reader = neo.io.TdtIO(dirname=path_tdt_tank)  # reader for loading data
    for name_tdt_block in name_tdt_blocks:       # for every data file
        if tf_verbose:
            print('loading TDT block: {}'.format(name_tdt_block))
        seg = reader.read_segment(blockname=name_tdt_block, sortname=sortname)   # read one TDT file as a segment of block
        blk.segments.append(seg)                 # append to blk object
    if tf_verbose:
        print('finish loading tdt blocks')


    """ ----- load behaviral data ----- """
    data_dfs = []                                # list containing multiple pandas dataframes, each represents data form one file
    for i in range(len(file_dgs)):                     # for every data file, read as a segment of block
        file_dg = file_dgs[i]
        if tf_verbose:
            print('loading dg: {}'.format(file_dg))
        path_dg = os.path.join(dir_dg, file_dg)
        data_df = dg2df.dg2df(path_dg)                # read using package dg2df, returns a pandas dataframe
        data_df['filename'] = [file_dg]*len(data_df)  # add a column for filename
        data_df['fileindex'] = [i] * len(data_df)     # add a column for file id (index from zero to n-1)
        data_dfs.append(data_df)

    if len(data_dfs)>0:
        data_df = pd.concat(data_dfs)                # concatenate in to one single data frame
        data_df = data_df.reset_index(range(len(data_df)))
    else:
        data_df = pd.DataFrame([])

    if tf_verbose:
        print('finish loading and concatenating dgs')

    return (blk, data_df, name_datafiles)



def get_ts_align(blk, data_df,
                  dg_name_obsid='obsid', dg_tos_align='stimon', dg_tof_obs = 'endobs',
                  neo_name_obson='obsv', neo_name_obsoff='obs\\',
                  tf_align_test=True, thrhld_misalign=0.002):
    """
    Get onset timestamps of the alignment events (eg. StimOn), in the neo time frame
    :param blk:              neo block
    :param data_df:          pandas dataframe from stim dg
    :param dg_name_obsid:    in dg, the name of obs id (obs index for a row of data)
    :param dg_tos_align:     in dg, the time of alignment event onset
    :param dg_tof_obs:       in dg, the time of alignment evetn offset
    :param neo_name_obson:   in neo block, the name of obs  onset event
    :param neo_name_obsoff:  in neo block, the name of obs offset event
    :param tf_align_test:    true/false flag, if doing alignment test (based on obs duration)
    :param thrhld_misalign:  threshold for mis-alignment warning, in sec
    :return:      a list of arrays, each array contains the time stamps of the alignment event onset of a data file
    """
    # get ts_align
    blk_StimOn = []
    for i in sorted(data_df['fileindex'].unique()):

        """ neo data blk """
        # get the timestamps of obs onset, in sec
        ts_ObsvOn = np.array(select_obj_by_attr(blk.segments[i].events, attr='name', value=neo_name_obson)[0].times)

        """ stim dg """
        data_df_segment = data_df[data_df['fileindex']==i]      # get the segment corresponding to a dg file
        id_Obsv = np.array(data_df_segment[dg_name_obsid])      # get the obsid (id of observation period / trials)

        # get the time of alignment event onset relative to obs onset in stimdg in ms
        tos_StimOn = np.array(data_df_segment[dg_tos_align])

        # calculate the timestamps of the alignment event onset, in sec
        ts_StimOn = ts_ObsvOn[np.array(id_Obsv)] + tos_StimOn/1000.0

        blk_StimOn.append(ts_StimOn)

        if tf_align_test:
            ts_ObsvOff = np.array(select_obj_by_attr(blk.segments[i].events, attr='name', value=neo_name_obsoff)[0].times)
            dur_obs_neo = ts_ObsvOff - ts_ObsvOn                         # obs duration from neo
            dur_obs_dg  = np.array(data_df_segment[dg_tof_obs])/1000.0   # obs duration from stim dg
            dur_misalign = np.max(np.abs(dur_obs_neo[id_Obsv] - dur_obs_dg))   # max diff for corresponding obs
            if dur_misalign > thrhld_misalign:
                cur_file_name =  data_df[data_df['fileindex'] == i]['filename'][i]
                print(red_text( 'maximum misalignment time is {} ms, in file {}'.format(dur_misalign*1000, cur_file_name) ))
    return blk_StimOn


def red_text(str_in):
    """
    tool function to set text font color to red, using ASCII
    :param str_in:  str
    :return:        str that will print in red
    """
    return('\033[91m{}\033[0m'.format(str_in))