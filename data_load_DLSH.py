"""
funciton for loading data, designed specifically for the format/naming scheme in the Sheinberg lab at Brown University
"""

import os     # for getting file paths
import neo    # for reading neural data (TDT format)
import dg2df  # for reading behavioral data
import pandas as pd
import re     # use regular expression to find file names
import numpy as np
from standardize_TDT_blk import select_obj_by_attr, convert_name_to_unicode
import warnings


def get_file_name(keyword = None,           # eg. 'd_.*_122816'
                  keyword_tank = None,      # eg. '.*GM32_U16.*161228.*'
                  tf_interactive = False, tf_verbose=False,
                  dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/',
                  dir_dg  = '/Volumes/Labfiles/projects/analysis/shaobo/data_dg',
                  mode='both'):
    """
    funciton to get the name of data files
    :param keyword:         key word for tdt blocks
    :param keyword_tank:    key word for tdt tanks
    :param tf_interactive:  flag for interactively selecting file
    :param tf_verbose:      flag for printing stuff
    :param dir_tdt_tank:    root directory for tdt tanks    (neural data)
    :param dir_dg:          root directory for stimdg files (behavioral data)
    :param mode:            'tdt', 'dg', or 'both'; default to 'both'
    :return:                (name_tdt_blocks, path_tdt_tank)
    """

    name_tdt_blocks = []
    path_tdt_tank = ''

    """ ===== read tdt block name ===== """
    if (mode == 'both') or (mode=='tdt'):

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

        # sort the names
        name_tdt_blocks.sort()

        """ if interactive mode, use keyboard to confirm the selection """
        if tf_interactive:           # if interactive, type in y/n to select every file
            name_tdt_blocks_select = []
            print('')
            print('the tank selected is {}'.format(name_tdt_tank))
            print('the blocks selected are {}'.format(name_tdt_blocks))
            action_keyboard = input('please type to confirm: accept all (a) / decline all (d) / select one-by-one (s)')
            if action_keyboard == 'a':
                pass
            elif action_keyboard == 'd':
                name_tdt_blocks = []
                path_tdt_tank   = ''
            elif action_keyboard == 's':
                i=0
                while i<len(name_tdt_blocks):
                    name_block = name_tdt_blocks[i]
                    yn_keyboard = input('keep file {}: (y/n)'.format(name_block))
                    if yn_keyboard == 'y':
                        name_tdt_blocks_select.append(name_block)
                        i = i + 1
                    elif yn_keyboard == 'n':
                        i = i + 1
                    else:
                        print('please type "y" or "n"')
                name_tdt_blocks = name_tdt_blocks_select


    """ ===== read dg name ===== """
    if (mode=='both') or (mode=='dg'):

        name_dgs = []
        for name_dg in os.listdir(dir_dg):
            if re.match(keyword, name_dg):   # if matches the keyword
                name_dgs.append(name_dg)
        # remove .dg extention name
        name_dgs = [re.search('(.*)\.dg', name_dg).group(1) for name_dg in name_dgs if re.search('.*\.dg', name_dg) is not None]
        name_dgs.sort()

        """ interactive mode if tdt is not used """
        if tf_interactive == True and path_tdt_tank == '':
            name_dgs_select = []
            print('')
            print('the dgs selected are {}'.format(name_dgs))
            action_keyboard = input('please type to confirm: accept all (a) / decline all (d) / select one-by-one (s)')
            if action_keyboard == 'a':
                pass
            elif action_keyboard == 'd':
                name_dgs = []
            elif action_keyboard == 's':
                for i in range(len(name_dgs)):
                    name_dg = name_dgs[i]
                    yn_keyboard = input('keep file {}: (y/n)'.format(name_dg))
                    if yn_keyboard == 'y':
                        name_dgs_select.append(name_dg)
                    elif yn_keyboard == 'n':
                        pass
                    else:
                        print('please type "y" or "n"')
                name_dgs = name_dgs_select

        if mode=='both':    # get intersection of name_tdt_blocks and name_dgs
            print('the following tdt blockes are selected: {}'.format(name_tdt_blocks))
            print('the following dg files are selected: {}'.format(name_dgs))
            name_tdt_blocks = list(np.intersect1d(name_tdt_blocks, name_dgs))
            print('the their intersections are: {}'.format(name_tdt_blocks))
        elif mode=='dg':
            name_tdt_blocks = name_dgs
            print('the following dg files are selected: {}'.format(name_dgs))
        elif mode=='tdt':
            print('the following tdt blosks are selected: {}'.format(name_tdt_blocks))

    return (name_tdt_blocks, path_tdt_tank)



def load_data(keyword = None,
              keyword_tank = None,
              dir_tdt_tank  = '/Volumes/Labfiles/projects/encounter/data/TDT/',
              dir_dg  = '/Volumes/Labfiles/projects/analysis/shaobo/data_dg',
              sortname = 'PLX',
              tf_interactive = True ,
              tf_verbose = True,
              mode = 'both'):
    """
    :param keyword:         key word for tdt blocks
    :param keyword_tank:    key word for tdt tanks
    :param dir_tdt_tank:    root directory for tdt tanks    (neural data)
    :param dir_dg:          root directory for stimdg files (behavioral data)
    :param sortname:        name of sort code in TDT format
    :param tf_interactive:  flag for interactively selecting file
    :param tf_verbose:      flag for print intermediate results
    :param mode:            'tdt', 'dg', or 'both'; default to 'both'
    :return:                (blk, data_df, name_datafiles)
    """


    """ ----- get the name and path of data files ----- """
    [name_tdt_blocks, path_tdt_tank] = \
        get_file_name(keyword, keyword_tank, dir_tdt_tank = dir_tdt_tank, dir_dg=dir_dg,
                      tf_verbose=tf_verbose, tf_interactive=tf_interactive, mode=mode)

    file_dgs = [name + '.dg' for name in name_tdt_blocks]   # add '.dg' to every name
    name_datafiles = name_tdt_blocks
    if tf_verbose:
        print('')
        print('the data files to be loaded are: {}'.format(name_datafiles))

    blk = None
    data_df = None

    """ ----- load neural data ----- """
    if mode=='both' or mode=='tdt':
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
    if mode=='both' or mode=='dg':
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
            data_df.reset_index(inplace=True, drop=True)
        else:
            data_df = pd.DataFrame([])

        data_df[''] = [''] * len(data_df)            # empty column, make some default condition easy

        if tf_verbose:
            print('finish loading and concatenating dgs')

    return (blk, data_df, name_datafiles)



def standardize_data_df(data_df, filename_common=''):
    """
    add colums that is necessary for later analysis
    :param data_df:            pandas dataframe, where every row represent one trial
    :param filename_common:    filemane, used to check what colums to add
    :return:                   data_df, with extra columns
    """

    # add empty column, make some default condition easy
    data_df[''] = [''] * len(data_df)
    # add index column
    if 'obsid' in data_df.keys():
        if 'fileindex' in data_df.keys():
            data_df['obs_total'] = np.sum(np.array(data_df.groupby('fileindex')['obsid'].agg(np.max)))
        else:
            data_df['obs_total'] = np.array(data_df['obsid']).max()

    if re.match('.*matchnot.*', filename_common) is not None:
        data_df['stim_names'] = data_df.SampleFilename
        data_df['stim_familiarized'] = data_df.SampleFamiliarized
        data_df['mask_opacity'] = data_df['MaskOpacity']
        data_df['mask_orientation'] = np.array(map(lambda a: int(re.match('^fftnoise_(\d*)_.*', a).group(1)), data_df['MaskFilename']))
    if re.match('.*matchnot.*', filename_common) is not None or re.match('.*_srv_mask.*', filename_common) is not None:
        # make mask opacity a int, better for printing
        data_df['mask_opacity_int'] = np.round(data_df['mask_opacity'] * 100).astype(int)
        # make short name for stim, e.g. face_fam_6016
        data_df['stim_sname'] = list(map((lambda temp: re.sub('_\w*_', '_fam_', temp[0]) if temp[1] == 1 else re.sub('_\w*_', '_nov_', temp[0])),
                                zip(data_df['stim_names'].tolist(), data_df['stim_familiarized'].tolist(), )))
    return data_df



def standardize_blk(blk):
    """
    standardize neo neuro data, add channel_index and sort code to the annotation field
    :param blk:    neo block object
    :return:       neo block object
    """

    try:
        convert_name_to_unicode(blk)
    except:
        warnings.warn('convert name to unicode was not successful')

    for seg in blk.segments:   # for every segment
        for spktrain in seg.spiketrains:       # for spiketrain, add channel_index and sort_code to annotations
            cur_chan = int(re.match('Chan(\d*) .*', spktrain.name).group(1))
            cur_code = int(re.match('.* Code(\d*)', spktrain.name).group(1))
            spktrain.annotations = {'channel_index': cur_chan, 'sort_code': cur_code}
        if False:
            for analogsignal in seg.analogsignals:  # for analogsignals, add channel_index
                try:
                    cur_chan = int(re.match('LFPs (\d*)', analogsignal.name).group(1))
                    analogsignal.annotations = {'channel_index': cur_chan}
                except:
                    pass
    return blk



def get_ts_align(blk, data_df,
                  dg_name_obsid='obsid', dg_tos_align='stimon', dg_tof_obs = 'endobs',
                  neo_name_obson=r'obsv', neo_name_obsoff=r'obs\\',
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
            try:
                ts_ObsvOff = np.array(select_obj_by_attr(blk.segments[i].events, attr='name', value=neo_name_obsoff)[0].times)
                dur_obs_neo = ts_ObsvOff - ts_ObsvOn                         # obs duration from neo
                dur_obs_dg  = np.array(data_df_segment[dg_tof_obs])/1000.0   # obs duration from stim dg
                dur_misalign = np.max(np.abs(dur_obs_neo[id_Obsv] - dur_obs_dg))   # max diff for corresponding obs
                if dur_misalign > thrhld_misalign:
                    num_misalign = np.sum( np.abs(dur_obs_neo[id_Obsv] - dur_obs_dg) > thrhld_misalign )
                    cur_file_name =  data_df[data_df['fileindex'] == i]['filename'].tolist()[0]
                    print(red_text( '{} misalignments, maximum misalignment time is {} ms, in file {}'.format(num_misalign, dur_misalign*1000, cur_file_name) ))
            except:
                warnings.warn('tf_align_test can not be executed')
    return blk_StimOn


def red_text(str_in):
    """
    tool function to set text font color to red, using ASCII
    :param str_in:  str
    :return:        str that will print in red
    """
    return('\033[91m{}\033[0m'.format(str_in))