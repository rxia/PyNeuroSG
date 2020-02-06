""" sub-moduel of PyNeuroData for storing data_neuro to hdf5 and loading it back """

import h5py
import pandas as pd


def SaveToH5(data_neuro, h5_filepath='', h5_groups=[]):
    """
    funcition to save data_neuro as hdf5 format

    :param data_neuro:  data_neuro object
    :param h5_filepath: str, path to the hdf5 file in the file system, e.g. './temp_data/example_data.hdf5'
    :param h5_groups:   list of string, the group names specifying the location in hdf5, e.g. ['2017_0501', 'spks']
    :return:            None
    """

    # numpy components using h5py api
    with h5py.File(h5_filepath, 'a') as hf:
        hf_cur = hf
        for level, h5_group in enumerate(h5_groups):
            if h5_group not in hf_cur.keys():
                hf_cur.create_group(h5_group)
            hf_cur = hf_cur[h5_group]

        if len(hf_cur.keys()) >0:
            raise Exception('hf_group_already_exsits in hdf5 file, {}'.format(h5_groups))

        hf_cur.create_dataset('data', data=data_neuro['data'])
        hf_cur.create_dataset('ts', data=data_neuro['ts'])

    # pandas components using pandas api
    group_key = '/'.join(h5_groups)
    with pd.HDFStore(h5_filepath) as hf_pandas:
        hf_pandas.put(key='{}/trial_info'.format(group_key), value=data_neuro['trial_info'], data_columns=True)
        hf_pandas.put(key='{}/signal_info'.format(group_key), value=pd.DataFrame(data_neuro['signal_info']), data_columns=True)

    print('finsihed saving data_neuro to gropu {}, in file {}'.format(group_key, h5_filepath))

    return None


def LoadFromH5(h5_filepath='', h5_groups=[]):

    # numpy components using h5py api
    with h5py.File(h5_filepath, 'r') as hf:
        hf_cur = hf
        for level, h5_group in enumerate(h5_groups):
            if h5_group not in hf_cur.keys():
                hf_cur.create_group(h5_group)
            hf_cur = hf_cur[h5_group]

        data_neuro = {}
        data_neuro['data'] = hf_cur['data'][:]
        data_neuro['ts'] = hf_cur['ts'][:]

    # pandas components using pandas api
    group_key = '/'.join(h5_groups)
    with pd.HDFStore(h5_filepath) as hf_pandas:
        data_neuro['trial_info'] = hf_pandas.get(key='{}/trial_info'.format(group_key))
        data_neuro['signal_info'] = hf_pandas.get(key='{}/signal_info'.format(group_key))

    return data_neuro


def ShowH5(h5_filepath='', h5_groups=[], yn_print=True):
    """
    display the structrue of hdf5 file

    :param h5_filepath: str, path to the hdf5 file in the file system, e.g. './temp_data/example_data.hdf5'
    :param h5_groups:   list of string, the group names specifying the location in hdf5, e.g. ['2017_0501', 'spks']
    :return:
    """

    def show_obj(obj, level, dict_obj):
        if isinstance(obj, h5py.Dataset):
            name = obj.name.split('/')[-1]
            dict_obj[name] = None
            if yn_print:
                print('| '*level + name + '    {}'.format(obj.shape))
        elif isinstance(obj, h5py.Group):
            name = obj.name.split('/')[-1]
            dict_obj[name] = dict()
            if yn_print:
                print('| '*level + name + '    (Group)')
            level += 1
            if 'block0_items' in list(obj.keys()):   # is pandas object
                return None
            for key in list(obj.keys()):      # recursion
                show_obj(obj[key], level, dict_obj[name])

    dict_tree = dict()
    with h5py.File(h5_filepath, 'r') as hf:
        hf_cur = hf
        for level, h5_group in enumerate(h5_groups):
            hf_cur = hf_cur[h5_group]

        show_obj(hf_cur, 0, dict_tree)

    return dict_tree['']