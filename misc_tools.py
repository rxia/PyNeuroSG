"""
misc toool functions
"""

import datetime
import numpy as np

def get_time_string(microsecond=True):
    """
    get the current time string, returns '%Y_%m%d_%H%M%S_%f' or '%Y_%m%d_%H%M%S'

    :param microsecond: True/False to keep milisecond
    :return:            string, '2012_0901_195930_999' or '2012_0901_195930'
    """

    if microsecond:
        str_datatime = datetime.datetime.now().strftime('%Y_%m%d_%H%M%S_%f')
    else:
        str_datatime = datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')
    return str_datatime


def str_common(list_of_strings):
    """
    Get common char of a list of string, returns a string, with mis-match positions replaced using "_"
    e.g. str_common(['abc','adc']) returns 'a_c'

    :param list_of_strings: a list of strings
    :return:  a string
    """
    len_min = 0
    string_out = ''
    for char in zip(*list_of_strings):
        if len(set(char)) is 1:
            char_cur=char[0]
        else:
            char_cur = '_'
        string_out = string_out+char_cur
    return string_out


def red_text(str_in):
    """
    tool function to set text font color to red, using ASCII

    :param str_in:  str
    :return:        str that will print in red
    """
    return('\033[91m{}\033[0m'.format(str_in))


def index_bool2int(index_bool):
    """
    tool function to transform bool index to int. e.g. turn [True, False, True] into [0, 2]

    :param index_bool: bool index, like [True, False, True]
    :return:           int index, like [0, 2]
    """
    return np.where(index_bool)


def index_int2bool(index_int, N=None):
    """
    tool function to transform int index to bool. e.g. turn into [0, 2] into [True, False, True]

    :param index_int: int index, like [0, 2]
    :param N:         length of boolean array
    :return:          bool index, like [True, False, True]
    """
    if N is None:
        N=np.max(index_int)+1
    index_bool = np.zeros(N)
    index_bool[index_int]=1
    return index_bool>0.5