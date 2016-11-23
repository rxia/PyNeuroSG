"""
misc toool functions
"""

import datetime

def get_time_string(microsecond=True):
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