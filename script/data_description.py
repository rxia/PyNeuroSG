""" stores the experiment/recording setup """
import numpy as np

""" recording area in IT """
date_area = dict()
date_area['161015'] = 'TEd'
date_area['161023'] = 'TEm'
date_area['161026'] = 'TEm'
date_area['161029'] = 'TEd'
date_area['161118'] = 'TEm'
date_area['161121'] = 'TEm'
date_area['161125'] = 'TEm'
date_area['161202'] = 'TEm'
date_area['161206'] = 'TEd'
date_area['161222'] = 'TEm'
date_area['161228'] = 'TEd'
date_area['170103'] = 'TEd'
date_area['170106'] = 'TEm'
date_area['170113'] = 'TEd'
date_area['170117'] = 'TEd'
date_area['170214'] = 'TEd'
date_area['170221'] = 'TEd'


""" the channel index (count from zero) of the granular layer """
indx_g_layer = dict()
indx_g_layer['161015'] = 8
indx_g_layer['161023'] = np.nan
indx_g_layer['161026'] = 8
indx_g_layer['161029'] = 9
indx_g_layer['161118'] = 6
indx_g_layer['161121'] = 4
indx_g_layer['161125'] = 5
indx_g_layer['161202'] = 5
indx_g_layer['161206'] = 8
indx_g_layer['161222'] = 6
indx_g_layer['161228'] = 7
indx_g_layer['170103'] = 7
indx_g_layer['170106'] = 2
indx_g_layer['170113'] = 9
indx_g_layer['170117'] = 6
indx_g_layer['170214'] = np.nan
indx_g_layer['170221'] = np.nan

depth_from_g = dict()
for (date, area) in date_area.items():
    if area == 'TEm':
        depth = np.arange(16,0,-1)-1 - indx_g_layer[date]
    elif area == 'TEd':
        depth =   np.arange(16) - indx_g_layer[date]
    depth_from_g[date] = depth



""" GM32 electrode layout """
# manually define the layout
ch = range(1,33)     # channel
r  = [5]*5+[4]*5+[3]*6+[2]*6+[1]*5+[0]*5
c  = range(4,-1,-1)*2 + range(5,-1,-1)*2 + range(4,-1,-1)*2
layout_GM32 = dict(zip(ch, zip(r,c)))


