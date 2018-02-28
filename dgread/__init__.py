import sys
import warnings

if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
    if sys.version_info[0] == 3:
        from .linux_py35.dgread import dgread
    elif sys.version_info[0] == 2:
        from .linux_py27.dgread import dgread
else:
    warnings.warn('function dgread has not been implemented in your OS: {}'.format(sys.platform))
    try:
        from .linux_py35.dgread import dgread
    except:
        pass
    try:
        from .linux_py35.dgread import dgread
    except:
        pass
