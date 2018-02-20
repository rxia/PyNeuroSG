import sys

if sys.version_info[0] == 3:
    from .linux_py35.dgread import dgread
elif sys.version_info[0] == 2:
    from .linux_py27.dgread import dgread