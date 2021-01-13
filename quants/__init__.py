import sys
import os

#sys.modules["ROOT_DIR"] = os.path.abspath(os.path.dirname(__file__))

from .chart import *
from .chan import *
from .datas import *
from .panel import *


from configparser import ConfigParser 
import os  
home=os.path.expanduser('~')
configure = ConfigParser() 
config_list=configure.read(home+'/quants.ini')
if len(config_list)>0:
    qdata=Datas(configure)
else:
    qdata=None
    print("""
没有在用户Home路径发现quants.ini
格式：
[jqdatasdk]
username=189*******
password= ******

[tushare]
token= df***************

[gm]
token= ab***************************

[mysql]
db_url= mysql+mysqlconnector://username:password@host:port/db?charset=utf8         
          """)
    

__all__ = ['qdata']
__all__.extend(chart.__all__)
__all__.extend(chan.__all__)
__all__.extend(datas.__all__)
__all__.extend(panel.__all__)


