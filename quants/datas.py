from __future__ import print_function, absolute_import

from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.types import *
import warnings
import datetime as dt
import pandas as pd
import numpy as np
import gm.api as gm 
import tushare as ts
import jqdatasdk as jq
import baostock as bs


class Code(object):
    def __init__(self,code):
        if code.endswith('.SZ') or code.endswith('.SH') or code.endswith('.HK'):#tushare格式
            self.code = code.replace('.SZ','.XSHE').replace('.SH','.XSHG')
        elif len(code)==8 and  (code.startswith('sh') or code.startswith('sz')):
            self.code=code[2:8]+code[0:2].replace('sz','.XSHE').replace('sh','.XSHG')
        elif len(code)==9 and  (code.startswith('sh') or code.startswith('sz')):
            self.code = code[3:9] + code[0:3].replace('sz.', '.XSHE').replace('sh.', '.XSHG')
        elif len(code)==11 and (code.startswith('SHSE') or code.startswith('SZSE')):
            self.code==code[5:]+code[0:4].replace('SZSE', '.XSHE').replace('SHSE', '.XSHG')
        else:
            self.code=code

    def tushare(self):
        return self.code.replace('.XSHE','.SZ').replace('.XSHG','.SH')
    def baostock(self):
        return self.code[6,11].replace('.XSHE','sz.').replace('.XSHG','sh.')+self.code[0:6]
    def jq(self):
        return self.code
    def akshare(self):
        return self.code[6,11].replace('.XSHE','sz').replace('.XSHG','sh')+self.code[0:6]
    def gm(self):
        return self.code[6:11].replace('.XSHE','SZSE.').replace('.XSHG','SHSE.')+self.code[0:6]

class DT(object):
    def __init__(self,dates):
        self.dt=dates
    def tushare(self):
        return self.dt.strftime("%Y%m%d")
    def baostock(self):
        return self.dt.strftime("%Y-%m-%d")
    def jqdata(self):
        return self.dt
    def gmsdk(self):
        return self.dt
    
class Unit(object):
    def __init__(self,unit):
        """1m 1d 1w 1M"""
        self.unit=unit
    def baostock(self):
        if self.unit[-1]=='m':
            return self.unit[:-1]
        elif self.unit[-1]=='d':
            return "d"
        elif self.unit[-1]=='w':
            return "w"
        elif self.unit[-1]=='M':
            return "m"
    def tushare(self):
        if self.unit[-1]=='m':
            return self.unit[:-1]+"min"
        elif self.unit[-1]=='d':
            return "D"
        elif self.unit[-1]=='w':
            return "W"
        elif self.unit[-1]=='M':
            return "M"
        
class Adjust(object):
    def __init__(self,adj):
        self.adj=adj
    def baostock(self):
        if self.adj=='qfq':
            return 2
        elif self.adj=='hfq':
            return 1
        elif self.adj=='':
            return 3
    def tushare(self):
        if self.adj=='qfq':
            return 'qfq'
        elif self.adj=='hfq':
            return 'hfq'
        elif self.adj=='':
            return None
    def jqdata(self):
        if self.adj=='qfq':
            return 'pre'
        elif self.adj=='hfq':
            return 'post'
        elif self.adj=='':
            return None

class Datas:
    def __init__(self,config):
        warnings.filterwarnings('ignore')
        self.engine = create_engine(
            config['mysql']['db_url'],
            echo=False)
        gm.set_token(config['gm']['token'])
        self.gm=gm
        ts.set_token(config['tushare']['token'])
        self.ts=ts
        self.ts_pro=ts.pro_api()
        jq.auth(config['jqdatasdk']['username'],config['jqdatasdk']['password'])
        self.jq=jq
        bs.login()
        self.bs=bs
        
    def jqdata(self):
        return self.jq
    def tushare(self):
        return self.ts
    def tushare_pro(self):
        return self.ts_pro
    def gmsdk(self):
        return self.gm
    def baostock(self):
        return self.bs       
    def get_con(self):
        return self.engine

    def stock_history_m(self,code,start_dt=(dt.datetime.now()-dt.timedelta(days=30)),end_dt=dt.datetime.now(),unit='30m'):
        rs = bs.query_history_k_data_plus(Code(code).baostock(),
                "date,time,open,high,low,close,volume,amount",
                start_date=start_dt, end_date=end_dt,
                frequency=Unit(unit).baostock(), adjustflag="2")
        df=rs.get_data()
        df.columns=['date','time','open','high','low','close','volume','money']
        df['datetime']=pd.to_datetime(df.time,format='%Y%m%d%H%M%S%f')
        df1=df.set_index('datetime')
        return df1[['open','high','low','close','volume','money']].copy()

    def stock_history_d(self,code,asset='E',start_dt=(dt.datetime.now()-dt.timedelta(days=750)),end_dt=dt.datetime.now(),unit='1d',adj='qfq'):
        df=self.ts.pro_bar(ts_code=Code(code).tushare(), asset=asset,adj=Adjust(adj).tushare(),freq=Unit(unit).tushare(), start_date=DT(start_dt).tushare(), end_date=DT(end_dt).tushare())
        df['datetime']=pd.to_datetime(df['trade_date'],format='%Y%m%d')
        df.columns=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close',
            'change', 'pct_chg', 'volume', 'money', 'datetime']
        df['volume']=df['volume']*100
        df['money']=df['money']*1000
        df1=df.set_index('datetime')
        return df1[['open','high','low','close','volume','money']].copy()
    
    def stock_instruments(self):
        return self.sql_df(SQLS.stock_concepts())
    
    def etf_instruments(self):
        return self.sql_df(SQLS.etf_securities())    
    
    def stock_concepts(self):
        return self.sql_df(SQLS.stock_concepts())
    
  

        
    


    def dtypes_normal(self, df):
        dtypedict = {}
        for i, j in zip(df.columns, df.dtypes):
            if "object" in str(j):
                dtypedict.update({i: NVARCHAR(length=20)})
            if "float" in str(j):
                dtypedict.update({i: Float(precision=4, asdecimal=True)})
            if "int" in str(j):
                dtypedict.update({i: Integer()})
        return dtypedict

    def dtypes_long_text(self, df):
        dtypedict = {}
        for i, j in zip(df.columns, df.dtypes):
            if "object" in str(j):
                dtypedict.update({i: Text()})
            if "float" in str(j):
                dtypedict.update({i: Float(precision=4, asdecimal=True)})
            if "int" in str(j):
                dtypedict.update({i: Integer()})
        return dtypedict

    def sql_update(self, sql_str, params):
        """
        例子 db.sql_update('delete from stock_daily where date=:date', {'date':str(dat)})
        :param sql_str:
        :param params:
        :return:
        """
        with self.get_con().connect() as con:
            try:
                con.execute(text(sql_str), params)
            except:
                print('sql执行出错:' + sql_str + " " + str(params))
    def sql_df(self,sql_str):
        return pd.read_sql_query(sql_str,self.get_con())

class SQLS(object):

    @staticmethod
    def stock_securities():
        return "select * from securities where type='stock'"

    @staticmethod
    def etf_securities():
        return "select * from securities where type='etf'"
    
    @staticmethod
    def stock_concepts():
        return "select * from stock_concept"
    @staticmethod
    def stock_hot_rank_code(code=None):
        return "select * from stock_hot_rank where code='{code}'".format(code=code)
            
    @staticmethod
    def stock_hot_rank_date(start_date=None,end_date=None):
        if start_date==None and end_date==None:
            return "select * from stock_hot_rank where date>='{date}'".format(date=(dt.datetime.now()-dt.timedelta(days=30)).strftime("%Y-%m-%d"))
        elif start_date!=None and end_date==None: 
            return "select * from stock_hot_rank where date>='{date}'".format(date=start_date.strftime("%Y-%m-%d"))
        elif start_date!=None and end_date!=None: 
            return "select * from stock_hot_rank where date>='{date}' and date<='{end}'".format(date=start_date.strftime("%Y-%m-%d"),end=end_date.strftime("%Y-%m-%d"))
        
    @staticmethod
    def stock_holder_number_code(code):
        return "select * from stk_holdernumber where code ='{code}'".format(code=code)
    @staticmethod
    def stock_holder_number_all(start_date=None,end_date=None):
        if start_date==None and end_date==None:
            return "select * from stk_holdernumber where ann_date>='{date}'".format(date=(dt.datetime.now()-dt.timedelta(days=30)).strftime("%Y-%m-%d"))
        elif start_date!=None and end_date==None: 
            return "select * from stock_hot_rank where ann_date>='{date}'".format(date=start_date.strftime("%Y-%m-%d"))
        elif start_date!=None and end_date!=None: 
            return "select * from stock_hot_rank where ann_date>='{date}' and ann_date<='{end}'".format(date=start_date.strftime("%Y-%m-%d"),end=end_date.strftime("%Y-%m-%d"))
    
    
        

__all__ = ["Code","Datas","SQLS","Unit","Adjust","DT"]





