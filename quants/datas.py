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
        """单只股票指定时间段分钟K线 支持 5m 15m 30m 60m """
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
        """单只股票指定日期K线"""
        df=self.ts.pro_bar(ts_code=Code(code).tushare(), asset=asset,adj=Adjust(adj).tushare(),freq=Unit(unit).tushare(), start_date=DT(start_dt).tushare(), end_date=DT(end_dt).tushare())
        df['datetime']=pd.to_datetime(df['trade_date'],format='%Y%m%d')
        df.columns=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close',
            'change', 'pct_chg', 'volume', 'money', 'datetime']
        df['volume']=df['volume']*100
        df['money']=df['money']*1000
        df1=df.set_index('datetime')
        return df1[['open','high','low','close','volume','money']].copy()

    def stock_history_date_all(self,date):
        """单日所有股票K线"""
        df = self.ts_pro.daily(trade_date=DT(date).tushare())
        df['datetime']=pd.to_datetime(df['trade_date'],format='%Y%m%d')
        df['code']=self.jq.normalize_code(df['ts_code'].tolist())
        df['volume']=df['vol']*100
        df['money']=df['amount']*1000
        return df[['code', 'datetime', 'open', 'high', 'low', 'close', 'pre_close','change', 'pct_chg', 'volume', 'money']].copy()

    def etf_history_d(self,code,start_date=(dt.datetime.now()-dt.timedelta(days=1500)),end_date=dt.datetime.now()):
        """单只ETF指定时间K线"""
        df=self.ts_pro.fund_daily(ts_code=Code(code).tushare(), start_date=DT(start_date).tushare(), end_date=DT(end_date).tushare())
        df['datetime']=pd.to_datetime(df['trade_date'],format='%Y%m%d')
        #df['code']=self.jq.normalize_code(df['ts_code'].tolist())
        df['volume']=df['vol']*100
        df['money']=df['amount']*1000
        return df.set_index('datetime')[['pre_close', 'open', 'high', 'low', 'close','change', 'pct_chg', 'volume', 'money']].copy()

    def eft_history_date_all(self,date):
        """单日ETF所有K线"""
        df=self.ts_pro.fund_daily(trade_date=DT(date).tushare())
        df['datetime']=pd.to_datetime(df['trade_date'],format='%Y%m%d')
        df['code']=self.jq.normalize_code(df['ts_code'].tolist())
        df['volume']=df['vol']*100
        df['money']=df['amount']*1000
        return df[['datetime','code','pre_close', 'open', 'high', 'low', 'close','change', 'pct_chg', 'volume', 'money']].copy()
    
    def stock_instruments(self):
        """所有股票"""
        return self.sql_df(SQLS.stock_concepts())
    
    def etf_instruments(self):
        """所有ETF"""
        return self.sql_df(SQLS.etf_securities())    
    
    def stock_concepts(self):
        """所有股票概念"""
        return self.sql_df(SQLS.stock_concepts())

    def stock_valuation_all(self,date):
        """单交易日所有股票估值"""
        df=self.ts_pro.daily_basic(ts_code='', trade_date=DT(date).tushare())
        df['datetime']=pd.to_datetime(df['trade_date'],format='%Y%m%d')
        df['code']=self.jq.normalize_code(df['ts_code'].tolist())
        df['total_share']=df['total_share']*10000
        df['float_share']=df['float_share']*10000
        df['free_share']=df['free_share']*10000
        df['total_mv']=df['total_mv']*10000
        df['circ_mv']=df['circ_mv']*10000
        return df[[ 'datetime', 'code','close', 'turnover_rate', 'turnover_rate_f',
       'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
       'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv',
       'circ_mv']].copy()
        
    def stock_valuation_single(self,code,start_date=(dt.datetime.now()-dt.timedelta(days=1500)),end_date=dt.datetime.now()):
        """单股票指定时间段估值"""
        df=self.ts_pro.daily_basic(ts_code=Code(code).tushare(), start_date=DT(start_date).tushare(),end_date=DT(end_date).tushare())
        df['datetime']=pd.to_datetime(df['trade_date'],format='%Y%m%d')
        df['code']=self.jq.normalize_code(df['ts_code'].tolist())
        df['total_share']=df['total_share']*10000
        df['float_share']=df['float_share']*10000
        df['free_share']=df['free_share']*10000
        df['total_mv']=df['total_mv']*10000
        df['circ_mv']=df['circ_mv']*10000
        return df[[ 'datetime', 'code','close', 'turnover_rate', 'turnover_rate_f',
       'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
       'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv',
       'circ_mv']].copy()

    def stock_block_trade_all(self,date):
        """单日所有大宗交易"""
        df = self.ts_pro.block_trade(trade_date=DT(date).tushare())
        df['datetime']=pd.to_datetime(df['trade_date'],format='%Y%m%d')
        df['code']=self.jq.normalize_code(df['ts_code'].tolist())
        df['vol']=df['vol']*10000
        df['amount']=df['amount']*10000
        return df[[ 'datetime', 'code','price', 'vol', 'amount','buyer','seller']].copy()

    def stock_block_trade_single(self,code,start_date=(dt.datetime.now()-dt.timedelta(days=1500)),end_date=dt.datetime.now()):
        """指定股票所有大宗交易"""
        df = self.ts_pro.block_trade(ts_code=Code(code).tushare(),start_date=DT(start_date).tushare(),end_date=DT(end_date).tushare())
        df['datetime']=pd.to_datetime(df['trade_date'],format='%Y%m%d')
        df['code']=self.jq.normalize_code(df['ts_code'].tolist())
        df['vol']=df['vol']*10000
        df['amount']=df['amount']*10000
        return df[[ 'datetime', 'code','price', 'vol', 'amount','buyer','seller']].copy()

    def stock_holdernumber_single(self,code,start_date=(dt.datetime.now()-dt.timedelta(days=1500)),end_date=dt.datetime.now()):
        """ 单只股票历史股票人数"""
        df = self.ts_pro.stk_holdernumber(ts_code=Code(code).tushare(), start_date=DT(start_date).tushare(), end_date=DT(end_date).tushare())
        df['ann_date']=pd.to_datetime(df['ann_date'],format='%Y%m%d')
        df['end_date']=pd.to_datetime(df['end_date'],format='%Y%m%d')
        df['code']=self.jq.normalize_code(df['ts_code'].tolist())
        return df[['code','ann_date','end_date','holder_num']].copy()

    def currency_shibor(self,start_date=(dt.datetime.now()-dt.timedelta(days=1500)),end_date=dt.datetime.now()):
        df = self.ts_pro.shibor(start_date=DT(start_date).tushare(), end_date=DT(end_date).tushare())
        df['date']=pd.to_datetime(df['date'],format='%Y%m%d')
        return df
    def currency_m(self,start_date=(dt.datetime.now()-dt.timedelta(days=1500)),end_date=dt.datetime.now()):
        df = self.ts_pro.cn_m(start_m=DT(start_date).tushare()[0:6], end_m=DT(end_date).tushare()[0:6])
        df['month']=pd.to_datetime(df['month'],format='%Y%m')
        return df

        
    
    
    
    
  

        
    


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





