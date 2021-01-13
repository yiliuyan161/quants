import numpy as np
import pandas as pd
import datetime as dt


class Chan(object):

    #高低分型 
    @staticmethod
    def fenxing(arr):
        row_count=arr.shape[0] #行数
        time,high,low=0,3,4  #给位置命名，方便阅读
        time,price,cat,index=0,1,2,3 
        ret=np.zeros((int(row_count),4),dtype=np.float64) #申请空间存放结果
        k=0
        #发现高低点
        for i in np.arange(1,row_count-1):
            if arr[i,high]>=arr[i-1,high] and arr[i,high]>=arr[i+1,high]:
                #前一个是高点，如果这次更高，覆盖上一个
                if k>0 and ret[k-1,cat]==1:
                    if arr[i,high]>ret[k-1,price]:
                        ret[k-1,time]=arr[i,time]
                        ret[k-1,price]=arr[i,high]
                        ret[k-1,index]=i
                else: 
                    #前面没有枢纽点或者前一个是低点
                    if (k>0 and arr[i,high]>ret[k-1,price]) or k==0:
                        ret[k]=np.array([arr[i,time],arr[i,high],1,i])
                        k=k+1
            if arr[i,low]<=arr[i-1,low] and arr[i,low]<=arr[i+1,low]:
                 #前一个是低点，如果这次更低，覆盖上一个
                if k>0 and ret[k-1,cat]==-1:
                    if arr[i,low]<ret[k-1,price]:
                        ret[k-1,time]=arr[i,time]
                        ret[k-1,price]=arr[i,low]
                        ret[k-1,index]=i
                else:
                    #前面没有枢纽点或者前一个是高点
                    if (k>0 and arr[i,low]<ret[k-1,price]) or k==0:
                        ret[k]=np.array([arr[i,time],arr[i,low],-1,i])
                        k=k+1
                
        #最后价格特殊处理，最终点
        if ret[k-1,cat]>0 and arr[-1,high]<ret[k-1,price]: #最后价格小于最后高点
            ret[k]=np.array([arr[-1,time],arr[-1,low],-ret[k-1,cat],arr.shape[0]-1])
        elif ret[k-1,cat]>0 and arr[-1,high]>=ret[k-1,price]:#最后高点后创了新高
            ret[k-1]=np.array([arr[-1,time],arr[-1,high],ret[k-1,cat],arr.shape[0]-1])
        elif ret[k-1,cat]<0 and arr[-1,low]>ret[k-1,price]: #最后价格小于最后低点
            ret[k]=np.array([arr[-1,time],arr[-1,high],-ret[k-1,cat],arr.shape[0]-1])
        elif ret[k-1,cat]>0 and arr[-1,high]>=ret[k-1,price]:#最后低点后创了新低
            ret[k-1]=np.array([arr[-1,time],arr[-1,low],ret[k-1,cat],arr.shape[0]-1])

        ret_left=ret[:k+1]
        arr_raw=ret_left[~(ret_left==0).any(axis=1)] 
        return arr_raw

    #分型连线
    @staticmethod
    def bi(arr):
        point_time,point_price,point_cat,point_index=0,1,2,3
        bi=np.empty((arr.shape[0]-1,7),np.float64)
        for i in np.arange(1,arr.shape[0]):
            bi[i-1]=np.array([arr[i-1,point_time],arr[i,point_time],arr[i-1,point_price],arr[i,point_price],arr[i,point_cat],arr[i-1,point_index],arr[i,point_index]])
        return bi

    #分型合并成线段
    @staticmethod
    #@jit('float64[:,:](float64[:,:],int32)')
    def xianduan(arr,max_combine_segs_length=15):
        start_time,end_time,start_price,end_price,cat,start_index,end_index=0,1,2,3,4,5,6  #位置命名，方便阅读
        segments=np.empty((arr.shape[0],7),np.float64) #线段存放申请内存
        #第一笔放入线段集合
        segments[0],segments[1]=arr[0],arr[1]
        k=1 #用于记录segments最新元素下标
        for i in np.arange(2,arr.shape[0]):
            segments[k+1]=arr[i]
            k=k+1
            #最近3条线段，连续两条长度最小值<max_combine_segs_length
            week_seg=(min(segments[k,end_index]-segments[k-1,start_index],segments[k-1,end_index]-segments[k-2,start_index]))<max_combine_segs_length
            #最近3条线段能合并成1条向上线段
            up_trend=segments[k,cat]>0 and segments[k,end_price]>=segments[k-2,end_price] and segments[k,start_price]>=segments[k-2,start_price]
            #最近3条线段能合并成1条向下线段
            dn_trend=segments[k,cat]<0 and segments[k,end_price]<=segments[k-2,end_price] and segments[k,start_price]<=segments[k-2,start_price]
            while (up_trend or dn_trend )and (week_seg):
                #合并最近三条线段
                segments[k-2,end_time]=segments[k,end_time]
                segments[k-2,end_price]=segments[k,end_price]
                segments[k-2,end_index]=segments[k,end_index]
                k=k-2
                #检查合并后的最后三条线段，是否还可以合并
                week_seg=(min(segments[k,end_index]-segments[k-1,start_index],segments[k-1,end_index]-segments[k-2,start_index]))<max_combine_segs_length
                up_trend=segments[k,cat]>0 and segments[k,end_price]>=segments[k-2,end_price] and segments[k,start_price]>=segments[k-2,start_price]
                dn_trend=segments[k,cat]<0 and segments[k,end_price]<=segments[k-2,end_price] and segments[k,start_price]<=segments[k-2,start_price]
        ret_left=segments[:k+1]            
        arr_raw=ret_left[~np.isnan(ret_left).any(axis=1)] 
        return arr_raw
    #线段重合确定中枢
    @staticmethod
    def zhongshu(xd):
        """
        接收线段 [[start_time,end_time,start_price,end_price,cat,start_index,end_index,status]]
        返回中枢 [[start_time,end_time,start_price,end_price,cat,order,status,range_high,range_low,start_index,end_index,volume]]
        定义：
            中枢至少cross_num条线段重叠区域构成
            中枢范围,起点和终点在内所有线段端点的最大值，最小值
            中枢的起点，终点 第一个重叠线段的终点，最后一个重叠线段的起点
        处理逻辑：
            中枢集合为空，或者最后一个中枢status为1(已经结束)
                找到第一个连续三个线段重合的,增加中枢status为0(没有完成)
            最后一个中枢status为0
                计算当前线段与最后一个中枢的重合区间,
                    有重合，更新前面中枢的高低范围,终点
                    没有重合，最后一个中枢结束，status为1，检查当前线段和前两个线段是否构成中枢
        """
        xd_start_time,xd_end_time,xd_start_price,xd_end_price,xd_cat,xd_start_index,xd_end_index,xd_status=0,1,2,3,4,5,6,7  #位置命名，方便阅读
        zs_start_time,zs_end_time,zs_start_price,zs_end_price,zs_cat,zs_order,zs_status,zs_range_high,zs_range_low,zs_start_index,zs_end_index,zs_volume=0,1,2,3,4,5,6,7,8,9,10,11
        default_zs_cat,default_zs_order,default_zs_status,default_zs_volume=0,0,0,0 
        zones=np.empty((xd.shape[0],12),np.float64) #线段存放申请内存
        last_xd_index=0
        k=0 #k==中枢长度
        for i in np.arange(2,xd.shape[0]):
            #前面有没有完成的中枢，有的话检查是否能扩展中枢
            #扩展了，更新最近中枢的结尾，检查下一个线段
            #没扩展，最近中枢结束，检查当前线段跟前两条是不是构成中枢
            if k>0 and zones[k-1,zs_status]==0:
                max1=max(xd[i,xd_start_price],xd[i,xd_end_price])
                min1=min(xd[i,xd_start_price],xd[i,xd_end_price])
                zone_max=min(max1,zones[k-1,zs_start_price])
                zone_min=max(min1,zones[k-1,zs_end_price])
                if zone_max>zone_min: # 中枢没有完成,扩展中枢到当前线段开始
                    zones[k-1,zs_end_time]=xd[i,xd_start_time]
                    zones[k-1,zs_range_high]=max(zones[k-1,zs_range_high],max(xd[i-1,xd_start_price],xd[i-1,xd_end_price]))
                    zones[k-1,zs_range_low]=min(zones[k-1,zs_range_low],min(xd[i-1,xd_start_price],xd[i-1,xd_end_price]))
                    zones[k-1,zs_end_index]=xd[i,xd_start_index]
                    #中枢range_high low 重合
                    #
                    continue
                else:
                    zones[k-1,zs_status]=1  #中枢完成，检查当前线段和前两个线段是否构成新中枢
                    last_xd_index=i
                    
            if k==0 or zones[k-1,zs_status]==1:
                #中枢起点 第一条线段终点，最后一条线段起点
                #最近4条线段重合
                max0=(max(xd[i,xd_start_price],xd[i,xd_end_price]))
                min0=(min(xd[i,xd_start_price],xd[i,xd_end_price]))
                max1=(max(xd[i-1,xd_start_price],xd[i-1,xd_end_price]))
                min1=(min(xd[i-1,xd_start_price],xd[i-1,xd_end_price]))
                max2=(max(xd[i-2,xd_start_price],xd[i-2,xd_end_price]))
                min2=(min(xd[i-2,xd_start_price],xd[i-2,xd_end_price]))
                max3=(max(xd[i-3,xd_start_price],xd[i-3,xd_end_price]))
                min3=(min(xd[i-3,xd_start_price],xd[i-3,xd_end_price]))
                zone_high=min(max0,max1,max2,max3)
                zone_min=max(min0,min1,min2,min3)
                range_high=max(max1,max2)
                range_low=min(min1,min2)
                if zone_high>zone_min and (zone_high-zone_min)/(range_high-range_low)>0.10:
                    #形成中枢
                    zones[k]=np.array([xd[i-2,xd_start_time],xd[i,xd_start_time],min(max0,max1,max2),max(min0,min1,min2),\
                            default_zs_cat,default_zs_order,default_zs_status,range_high,range_low,xd[i-2,xd_start_index],xd[i,xd_start_index],default_zs_volume])
                    k=k+1
        ret_left=zones[:k]            
        xd_raw=ret_left[~np.isnan(ret_left).any(axis=1)]
        return xd_raw
    #低级中枢确定高级中枢范围
    @staticmethod
    def zhongshu_fix(zs1,zs2):
        """
        低级别中枢，确定高级别中枢范围,中枢开始结束时间对齐，便于下一步删除时间范围相同中枢
        """
        #
        zs2_ret=zs2.copy()
        time,opn,high,low,close,volume=0,1,2,3,4,5
        zs_start_time,zs_end_time,zs_start_price,zs_end_price,zs_cat,zs_order,zs_status,zs_range_high,zs_range_low,zs_start_index,zs_end_index,zs_volume=0,1,2,3,4,5,6,7,8,9,10,11

        l1,l2=0,0
        while l1<zs1.shape[0] and l2<zs2.shape[0]:
            #跟当前子中枢重合，跟前一个子中枢不重合，设置中枢开始为子中枢开始
            #跟当前子中枢重合，跟后一个子中枢不重合,设置中枢结束为子中枢结束
            # k-1级别中枢和k级别中枢重合
            time_chonghe_cur=max(zs2[l2,zs_start_index],zs1[l1,zs_start_index])<=min(zs2[l2,zs_end_index],zs1[l1,zs_end_index])
            time_not_chonghe_pre= l1>0 and max(zs2[l2,zs_start_index],zs1[l1-1,zs_start_index])>min(zs2[l2,zs_end_index],zs1[l1-1,zs_end_index])
            range_chonghe_cur=min(zs2[l2,zs_start_price],zs1[l1,zs_start_price])>=max(zs2[l2,zs_end_price],zs1[l1,zs_end_price])
            if time_chonghe_cur:
                if range_chonghe_cur:
                    if l1==0 or time_not_chonghe_pre:
                        #当前中枢和前一个低层级中枢时间上没有重合
                        zs2_ret[l2,zs_start_index]=zs1[l1,zs_start_index]
                        zs2_ret[l2,zs_start_time]=zs1[l1,zs_start_time]
                    zs2_ret[l2,zs_end_index]=zs1[l1,zs_end_index]
                    zs2_ret[l2,zs_end_time]=zs1[l1,zs_end_time]
                l1=l1+1
            elif zs1[l1,zs_start_index]>zs2[l2,zs_end_index]:
                #k-1级中枢开始>当前k级中枢结束
                l2=l2+1
            elif zs1[l1,zs_end_index]<zs2[l2,zs_start_index]:
                # k-1级中枢结束<k级中枢开始
                l1=l1+1
        
        return zs2_ret
    #给中枢加上成交量信息
    @staticmethod
    #@jit('float64[:,:](float64[:,:],float64[:,:])')
    def zones_add_volume(zs,price):
        """
        给中枢添加成交量信息
        """
        time,opn,high,low,close,volume=0,1,2,3,4,5
        zs_start_time,zs_end_time,zs_start_price,zs_end_price,zs_cat,zs_order,zs_status,zs_range_high,zs_range_low,zs_start_index,zs_end_index,zs_volume=0,1,2,3,4,5,6,7,8,9,10,11
        for i in np.arange(0,zs.shape[0]):
            idx_start=int(zs[i,zs_start_index])
            idx_end=int(zs[i,zs_end_index])
            zs[i,zs_volume]=price[idx_start:idx_end+1,volume].sum()
        return zs
    
    
    #合并中枢，对中枢按照开始结束时间去重
    @staticmethod
    def drop_duplicate_zones(zone_list):
        """
        删除重复的中枢
        """
        arrs= np.concatenate(zone_list)
        df=pd.DataFrame(arrs,columns=['start_time','end_time','start_price','end_price','cat','order','status','range_high','range_low','start_index','end_index','zone_volume'])
        return df.drop_duplicates(subset=['start_index','end_index'],keep='first')

    # 给中枢添加方向，序号，波动区间，前后趋势变化百分比，区间长度等信息
    @staticmethod
    def zones_add_extra_meta(df):
        zs=df.values
        # 中枢添加顺序，波动范围，中枢距离等信息
        ret_arr=np.zeros((zs.shape[0],9),np.float64)
        zs_start_time,zs_end_time,zs_start_price,zs_end_price,zs_cat,zs_order,zs_status,zs_range_high,zs_range_low,zs_start_index,zs_end_index,zs_volume=0,1,2,3,4,5,6,7,8,9,10,11
        z_start_time,z_start_price,z_end_time,z_end_price,z_cat,z_order,z_width,z_range,z_distance=0,1,2,3,4,5,6,7,8
        for i in np.arange(0,zs.shape[0]):
            ret_arr[i,z_start_time]=zs[i,zs_start_time]
            ret_arr[i,z_start_price]=zs[i,zs_start_price]
            ret_arr[i,z_end_time]=zs[i,zs_end_time]
            ret_arr[i,z_end_price]=zs[i,zs_end_price]
            ret_arr[i,z_width]=zs[i,zs_end_index]-zs[i,zs_start_index]
            if i>0:
                ret_arr[i,z_cat]= 1 if zs[i,zs_start_price]>zs[i-1,zs_start_price] else -1
                if ret_arr[i-1,z_cat]==ret_arr[i,z_cat]:
                    ret_arr[i,z_order]=ret_arr[i-1,z_order]+1
                else:
                    ret_arr[i,z_order]=1
                pre_zone_center=(zs[i-1,zs_start_price]+zs[i-1,zs_end_price])
                ret_arr[i,z_distance]=((zs[i,zs_start_price]+zs[i,zs_end_price])/pre_zone_center-1) if pre_zone_center>0 else 0
            ret_arr[i,z_range]=(zs[i,zs_range_high]-zs[i,zs_range_low])/(zs[i,zs_range_high] if ret_arr[i,z_cat]<0 else zs[i,zs_range_low])
        return pd.DataFrame(ret_arr,columns=['start_time','start_price','end_time','end_price','cat','order','width','range','distance'])
    
    # 按照时间范围，把中枢分成多个集合
    @staticmethod
    def group_zones(df_zones,time_ranges=[36,180,900,3600]):
        """
        产生time_ranges 个数+1个范围
        """
        dfs=[]
        df_zones['index_range']=df_zones['end_index']-df_zones['start_index']
        for i in np.arange(0,len(time_ranges)):
            if i>0:
                start=time_ranges[i-1]
            else:
                start=0
            df_filtered=Chan.zones_add_extra_meta(df_zones[(df_zones['index_range']>=start)&(df_zones['index_range']<time_ranges[i])])
            dfs.append(df_filtered)
        df_filtered=Chan.zones_add_extra_meta(df_zones[(df_zones['index_range']>=time_ranges[-1])])
        dfs.append(df_filtered)
        return dfs

    # 中枢ndarray转成dataframe
    @staticmethod
    def zhongshu_array2df(zones):
        df_zones=pd.DataFrame(zones,columns=['start_time','end_time','start_price','end_price','cat','order','status','range_high','range_low','start_index','end_index','zone_volume'])
        df_zones.start_time=pd.to_datetime(df_zones.start_time).dt.strftime('%Y%m%d%H%M')
        df_zones.end_time=pd.to_datetime(df_zones.end_time).dt.strftime('%Y%m%d%H%M')
        return df_zones
    
    # 线段ndarray转成dataframe
    @staticmethod
    def xd_array2df(bis):
        df_bis=pd.DataFrame(bis,columns=['start_time','end_time','start_price','end_price','cat','start_index','end_index'])
        df_bis.start_time=pd.to_datetime(df_bis.start_time).dt.strftime('%Y%m%d%H%M')
        df_bis.end_time=pd.to_datetime(df_bis.end_time).dt.strftime('%Y%m%d%H%M')
        return df_bis

    #一次获得笔，线段，中枢dataframe
    @staticmethod
    def bi_xianduan_zhongshu_df(df,combine_seg_less_than=5):
        bis,segs,zones=Chan.bi_xianduan_zhongshu_array(df,combine_seg_less_than)
        return Chan.xd_array2df(bis),Chan.xd_array2df(segs),Chan.zhongshu_array2df(zones)
    
    #一次获得笔，线段，中枢dataframe
    @staticmethod
    def bi_xianduan_zhongshu_array(df,combine_seg_less_than=5):
        df['date']=pd.to_datetime(df['date']).astype(np.int64).astype(np.float64)
        arr=df[['date','open','close','high','low','volume']].values
        bis=Chan.bi(Chan.fenxing(arr))
        segs=Chan.xianduan(bis,combine_seg_less_than)
        zones=Chan.zhongshu(segs)
        return bis,segs,zones

    #获得线段中枢
    @staticmethod
    def xianduan_zhongshu_array(bis,combine_seg_less_than=5):
        segs=Chan.xianduan(bis,combine_seg_less_than)
        zones=Chan.zhongshu(segs)
        return segs,zones
    #获得线段中枢
    @staticmethod
    def xianduan_zhongshu_df(bis,combine_seg_less_than=5):
        segs=Chan.xianduan(bis,combine_seg_less_than)
        zones=Chan.zhongshu(segs)
        return Chan.xd_array2df(segs),Chan.zhongshu_array2df(zones)

    #批量获得线段中枢
    @staticmethod
    def xianduan_zhongshu_list(bis,seg_limits=[5,20,50,100,250]):
        seg_list=[]
        zone_list=[]
        for seg_limit in seg_limits:
            segs,zones=Chan.xianduan_zhongshu_array(bis,seg_limit)
            seg_list.append(segs)
            zone_list.append(zones)
        return seg_list,zone_list
        
    #中枢对齐，去重，按时间长度重新分组
    @staticmethod
    def zhongshu_fixRange_groupbyTime(zones_list,zone_levels=[36,180,900,3600]):
        """
        利用低级中枢，修正高级中枢范围(对齐)
        删除开始结束时间相同的中枢
        根据中枢时间范围，重新划分中枢等级
        """
        zones_fix_list=[] #修正过范围的中枢  
        zones_fix_list.append(zones_list[0])
        for i in range(1,len(zones_list)):
            zones_fix=Chan.zhongshu_fix(zones_fix_list[i-1],zones_list[i])
            zones_fix_list.append(zones_fix)
        df_zones=Chan.drop_duplicate_zones(zones_fix_list)
        df_zone_list=Chan.group_zones(df_zones,time_ranges=zone_levels)
        return df_zone_list

    @staticmethod
    def zones(df,unit='5m'):
        df['date']=pd.to_datetime(df['date']).astype(np.int64).astype(np.float64)
        arr=df[['date','open','close','high','low','volume']].values
        bis=Chan.bi(Chan.fenxing(arr))
        seg_limits,zone_levels,level_names=Chan.levels(unit=unit)
        #ndarray
        seg_list,zone_list= Chan.xianduan_zhongshu_list(bis,seg_limits=seg_limits)
        #df
        zone_regroup_list=Chan.zhongshu_fixRange_groupbyTime(zones_list=zone_list,zone_levels=zone_levels)
        #ndarray,df,list
        return seg_list,zone_regroup_list,level_names
            
    @staticmethod        
    def levels(unit='5m'):
        #1m 5m 15m 30m 1h 1d 1w 1m
        unit_array=[1,5,3,2,2,4,5,4] #时间单位乘数
        unit_names=['1m','5m','15m','30m','60m','1d','1w','1M'] #时间单位名称
        min_segs=4 #线段最小构成的bar数量
        start=unit_names.index(unit)
        seg_limits=[min_segs]
        time_limits=[min_segs*3]
        cur_seg=min_segs
        cur_zone_range=min_segs*3
        for ratio in unit_array[start+1:]:
            cur_seg=cur_seg*ratio
            cur_zone_range=cur_zone_range*ratio
            seg_limits.append(cur_seg)
            time_limits.append(cur_zone_range)
        return seg_limits,time_limits[1:],unit_names[start:]
        
__all__ = ["Chan"]