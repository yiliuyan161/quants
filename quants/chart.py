from pyecharts import options as opts
from pyecharts.charts import *
from pyecharts.commons.utils import JsCode
import pyecharts.globals as g
import pandas as pd
import datetime as dt
import numpy as np
from math import sqrt
from scipy import stats

g.WarningType.ShowWarning = False

def count9(values):
    arr=np.zeros((values.shape[0],2),dtype=np.int32)
    for i in np.arange(5,values.shape[0]):
        if values[i]>values[i-4]:
            arr[i,0]=arr[i-1,0]+1
        elif values[i]<values[i-4]:
            arr[i,1]=arr[i-1,1]+1
    return arr


def line_reg(df,col):
    x=np.arange(0,len(df))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, df[col].values)
    predicted_price = x * slope + intercept
    residuals = (df[col].values - predicted_price) ** 2
    stderr = sqrt(sum(residuals) / len(df))
    up1=predicted_price+stderr
    up2=predicted_price+2*stderr
    dn1=predicted_price-stderr
    dn2=predicted_price-2*stderr
    df['p_price']=predicted_price
    df['up1']=up1
    df['up2']=up2
    df['dn1']=dn1
    df['dn2']=dn2
    return df

def chart_line(df, x_col=None, y_cols=[],col_names={},line_width=1, width="100%", height="500px", title='',colors=['red','orange','yellow','green','olive','blue','purple']):
    """
    线图
    :param x_col: x轴数据对应列
    :param y_cols: 多个y对应的列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    if x_col == None:
        if str(df.index.dtype) == 'datetime64[ns]':
            xaxis_data = df.index.strftime('%Y%m%d%H%M').values.tolist()
        else:
            xaxis_data = df.index.values.tolist()
    else:
        xaxis_data = __fixdatetime(df, x_col).values.tolist()

    line = Line(init_opts=opts.InitOpts(width=width, height=height)) \
        .set_global_opts(title_opts=opts.TitleOpts(title=title), \
                         yaxis_opts=opts.AxisOpts(is_scale=True),datazoom_opts=opts.DataZoomOpts()) \
        .add_xaxis(xaxis_data)
    i=0
    for col in y_cols:
        if col in col_names:
            title=col_names[col]
        else:
            title=col
        line.add_yaxis(title, df[col].values.tolist(), label_opts=opts.LabelOpts(is_show=False),linestyle_opts=opts.LineStyleOpts(color=colors[i%len(colors)],width=line_width))
        i=i+1
    return line

def chart_kline(df, opn='open', high='high', low='low', close='close',
          volume='volume', kline_overlaps=[],mark_area=[],indicator=None, mas=[5, 10, 21],colors=[], width="100%", height='600px', title=""):
    """
     K线图
    :param datetime: "datetime"
    :param open:"open"
    :param high:"high"
    :param low:"low"
    :param close:"close"
    :param volume:"volume"
    :param mark:None
    :param mas:[5,10,21]
    :param colors:[]
    :param width: "100%"
    :param height: "600px"
    :param title: ""
    :return:
    """
    tooltip_formatter = JsCode("""
     function(params){
        var text='<span>'+params[0].name+'</span>';
        var s_dict={};
        for(var i in params){
            s_dict[params[i].seriesName]=params[i];
        }
        for (var j in s_dict){
            var s=s_dict[j];
            var data=s.data;
            var names=s.dimensionNames;
            text=text+'<br/><span style="color:lightblue;">'+s.seriesName+'</span>';
            if((!(data instanceof Array))){
                text=text+'<span>'+':'+data+'</span>';
            }else if(data instanceof Array & data.length==2){
                text=text+'<span>'+':'+data[1]+'</span>';
            }else if(data instanceof Array & data.length>2){
                for (var k in names){
                    if(k>0 && data instanceof Array){
                        text=text+'<br/><span>'+names[k]+':'+data[k]+'</span>';
                    }
                }
            } 
        }
        return text;
     }
    """)
    if str(df.index.dtype) == 'datetime64[ns]':
        xaxis_data = df.index.strftime('%Y%m%d%H%M').values.tolist()
    else:
        xaxis_data = df.index.values.tolist()
    
    if indicator==None:
        xaxis_index_zoom=[0,1]
    else:
        xaxis_index_zoom=[0,1,2]
    kline = Kline().add_xaxis(xaxis_data=xaxis_data).add_yaxis(
        series_name="K线",
        y_axis=df[[opn, close, low, high]].values.tolist(),
        itemstyle_opts=opts.ItemStyleOpts(
            color="#ef232a",
            color0="#14b143",
            border_color="#ef232a",
            border_color0="#14b143",
        ),

        markpoint_opts=opts.MarkPointOpts(
            label_opts=opts.LabelOpts(position="inside", color="#fff"),
            data=[
                opts.MarkPointItem(type_='max', name="最大值"),
                opts.MarkPointItem(type_='min', name="最小值")
            ]
        ),
    ).set_global_opts(
        title_opts=opts.TitleOpts(title=title, pos_left="0",pos_top="0"),
        legend_opts= opts.LegendOpts(type_='scroll',pos_left='0',pos_top='30'),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            is_scale=True,
            boundary_gap=False,
            axisline_opts=opts.AxisLineOpts(is_on_zero=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
            split_number=20,
            min_="dataMin",
            max_="dataMax",
        ),
        yaxis_opts=opts.AxisOpts(
            is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line"
                                      , formatter=tooltip_formatter
                                      ),
        datazoom_opts=[
            opts.DataZoomOpts(is_show=False, type_="inside", xaxis_index=[0, 0], range_end=100),
            opts.DataZoomOpts(is_show=True, xaxis_index=xaxis_index_zoom, pos_top="97%", range_end=100),
        ],
        # 三个图的 axis 连在一块
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True,
            link=[{"xAxisIndex": "all"}],
            label=opts.LabelOpts(background_color="#777"),
        ),
    )
    if len(mark_area)>0:
        kline.set_series_opts(
            markarea_opts=opts.MarkAreaOpts(is_silent=True, data=mark_area)
        )

    color_num=1
    if len(kline_overlaps) > 0:
        for chart in kline_overlaps:
            kline = kline.overlap(chart)
            if len(colors)>color_num:
                kline.colors[color_num]=colors[color_num-1] 
            
    line = Line().add_xaxis(xaxis_data=xaxis_data)
    for ma_len in mas:
        line.add_yaxis(
            series_name="MA" + str(ma_len),
            y_axis=df[close].rolling(ma_len).mean().round(2).values.tolist(),
            linestyle_opts=opts.LineStyleOpts(opacity=0.5, width=2),
            label_opts=opts.LabelOpts(is_show=False)
        )
    
    # Overlap Kline + Line
    overlap_kline_line = kline.overlap(line)
    bar_volume = (
        Bar()
            .add_xaxis(xaxis_data=xaxis_data)
            .add_yaxis(
            series_name="成交量",
            y_axis=list((df[volume]/1000000).round(2).values),
            xaxis_index=0,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
        ).set_global_opts(
            legend_opts=opts.LegendOpts(is_show=False),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}M")),
            xaxis_opts=opts.AxisOpts(is_show=False)
        )
    )
    # 最后的 Grid
    grid_chart = Grid(init_opts=opts.InitOpts(width=width, height=height))
    
    if indicator==None:
        # K线图和 MA5 的折线图
        grid_chart.add(
            overlap_kline_line,
            grid_opts=opts.GridOpts(pos_left="7%", pos_right="1%", height="75%"),
            grid_index=0,
        )
        grid_chart.add(
            bar_volume,
            grid_opts=opts.GridOpts(
                pos_left="7%", pos_right="1%", pos_top="80%", height="15%"),
            grid_index=1
        )
    else:
        grid_chart.add(
            overlap_kline_line,
            grid_opts=opts.GridOpts(pos_left="7%", pos_right="1%", height="60%"),
            grid_index=0
        )
        grid_chart.add(
            bar_volume,
            grid_opts=opts.GridOpts(
                pos_left="7%", pos_right="1%", pos_top="60%", height="20%"),
            grid_index=1
        )
        grid_chart.add(
            indicator,
            grid_opts=opts.GridOpts(
                pos_left="7%", pos_right="1%", pos_top="80%", height="20%"),
            grid_index=2
        )
    return grid_chart


def chart_kline_indicator(df,chart_type='line',cols={}):
    if str(df.index.dtype) == 'datetime64[ns]':
        xaxis_data = df.index.strftime('%Y%m%d%H%M').values.tolist()
    else:
        xaxis_data = df.index.values.tolist()
    if chart_type=='line':
        chart=Line()
    else:
        chart=Bar()
    line = chart.add_xaxis(xaxis_data=xaxis_data)
    for col in cols.keys():
        line.add_yaxis(
            series_name=cols[col],
            y_axis=df[col].values.tolist(),
            #linestyle_opts=opts.LineStyleOpts(opacity=1, width=1),
            label_opts=opts.LabelOpts(is_show=False),
            xaxis_index=0,
            yaxis_index=2
        )
    return line

def chart_kline_overlap_tds(df):
    """
    添加神奇9转指标
    """
    df2=df.merge(pd.DataFrame(count9(df['close'].values),columns=['up9','dn9'],index=df.index),left_index=True,right_index=True)
    df2['day']=pd.to_datetime(df2.index).strftime("%Y%m%d%H%M")
    df_up9=df2[df2['up9']>0][['day','close','up9']]
    df2['上涨序列']=df2['close']
    df2['下跌序列']=df2['close']
    data_up9=[]
    for index,row in df_up9.iterrows():
        data_up9.append({'coord':[row['day'],row['close']],'value':row['up9']})
        
    df_dn9=df2[df2['dn9']>0][['day','close','dn9']]
    data_dn9=[]
    for index,row in df_dn9.iterrows():
        data_dn9.append({'coord':[row['day'],row['close']],'value':row['dn9']})
        
    line_up9=chart_line(df2.iloc[0:2],x_col='day',y_cols=['上涨序列'],colors=['red'])\
        .set_series_opts(
            markpoint_opts=opts.MarkPointOpts(symbol_size=1,\
            label_opts=opts.LabelOpts(position="top", color="red",distance=15),\
            data=data_up9)\
            )
    line_dn9=chart_line(df2.iloc[0:2],x_col='day',y_cols=['下跌序列'],colors=['black'])\
        .set_series_opts(
            markpoint_opts=opts.MarkPointOpts(symbol_size=1,\
            label_opts=opts.LabelOpts(position="bottom", color="green",distance=15),\
            data=data_dn9)\
            )
    return [line_up9,line_dn9]

    
def chart_kline_overlap_trend(df):
    """
    添加长期趋势线
    """
    df_width_trend=line_reg(df,'close')
    trend_lines=chart_line(df_width_trend,y_cols=['p_price','up1','up2','dn1','dn2'])
    return trend_lines

def __fixdatetime(df, col):
    if str(df[col].dtype) == 'datetime64[ns]':
        return df[col].dt.strftime('%Y%m%d%H%M')
    else:
        return df[col]

def chart_kline_overlap_zone(df_segs,df_zones,name,color,font_size=12,show_label=True,show_area=True):
    """
    添加中枢
    """
    zone_level2=[]
    zone_line2=[]
    z_start_time,z_start_price,z_end_time,z_end_price,z_cat,z_order,z_width,z_range,z_distance=0,1,2,3,4,5,6,7,8
    arr=df_zones.values
    for i in np.arange(0,arr.shape[0]):
        zone_level2.append([{'coord':[pd.to_datetime(arr[i,z_start_time]).strftime("%Y%m%d%H%M"),arr[i,z_start_price]]},{'coord':[pd.to_datetime(arr[i,z_end_time]).strftime("%Y%m%d%H%M"),arr[i,z_end_price]]}])
        zone_line2.append([{'coord':[pd.to_datetime(arr[i,z_start_time]).strftime("%Y%m%d%H%M"),arr[i,z_start_price]],\
                        'value':str(int(arr[i,z_cat]*arr[i,z_order]))+" #"+str(int(arr[i,z_width]))+\
                            " @"+str(round(arr[i,z_range]*100,2))+"%,$"+str(round(arr[i,z_distance]*100,2))+"%"
                        },{'coord':[pd.to_datetime(arr[i,z_end_time]).strftime("%Y%m%d%H%M"),arr[i,z_start_price]]}])
    df_segs[name]=df_segs['end_price']
    seg=chart_line(df_segs,x_col='end_time',y_cols=['end_price'],col_names={'end_price':name},colors=[color])
    if show_label:
        seg.set_series_opts(
            markline_opts=opts.MarkLineOpts(is_silent=False,symbol=["circle", "none"], data=zone_line2,\
                                                        label_opts=opts.LabelOpts(position="middle", color=color, font_size=font_size),\
                                            linestyle_opts=opts.LineStyleOpts(color=color))
                    )
    if show_area:
        seg.set_series_opts(
            markarea_opts=opts.MarkAreaOpts(is_silent=False, data=zone_level2,itemstyle_opts={'color':color,'opacity':0.15})
                    )
    return seg



def chart_kline_overlap_signal(title,df_target,df_signal,cols={'x':'date','y':'price','value':'value'},symbol_index=0):
    symbols=['circle', 'rect', 'roundRect', 'triangle', 'diamond', 'pin', 'arrow']
    signal_data=[]
    for index,row in df_signal.iterrows():
        signal_data.append({'coord':[pd.to_datetime(row[cols['x']]).strftime("%Y%m%d%H%M"),row[cols['y']] ],'value':row[cols['value']]})
    line1=chart_line(df_target.iloc[:2],y_cols=['close'],col_names={'close':title})\
        .set_series_opts(markpoint_opts=opts.MarkPointOpts(\
                            data=signal_data,symbol=symbols[symbol_index],symbol_size=12,label_opts=opts.LabelOpts(position="inside", color="black",font_size=12,rotate=45,distance='30%')))
    return line1

def chart_kline_overlap_line(title,df_target,df_signal,cols={'x1':'end_date','x2':'pub_date','y':'price','value':'value'},color='red',font_size=12):
    signal_data=[]
    for index,row in df_signal.iterrows():
        signal_data.append([{'coord':[pd.to_datetime(row[cols['x1']]).strftime("%Y%m%d%H%M"),row[cols['y']]],'value':row[cols['value']]},\
                              {'coord':[pd.to_datetime(row[cols['x2']]).strftime("%Y%m%d%H%M"),row[cols['y']]]} ])
    line1=chart_line(df_target.iloc[:2],y_cols=['close'],col_names={'close':title})\
        .set_series_opts(\
            markline_opts=opts.MarkLineOpts(\
                is_silent=False,symbol=["circle", "none"], data=signal_data,\
                    label_opts=opts.LabelOpts(position="middle", color=color, font_size=font_size),linestyle_opts=opts.LineStyleOpts(color=color))
         )
    return line1


def chart_radar(df, value_cols_max={}, width='300px', height='300px', title=""):
    """
    雷达图
    :param name_col: ""
    :param value_cols: {“roa”:100,"roe":100} 列名:最大值
    :param width:'300px'
    :param height:'300px'
    :param title:""
    :return:
    """
    colors = ['red', 'blue', 'green', 'orange', 'pink', 'yellow', 'black', 'maroon', 'purple', 'brown', 'olive', 'teal',
              'fuchsia', 'aqua']
    color_iter = iter(colors)
    radar = Radar(init_opts=opts.InitOpts(width=width, height=height)).add_schema(
        schema=[
            opts.RadarIndicatorItem(name=col, max_=value_cols_max[col]) for col in sorted(list(value_cols_max.keys()))
        ],
        splitarea_opt=opts.SplitAreaOpts(
            is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0)
        ),
        textstyle_opts=opts.TextStyleOpts(color="black"),
    )
    for index, row in df[sorted(list(value_cols_max.keys()))].iterrows():
        color = next(color_iter)
        radar.add(
            series_name=str(index),
            data=[row.values.tolist()],
            areastyle_opts=opts.AreaStyleOpts(opacity=0.1, color=color),
            linestyle_opts=opts.LineStyleOpts(color=color, width=2)
        )
    radar.set_series_opts(label_opts=opts.LabelOpts(is_show=False)).set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        legend_opts=opts.LegendOpts(type_='scroll', pos_top='20px', align='left')
    )
    return radar


def chart_sankey(df, source_col='', target_col='', value_col='', width='100%', height='600px', title=""):
    """
    桑基图
    :param source_col: 源节点列
    :param target_col: 目标节点列
    :param value_col: 值列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    nodes = [{"name": node} for node in list(set(df[source_col].unique()) | set(df[target_col].unique()))]
    links = [{'source': row[0], 'target': row[1], 'value': row[2]} for row in
             df[[source_col, target_col, value_col]].values.tolist()]
    sankey = Sankey(init_opts=opts.InitOpts(width=width, height=height)).add(
        title,
        nodes,
        links,
        linestyle_opt=opts.LineStyleOpts(opacity=0.3, curve=0.5, color="source"),
        label_opts=opts.LabelOpts(position="right"),
        pos_bottom='10%',
        node_align='justify',
        focus_node_adjacency=True,
        layout_iterations=0,
        node_width=10,
        node_gap=8
    ).set_global_opts(title_opts=opts.TitleOpts(title=title))
    return sankey


def chart_theme_river(df_source, date='date', value='value', category='category', width='100%', height='800px', title=""):
    """
    主题河流图
    :param date: 日期列
    :param value: 值列
    :param category: 类别列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    df=df_source.copy()
    df[date] = __fixdatetime(df, date)
    theme_river = ThemeRiver(init_opts=opts.InitOpts(width=width, height=height)) \
        .add(
        series_name=list(df[category].unique()),
        data=df[[date, value, category]].values.tolist(),
        singleaxis_opts=opts.SingleAxisOpts(
            pos_top="60", pos_bottom="50", type_="time"
        ),
    ).set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line"),
        legend_opts=opts.LegendOpts(selected_mode='multiple', pos_top='20'),
        title_opts=opts.TitleOpts(title=title)
    )
    return theme_river


def chart_area_stack(df, x_col=None, y_cols=[], width="100%", height="500px", title=''):
    """
    堆叠面积图
    :param x_col: x轴数据对应列
    :param y_cols: 多个y对应的列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    if x_col == None:
        if str(df.index.dtype) == 'datetime64[ns]':
            xaxis_data = df.index.strftime('%Y%m%d%H%M').values.tolist()
        else:
            xaxis_data = df.index.values.tolist()
    else:
        xaxis_data = __fixdatetime(df, x_col).values.tolist()

    line = Line(init_opts=opts.InitOpts(width=width, height=height)).set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis", axis_pointer_type="cross"),
        title_opts=opts.TitleOpts(title=title),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
    ).add_xaxis(xaxis_data=xaxis_data)
    for col in y_cols:
        line.add_yaxis(
            series_name=str(col),
            stack="总量",
            y_axis=df[col].values.tolist(),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
        )
    return line


def chart_bar_row(df, x_col=None, y_col='', width="100%", height="500px", title='',reversal_axis=False):
    if x_col == None:
        df=df.copy()
        x_col='x_col'
        if str(df.index.dtype) == 'datetime64[ns]':
            xaxis_data = df.index.strftime('%Y%m%d%H%M').values.tolist()
        else:
            xaxis_data = df.index.values.tolist()
        df[x_col]=xaxis_data
    else:
        xaxis_data = __fixdatetime(df, x_col).values.tolist()

    b = Bar(init_opts=opts.InitOpts(width=width, height=height)) \
        .set_global_opts(title_opts=opts.TitleOpts(title=title)) \
        .add_xaxis(xaxis_data)
    for index,row  in df.iterrows():
        b.add_yaxis(row[x_col], [row[y_col]], label_opts=opts.LabelOpts(is_show=False))
    if reversal_axis==True:
        b=b.reversal_axis()
    return b

def chart_bar(df, x_col=None, y_cols=[],col_names={}, width="100%", height="500px", title='',reversal_axis=False):
    """
    柱状图
    :param x_col: x轴数据对应列
    :param y_cols: 多个y对应的列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    if x_col == None:
        if str(df.index.dtype) == 'datetime64[ns]':
            xaxis_data = df.index.strftime('%Y%m%d%H%M').values.tolist()
        else:
            xaxis_data = df.index.values.tolist()
    else:
        xaxis_data = __fixdatetime(df, x_col).values.tolist()

    b = Bar(init_opts=opts.InitOpts(width=width, height=height)) \
        .set_global_opts(title_opts=opts.TitleOpts(title=title),datazoom_opts=opts.DataZoomOpts()) \
        .add_xaxis(xaxis_data)
    for col in y_cols:
        if col in col_names:
            title=col_names[col]
        else:
            title=col
        b.add_yaxis(title, df[col].values.tolist(), label_opts=opts.LabelOpts(is_show=False))
    if reversal_axis==True:
        b=b.reversal_axis()
    return b




def chart_bar_stack(df, x_col=None, y_cols=[],col_names={}, width="100%", height="500px", title='',reversal_axis=False):
    """
    堆叠柱状图
    :param x_col: x轴数据对应列
    :param y_cols: 多个y对应的列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    if x_col == None:
        if str(df.index.dtype) == 'datetime64[ns]':
            xaxis_data = df.index.strftime('%Y%m%d%H%M').values.tolist()
        else:
            xaxis_data = df.index.values.tolist()
    else:
        xaxis_data = __fixdatetime(df, x_col).values.tolist()

    b = Bar(init_opts=opts.InitOpts(width=width, height=height)) \
        .set_global_opts(title_opts=opts.TitleOpts(title=title),datazoom_opts=opts.DataZoomOpts()) \
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False)) \
        .add_xaxis(xaxis_data)
    for col in y_cols:
        if col in col_names:
            title=col_names[col]
        else:
            title=col
        b.add_yaxis(title, df[col].values.tolist(), stack="stack1", label_opts=opts.LabelOpts(is_show=False))
    if reversal_axis==True:
        b=b.reversal_axis()
    return b


def chart_map_province(df, province_col, value_col, width="100%", height="500px", title=''):
    """
    中国地图省级
    :param province_col: 省列
    :param value_col: 值列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    m = Map(init_opts=opts.InitOpts(width=width, height=height)) \
        .add(title, df[[province_col, value_col]].values.tolist(), "china") \
        .set_global_opts(title_opts=opts.TitleOpts(title=title),
                         visualmap_opts=opts.VisualMapOpts(max_=int(df[value_col].max())))
    return m


def chart_wordcloud(df, word_col, count_col, width="100%", height="500px", title=''):
    """
    词云图
    :param word_col: 关键词列
    :param count_col: 数量列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    word_count_dict = df[[word_col, count_col]].set_index(word_col)[count_col].to_dict()
    data = list(word_count_dict.items())
    wc = WordCloud(init_opts=opts.InitOpts(width=width, height=height)) \
        .add(series_name=title, data_pair=data, word_size_range=[8, 100], shape='circle', width=width,
             height=height, rotate_step=5) \
        .set_global_opts(
        title_opts=opts.TitleOpts(
            title=title, title_textstyle_opts=opts.TextStyleOpts(font_size=23)
        ),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
    return wc


def chart_scatter(df, x_col=None, y_cols=[], y_symbols=[], width="100%", height="500px", title=''):
    if x_col == None:
        if str(df.index.dtype) == 'datetime64[ns]':
            xaxis_data = df.index.strftime('%Y%m%d%H%M').values.tolist()
        else:
            xaxis_data = df.index.values.tolist()
    else:
        xaxis_data = __fixdatetime(df, x_col).values.tolist()
    s = Scatter(init_opts=opts.InitOpts(width=width, height=height)) \
        .add_xaxis(xaxis_data=xaxis_data).set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),datazoom_opts=opts.DataZoomOpts()
    )
    symbols = ['circle', 'rect', 'roundRect', 'triangle', 'diamond', 'pin', 'arrow']
    for i in range(len(y_cols)):
        s.add_yaxis(
            series_name=y_cols[i],
            y_axis=df[y_cols[i]].values.tolist(),
            symbol=symbols[y_symbols[i]],
            symbol_size=12,
            label_opts=opts.LabelOpts(is_show=False),
        )
    return s


def chart_scatter3d(df, x_col, y_col, z_col, color_col, width="100%", height="500px", title=''):
    """
    3d散点图
    :param x_col: x轴对应列
    :param y_col: y轴对应列
    :param z_col: z轴对应列
    :param color_col: 颜色对应列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    df[x_col] = __fixdatetime(df, x_col)
    df[y_col] = __fixdatetime(df, y_col)
    df[color_col] = (df[color_col] / (df[color_col].max()) * 100).astype(int)
    df_data = df[[x_col, y_col, z_col, color_col]]
    scatter = Scatter3D(init_opts=opts.InitOpts(width=width, height=height)) \
        .add(
        series_name=title,
        shading="lambert",
        data=df_data.values.tolist(),
        xaxis3d_opts=opts.Axis3DOpts(type_="category", data=sorted(list(df[x_col].unique()))),
        yaxis3d_opts=opts.Axis3DOpts(type_="category", data=sorted(list(df[y_col].unique()))),
        zaxis3d_opts=opts.Axis3DOpts(type_="value"),
    ).set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            max_=100,
            dimension=3,
            range_color=[
                "#313695",
                "#4575b4",
                "#74add1",
                "#abd9e9",
                "#e0f3f8",
                "#ffffbf",
                "#fee090",
                "#fdae61",
                "#f46d43",
                "#d73027",
                "#a50026"
            ]
        )
    )
    return scatter


def chart_bar3d(df, x_col, y_col, z_col, color_col, width="100%", height="500px", title=''):
    """
    3d柱状图
    :param x_col: x轴对应列
    :param y_col: y轴对应列
    :param z_col: z轴对应列
    :param color_col: 颜色对应列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    df[x_col] = __fixdatetime(df, x_col)
    df[y_col] = __fixdatetime(df, y_col)
    df[color_col] = (df[color_col] / (df[color_col].max()) * 100).astype(int)
    df_data = df[[x_col, y_col, z_col, color_col]]
    bar = Bar3D(init_opts=opts.InitOpts(width=width, height=height)) \
        .add(
        series_name=title,
        shading="lambert",
        data=df_data.values.tolist(),
        xaxis3d_opts=opts.Axis3DOpts(type_="category", data=sorted(list(df[x_col].unique()))),
        yaxis3d_opts=opts.Axis3DOpts(type_="category", data=sorted(list(df[y_col].unique()))),
        zaxis3d_opts=opts.Axis3DOpts(type_="value"),
    ).set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            max_=100,
            dimension=3,
            range_color=[
                "#313695",
                "#4575b4",
                "#74add1",
                "#abd9e9",
                "#e0f3f8",
                "#ffffbf",
                "#fee090",
                "#fdae61",
                "#f46d43",
                "#d73027",
                "#a50026"
            ]
        )
    )
    return bar

def chart_pie(df, category_col='', value_col='', width="100%", height="500px", title=''):
    """
    玫瑰图
    :param category_col: 类别列
    :param value_col: 值列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    data_pair = df[[category_col, value_col]].values.tolist()
    pie = Pie(init_opts=opts.InitOpts(width=width, height=height)) \
        .add(
        series_name=title,
        data_pair=data_pair,
        label_opts=opts.LabelOpts(is_show=True, position="center"),
    ).set_global_opts(
        title_opts=opts.TitleOpts(
            title=title,
            pos_left="center",
            pos_top="20",
            title_textstyle_opts=opts.TextStyleOpts(color="black"),
        ),
        legend_opts=opts.LegendOpts(is_show=True, type_='scroll'),
    ).set_series_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
        ),
        label_opts=opts.LabelOpts(color="black"),
    )
    return pie

def chart_rose(df, category_col='', value_col='', width="100%", height="500px", title=''):
    """
    玫瑰图
    :param category_col: 类别列
    :param value_col: 值列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    data_pair = df[[category_col, value_col]].values.tolist()
    pie = Pie(init_opts=opts.InitOpts(width=width, height=height)) \
        .add(
        series_name=title,
        data_pair=data_pair,
        rosetype="radius",
        radius="55%",
        center=["50%", "50%"],
        label_opts=opts.LabelOpts(is_show=True, position="center"),
    ).set_global_opts(
        title_opts=opts.TitleOpts(
            title=title,
            pos_left="center",
            pos_top="20",
            title_textstyle_opts=opts.TextStyleOpts(color="black"),
        ),
        legend_opts=opts.LegendOpts(is_show=True, type_='scroll'),
    ).set_series_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
        ),
        label_opts=opts.LabelOpts(color="black"),
    )
    return pie


def chart_parallel(df, id_col='', value_cols=[], high_light_id=None, width='100%', height='600px', title=""):
    """
    平行坐标图
    :param id_col: id列
    :param value_cols: 值列集合
    :param high_light_id: 高亮哪个id对应数据
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    p = Parallel(init_opts=opts.InitOpts(width=width, height=height))
    p.add_schema([opts.ParallelAxisOpts(dim=i, name=value_cols[i]) for i in range(0, len(value_cols))])
    p.set_global_opts(title_opts=opts.TitleOpts(title=title)
                      , legend_opts=opts.LegendOpts(type_='scroll', pos_top=20)
                      , tooltip_opts=opts.TooltipOpts(trigger='axis', trigger_on="mousemove|click",
                                                      axis_pointer_type='line')
                      )
    df_data = df[[id_col] + value_cols]
    if high_light_id != None:
        p.add(high_light_id, [df_data[df_data[id_col] == high_light_id][value_cols].iloc[0].values.tolist()],
              linestyle_opts=opts.LineStyleOpts(width=5, opacity=1, color='red'))
        for index, row in df_data.iterrows():
            if row[id_col] != high_light_id:
                p.add(str(row[id_col]), [row[value_cols].values.tolist()],
                      linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.5))
    else:
        for index, row in df_data.iterrows():
            p.add(str(row[id_col]), [row[value_cols].values.tolist()],
                  linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.5))
    return p


# 漏斗图
def chart_funnel(df, category_col='', value_col='', width='100%', height='600px', title=""):
    """
    漏斗图
    :param category_col: 类别列
    :param value_col: 值列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    data = df[[category_col, value_col]].sort_values(value_col, ascending=False).values.tolist()
    fnl = Funnel(init_opts=opts.InitOpts(width=width, height=height)) \
        .add(
        series_name="",
        data_pair=data,
        gap=2,
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b} : {c}%"),
        label_opts=opts.LabelOpts(is_show=True, position="inside"),
        itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=1),
    ).set_global_opts(title_opts=opts.TitleOpts(title=title))
    return fnl


def chart_calendar(df,date_col=None, value_col='value', width='100%', height='600px', title=""):
    """
    日历热度图
    :param date_col: 日期列
    :param value_col: 值列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    if date_col == None:
        date_col='date_col'
        df_data = df[[value_col]].reset_index()
        df_data.columns=[date_col,value_col]
        if str(df[date_col].dtype) == 'datetime64[ns]':
            df_data[date_col]=df_data[date_col].dt.strftime('%Y%m%d%H%M')
    else:
        df[date_col] = __fixdatetime(df, date_col)
        df_data = df[[date_col, value_col]]

    cal = Calendar(init_opts=opts.InitOpts(width=width, height=height)) \
        .add(
        series_name="",
        yaxis_data=df_data.values.tolist(),
        calendar_opts=opts.CalendarOpts(
            pos_top="120",
            pos_left="50",
            pos_right="30",
            range_=[str(df_data[date_col].min()), str(df_data[date_col].max())],
            daylabel_opts=opts.CalendarDayLabelOpts(name_map="cn", first_day=1),
            monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="cn"),
            yearlabel_opts=opts.CalendarYearLabelOpts(is_show=True),
        ),
    ).set_global_opts(
        title_opts=opts.TitleOpts(pos_top="20", pos_left="left", title=title),
        tooltip_opts=opts.TooltipOpts(formatter="{c}"),
        visualmap_opts=opts.VisualMapOpts(
            max_=int(df_data[value_col].max()), min_=int(df_data[value_col].min()), orient="horizontal",
            is_piecewise=False, pos_left='center', pos_top='top'
        ),
    )
    return cal


def chart_sunburst(df, category_cols=[], value_col="", width='100%', height='800px', title=""):
    """
    日出图
    :param category_cols: 类别列 eg[‘sw_l1','sw_l2']
    :param value_col: 值列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    data = __df2tree(df, category_cols, value_col)
    sunburst = Sunburst(init_opts=opts.InitOpts(width=width, height=height)) \
        .add(series_name=title, data_pair=data, radius=[0, "90%"]) \
        .set_global_opts(title_opts=opts.TitleOpts(title=title)) \
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}"))
    return sunburst


def chart_treemap(df, category_cols=[], value_col="", width='100%', height='600px', title=""):
    """
    树地图
    :param category_cols: 类别列 类别列 eg[‘sw_l1','sw_l2']
    :param value_col: 值列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    data = __df2tree(df, category_cols, value_col)
    t = TreeMap(init_opts=opts.InitOpts(width=width, height=height)) \
        .add(title, data) \
        .set_global_opts(title_opts=opts.TitleOpts(title=title))
    return t


# 箱型图
def chart_box(df, y_cols=[], width='100%', height='500px', title=""):
    """
    箱型图
    :param y_cols:要绘制箱型分布的列
    :param width: '100%'
    :param height:'500px'
    :param title: ""
    :return:
    """
    box_plot = Boxplot(init_opts=opts.InitOpts(width=width, height=height))
    box_plot.add_xaxis(xaxis_data=y_cols) \
        .set_global_opts(
        title_opts=opts.TitleOpts(
            pos_left="center", title=title
        ),
        tooltip_opts=opts.TooltipOpts(trigger="item", axis_pointer_type="shadow"),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=True,
            splitarea_opts=opts.SplitAreaOpts(is_show=True),
            axislabel_opts=opts.LabelOpts(formatter="{value}"),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="分位值",
            is_scale=True,
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        ),
    ).set_series_opts(tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}"))
    box_plot.add_yaxis(series_name=title, y_axis=box_plot.prepare_data(df[y_cols].T.values.tolist()))

    return box_plot


def chart_heatmap(df, x_col='', y_col="", value_col="", width='100%', height='800px', title=""):
    """
    热力图
    :param x_col: X列
    :param y_col: y列
    :param value_col: 值列
    :param width: '100%'
    :param height:'600px'
    :param title: ""
    :return:
    """
    x_index = sorted(list(__fixdatetime(df, x_col).unique()))
    y_index = sorted(list(df[y_col].unique()))
    df_data = df[[x_col, y_col, value_col]].copy()
    df_data[x_col] = __fixdatetime(df, x_col)
    df_data[y_col] = __fixdatetime(df, y_col)
    data = df_data[[x_col, y_col, value_col]].values.tolist()
    h = HeatMap(init_opts=opts.InitOpts(width=width, height=height)) \
        .add_xaxis(x_index) \
        .add_yaxis(title, y_index, data) \
        .set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        visualmap_opts=opts.VisualMapOpts()
    )
    return h


def __fixdatetime(df, col):
    if str(df[col].dtype) == 'datetime64[ns]':
        return df[col].dt.strftime('%Y%m%d%H%M')
    else:
        return df[col]


# dataframe 转换成pyecharts需要的树形结构
def __df2tree(df, category_cols=[], value_col=""):
    cols = category_cols
    df_data = df[category_cols + [value_col]]
    tmp_dict = {}
    for i in range(len(cols) - 1, 0, -1):
        df_tmp = df_data[[cols[i - 1], cols[i], value_col]]
        key_map = df_tmp.groupby(cols[i - 1]).apply(lambda dx: list(dx[cols[i]].unique())).to_dict()
        cat0_value_dict = df_tmp[[cols[i - 1], value_col]].groupby(cols[i - 1]).sum()[value_col].to_dict()
        if i == len(cols) - 1:
            cat1_value_dict = df_tmp[[cols[i], value_col]].groupby(cols[i]).sum()[value_col].to_dict()
            for cat0 in key_map.keys():
                children = []
                for cat1 in key_map[cat0]:
                    value = cat1_value_dict[cat1]
                    children.append({'name': cat1, 'value': value})
                tmp_dict[cat0] = {'name': cat0, 'value': cat0_value_dict[cat0], 'children': children}
        else:
            for cat0 in key_map.keys():
                children = []
                for cat1 in key_map[cat0]:
                    children.append(tmp_dict[cat1])
                tmp_dict[cat0] = {'name': cat0, 'value': cat0_value_dict[cat0], 'children': children}
    data = []
    for cat in df_data[cols[0]].unique():
        data.append(tmp_dict[cat])
    return data


def chart_timeline(charts_dict={}):
    """
    timeline组件包裹多图表
    :param charts_dict: {name:chart}
    :return:
    """
    tl = Timeline()
    for name in charts_dict.keys():
        tl.add(charts_dict[name], name)
    return tl


def chart_tab(charts_dict={}):
    """
    tab组件多图表
    :param charts_dict:{name:chart}
    :return:
    """
    tab = Tab()
    for name in charts_dict.keys():
        tab.add(charts_dict[name], name)
    return tab


def chart_page(chart_list):
    """
    顺序多图
    :param charts:
    :return:
    """
    page = Page(layout=Page.SimplePageLayout)
    for chart in chart_list:
        page.add(
            chart
        )
    return page

__all__ = ['line_reg','count9']
def _collect_func():
    funcs = []
    for func in globals().keys():
        if func.startswith("chart_"):
            funcs.append(func)
    return funcs

__all__.extend(_collect_func())

del _collect_func

