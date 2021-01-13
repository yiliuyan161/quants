
import pandas as pd
import datetime as dt
import numpy as np
import IPython.display as dspl
import ipywidgets as widgets
from IPython.core.display import display
import warnings
from .chan import Chan
from .chart import chart_kline_overlap_tds,chart_kline_overlap_signal,chart_kline_overlap_zone,chart_kline_overlap_trend,chart_kline,chart_tab




"""
面板设计，
容器布局，output 绑定，注册output函数
"""

class Panel:
    @staticmethod
    def init(full_width=True):
        warnings.filterwarnings('ignore')
        style = """
                <style>
                   .jupyter-widgets-output-area .output_scroll {
                        height: unset !important;
                        border-radius: unset !important;
                        -webkit-box-shadow: unset !important;
                        box-shadow: unset !important;
                    }
                    .jupyter-widgets-output-area  {
                        height: auto !important;
                    }
                    """ \
                    + (".container{width:95%!important}" if full_width else "") + \
                    """
                </style>
                """
        display(widgets.HTML(style))
        
    @staticmethod
    def help():
        print( """
        import IPython.display as dspl
        import ipywidgets as widgets
        out_filter = widgets.Output()
        opt_display = widgets.Output()
        text1 = widgets.Text(value='',placeholder=name,description=name,disabled=False)
        @out_filter.capture()
        def on_value_change(change):
            dspl.clear_output()
            keywords = change['new']
        text1.observe(on_value_change, names='value')
        @opt_display.capture()
        def on_button_clicked(b):
            dspl.clear_output()
        btn1.on_click(on_button_clicked)
        """)
    @staticmethod
    def toggle_input():
        code="""
            <script>
            if (!document.getElementById("toggle_button")){
                var div = document.createElement("div");
                div.className = 'btn-group';
                var btn = document.createElement("button");
                btn.className='btn btn-default'
                btn.setAttribute("id", "toggle_button");
                var i=document.createElement("i");
                i.className='fa-eye fa'
                btn.appendChild(i)
                div.appendChild(btn)
                document.getElementById("maintoolbar-container").append(div)
            }
            function toggleInput(){
                nodes=document.querySelectorAll("div.input")
                nodes.forEach(function(ele,index,arr){
                    var display = ele.style.display;
                    if(display=='none'){
                        ele.style.display = '';
                    }else{
                        ele.style.display = 'none';
                    }
                })

            }
            document.getElementById("toggle_button").onclick = toggleInput
            </script>
         """
        display(dspl.HTML(code))

    # 简单DataFrame查询面板
    @staticmethod
    def df_query_form(title, df, query_cols, target_cols, func_callback):
        """
        查询面板
        title:名称
        df:要查询的DataFrame
        query_cols:查询的列
        target_cols:目标列
        func_callback:回调函数
        """
        text1 = Panel.text(title)
        btn1 = Panel.button('选中第一条')
        out_filter = widgets.Output()
        vbox = widgets.VBox([widgets.HBox([text1, btn1]), out_filter])
        df_filter = [df]

        @out_filter.capture()
        def on_value_change(change):
            dspl.clear_output()
            keywords = change['new']
            dfs = []
            for col in query_cols:
                df_left = df[df[col].str.lower().str.contains(keywords)]
                if len(df_left) > 0:
                    dfs.append(df_left)
            if len(dfs)>0:
                df_filter[0] = pd.concat(dfs).drop_duplicates()
                display(df_filter[0].head(10))

        text1.observe(on_value_change, names='value')
        def on_button_clicked(b):
            dspl.clear_output()
            params = df_filter[0].iloc[0][target_cols].to_dict()
            func_callback(params)
        btn1.on_click(on_button_clicked)
        return vbox
    
        
    @staticmethod
    def tab_outputs(titles=[]):
        """
        tab output输出
        """
        outputs={titles[i]:widgets.Output() for i in range(0,len(titles))}
        tab = widgets.Tab()
        tab.children = list(outputs.values())
        for i in range(0,len(titles)):
            tab.set_title(i, titles[i])    
        return tab,outputs
        

    @staticmethod   
    def button(name):
        return widgets.Button(
            description=name,
            disabled=False,
            tooltip=name)
    @staticmethod  
    def intSlider(name, series):
        max = series.max()
        min = series.min()
        return widgets.IntRangeSlider(
            value=[min, max],
            min=min,
            max=max,
            step=1,
            description=name,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )
    @staticmethod  
    def dropdown(name, series):
        return widgets.Dropdown(
            options=series.values.tolist(),
            value=series.values[0],
            description=name,
            disabled=False,
        )
    @staticmethod  
    def text(name):
        return widgets.Text(
            value='',
            placeholder=name,
            description=name,
            disabled=False
        )


__all__=['Panel']