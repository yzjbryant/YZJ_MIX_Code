##Question: 靠海对气候的影响
import numpy as np
import pandas as pd
import datetime

ferrara=pd.read_json('http://api.openweathermap.org/data/2.5/history/city?q=Ferrara,IT')
#faenza 看dataframe结构

#将树状结构转换为列表式结构
#用于pandas分析

def prepare(city_list,city_name):
    temp=[ ]
    humidity=[ ]
    pressure=[ ]
    description=[ ]
    dt=[ ]
    wind_speed=[ ]
    wind_deg=[ ]
    for row in city_list:
        temp.append(row['main']['temp']-273.15)
        pass
    headings=['temp']
    data=[temp]
    df=pd.DataFrame(data,index=headings)
    city=df.T
    return city
df_ferrara=prepare(ferrara.list,'Ferrara')
df_ferrara.to_csv('sds.csv')
df_ferrara.read_csv('dsd.csv')

