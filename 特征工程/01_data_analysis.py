# !/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:yanqiang
@File: weight_demo.py
@Time: 2018/11/27 11:12
@Software: PyCharm
@Description: 统计每列缺失值比例   统计每列类别值所占该列的比例
"""
import pandas as pd
import numpy as np


def statics(data):
    stats = []
    for col in data.columns:
        stats.append((col, data[col].nunique(), data[col].isnull().sum() * 100 / data.shape[0],
                      data[col].value_counts(normalize=True, dropna=False).values[0] * 100, data[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage_of_missing_values',
                                            'Percentage_of_values_in_the_biggest category', 'type'])
    stats_df.sort_values('Percentage_of_missing_values', ascending=False, inplace=True)
    return stats_df


def missing_data(data):
    """
    data:dataframe，展示每列缺失比列
    """
    total=data.isnull().sum()
    percent=(data.isnull().sum()/data.isnull().count()*100)
    tt=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
    types=[]
    for col in data.columns:
        dtype=str(data[col].dtype)
        types.append(dtype)
    tt['Types']=types
    return(np.transpose(tt))
