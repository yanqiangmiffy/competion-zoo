#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: quincy qiang 
@license: Apache Licence 
@file: gen_feas.py 
@time: 2019/12/10
@software: PyCharm 
"""

from pathlib import Path
import pandas as pd
import numpy as np
import requests
import pandas_profiling
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations

pd.set_option('display.float_format', lambda x: '%.3f' % x)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# simple_train_R04_jet.csv
# simple_test_R04_jet.csv
# sample_submmission.csv
df_train = pd.read_csv('data/simple_train_R04_jet.csv')
df_test = pd.read_csv('data/simple_test_R04_jet.csv')

df = pd.concat([df_train, df_test], sort=True)
print(len(df))

# 添加rank
base_fea = ['number_of_particles_in_this_jet',
            'jet_px',
            'jet_py',
            'jet_pz',
            'jet_energy',
            'jet_mass']
for col in base_fea:
    df['{}_rank'.format(col)] = df[col].rank()

# mean coding
for col in tqdm(['event_id']):
    for fea in base_fea:
        grouped_df = df.groupby(col).agg({fea: ['mean']})
        grouped_df.columns = [col + '_' + '_'.join(gp_col).strip() for gp_col in grouped_df.columns.values]
        grouped_df = grouped_df.reset_index()
        df = pd.merge(df, grouped_df, on=col, how='left')

for col in tqdm(['number_of_particles_in_this_jet']):
    for fea in base_fea:
        if fea != 'number_of_particles_in_this_jet':
            grouped_df = df.groupby(col).agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
            grouped_df.columns = [col + '_' + '_'.join(gp_col).strip() for gp_col in grouped_df.columns.values]
            grouped_df = grouped_df.reset_index()
            df = pd.merge(df, grouped_df, on=col, how='left')

# 添加count特征
for col in base_fea + ['event_id']:
    print(col)
    df['{}_count'.format(col)] = df.groupby(col)['jet_id'].transform('count')

# 角度特征
# def cal_cos(row, base=None):
#     v = np.array(row[['jet_px', 'jet_py', 'jet_pz']].values.tolist())
#     lx = np.sqrt(base.dot(base))
#     lv = np.sqrt(v.dot(v))
#     cos_angle = base.dot(v) / (lx * lv)
#     angle = np.arccos(cos_angle)
# return cos_angle
# tmp=[]
# for _,row in tqdm(df.iterrows()):
#     tmp.append(cal_cos(row, np.array([1, 0, 0])))
# df['x_cos']=tmp
#
# tmp=[]
# for _,row in tqdm(df.iterrows()):
#     tmp.append(cal_cos(row, np.array([0, 1, 0])))
# df['y_cos']=tmp
#
# tmp=[]
# for _,row in tqdm(df.iterrows()):
#     tmp.append(cal_cos(row, np.array([0, 0, 1])))
# df['z_cos']=tmp
#
# df['x_angle']=np.arccos(df['x_cos'])
# df['y_angle']=np.arccos(df['y_cos'])
# df['z_angle']=np.arccos(df['z_cos'])

# df['x_angle'] = df.apply(lambda row: cal_angle(row, np.array([1, 0, 0])), axis=1)
# df['y_angle'] = df.apply(lambda row: cal_angle(row, np.array([0, 1, 0])), axis=1)
# df['z_angle'] = df.apply(lambda row: cal_angle(row, np.array([0, 0, 1])), axis=1)
#

# # 比例特征
# df[str('jet_px') + '/' + str('jet_py')] = df['jet_px'].astype(float) / (df['jet_py'].astype(float) + 1e-10)
# df[str('jet_px') + '/' + str('jet_pz')] = df['jet_px'].astype(float) / (df['jet_pz'].astype(float) + 1e-10)
# df[str('jet_py') + '/' + str('jet_pz')] = df['jet_py'].astype(float) / (df['jet_pz'].astype(float) + 1e-10)
#
# df[str('jet_energy') + '/' + str('jet_px')] = df['jet_energy'].astype(float) / (df['jet_px'].astype(float) + 1e-10)
# df[str('jet_energy') + '/' + str('jet_pz')] = df['jet_energy'].astype(float) / (df['jet_py'].astype(float) + 1e-10)
# df[str('jet_energy') + '/' + str('jet_pz')] = df['jet_energy'].astype(float) / (df['jet_pz'].astype(float) + 1e-10)
#
# df[str('jet_mass') + '/' + str('jet_px')] = df['jet_mass'].astype(float) / (df['jet_px'].astype(float) + 1e-10)
# df[str('jet_mass') + '/' + str('jet_pz')] = df['jet_mass'].astype(float) / (df['jet_py'].astype(float) + 1e-10)
# df[str('jet_mass') + '/' + str('jet_pz')] = df['jet_mass'].astype(float) / (df['jet_pz'].astype(float) + 1e-10)
#
# df[str('jet_energy') + '/' + str('number_of_particles_in_this_jet')] = df['jet_energy'].astype(float) / (
#         df['number_of_particles_in_this_jet'].astype(float) + 1e-10)
# df[str('jet_mass') + '/' + str('number_of_particles_in_this_jet')] = df['jet_mass'].astype(float) / (
#         df['number_of_particles_in_this_jet'].astype(float) + 1e-10)
#
# df[str('jet_mass') + '/' + str('jet_energy')] = df['jet_mass'].astype(float) / (df['jet_energy'].astype(float) + 1e-10)


print("加减乘除...")
for a, b in tqdm(combinations(base_fea, 2)):
    df[str(a) + '_' + str(b)] = df[a].astype(float) + df[b].astype(float)
    df[str(a) + '/' + str(b)] = df[a].astype(float) / (df[b].astype(float) + 1e-10)
    df[str(a) + '*' + str(b)] = df[a].astype(float) * df[b].astype(float)
    df[str(a) + '/log' + str(b)] = df[a].astype(float) / np.log1p(df[b].astype(float))


def get_cvr_fea(data, cat_list=None):
    print("cat_list", cat_list)
    # 类别特征五折转化率特征
    print("转化率特征....")
    data['ID'] = data.index
    data['fold'] = data['ID'] % 5
    data.loc[data.label.isnull(), 'fold'] = 5
    label_feat = []
    for i in tqdm(cat_list):
        label_feat.extend([i + '_mean_last_1'])
        data[i + '_mean_last_1'] = None
        for fold in range(6):
            data.loc[data['fold'] == fold, i + '_mean_last_1'] = data[data['fold'] == fold][i].map(
                data[(data['fold'] != fold) & (data['fold'] != 5)].groupby(i)['label'].mean()
            )
        data[i + '_mean_last_1'] = data[i + '_mean_last_1'].astype(float)

    return data


# 连续特征离散化
cate_cols = []
for col in base_fea:
    df['{}_label'.format(col)] = pd.qcut(df[col], 6, labels=[1, 2, 3, 4, 5, 6])
    cate_cols.append('{}_label'.format(col))

df = get_cvr_fea(df, cate_cols)
no_features = ['jet_id', 'label'] + ['event_id', 'ID', 'fold']
features = [fea for fea in df.columns if fea not in no_features]

print(len(features), features)
train, test = df[:len(df_train)], df[len(df_train):]


def load_data():
    return train, test, no_features, features
