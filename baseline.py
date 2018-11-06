# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: baseline.py
@Time: 2018/11/6 10:48
@Software: PyCharm 
@Description:
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


class Baseline(object):
    def __init__(self):
        pass


def get_models():
    """
    生成机器学习库
    :return:
    """
    nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    ln_svc = LinearSVC()
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    nn = MLPClassifier()
    ab = AdaBoostClassifier()
    gb = GradientBoostingClassifier()
    rf = RandomForestClassifier()
    xgb = XGBClassifier()
    lgb = LGBMClassifier()
    models = {
        'naive bayes': nb,
        # 'svm': svc,
        # 'linear_svm': ln_svc,
        'knn': knn,
        'logistic': lr,
        'mlp-nn': nn,
        'ada boost':ab,
        'random forest': rf,
        'gradient boost': gb,
        'xgb':xgb,
        'lgb':lgb
    }
    return models


def score_models(models, X,y):
    """Score model in prediction DF"""
    print("评价每个模型.")
    for name,model in models.items():
        score = cross_val_score(model,X,y,scoring='roc_auc',cv=5)
        mean_score=np.mean(score)
        print("{}: {}" .format(name, mean_score))
    print("Done.\n")


train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# 数据预处理 类别编码
cate_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
for col in cate_cols:
    lb_encoder = LabelEncoder()
    train[col] = lb_encoder.fit_transform(train.job)
    test[col] = lb_encoder.transform(test.job)  # 这个步骤风险有点大，因为test的类别标签不一定都出现在train里面，这里比较幸运

train = pd.get_dummies(train, columns=cate_cols)
test = pd.get_dummies(test, columns=cate_cols)
# 数据预处理 数值型数据
# num_cols=['age','balance','duration','campaign','pdays','previous']
# scaler=MinMaxScaler()
# train[num_cols] = scaler.fit_transform(train[num_cols].values)
# test[num_cols] = scaler.transform(test[num_cols].values)
print(train.shape)
print(test.shape)
cols = [col for col in train.columns if col not in ['id', 'y']]
X = train[cols]
y = train['y']

models = get_models()
score_models(models, X, y)


