# !/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:yanqiang
@File: weight_demo.py
@Time: 2018/11/27 11:12
@Software: PyCharm
@Description: 交叉验证
"""
param = {'num_leaves': 120,
         'min_data_in_leaf': 30,
         'objective': 'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[
                    trn_data, val_data], verbose_eval=200, early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(
        X_train[val_idx], num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test,
                                   num_iteration=clf.best_iteration) / folds.n_splits
