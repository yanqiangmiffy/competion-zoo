# !/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:yanqiang
@File: 03_stacking_demo.py
@Time: 2018/11/27 10:59
@Software: PyCharm
@Description:
"""
from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from xgboost import XGBClassifier as xgb
from lightgbm import LGBMClassifier as lgb
import numpy as np

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y)

# 第一个例子
nn = KNeighborsClassifier(n_neighbors=1)
rfc = RandomForestClassifier(random_state=1)
nb = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[nn, rfc, nb],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([nn, rfc, nb, sclf],
                      ['KNN',
                       'Random Forest',
                       'Naive Bayes',
                       'StackingClassifier']):
    scores = cross_val_score(clf, X, y,
                             cv=3, scoring='accuracy')
    # clf.fit(X,y)
    # print(clf.predict(X))
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

# 第二个例子：
stack_model = [nn, rfc, nb, xgb, lgb]  # 这个地方我是简写，实际这样训练会报错，需要自己导库，定义分类器
# train_data 表示训练集，train_label 表示训练集的标签，test_data表示训练集
ntrain = x_train.shape[0]  # 训练集样本数量
print(ntrain)
ntest = x_test.shape[0]  # 测试集样本数量
train_stack = np.zeros((ntrain, 5))  # n表示n个模型
test_stack = np.zeros((ntest, 5))
kfold = KFold(n_splits=5)
kf = kfold.split(x_train, y_train)

for i, model in enumerate(stack_model):
    for j, (train_fold, validate) in enumerate(kf):
        X_train, X_validate, label_train, label_validate = \
            x_train[train_fold, :], x_train[validate,
                                            :], y_train[train_fold], y_train[validate]

        model.fit(X_train, label_train)
        train_stack[validate, i] = model.predict(X_validate)
        test_stack[:, i] = model.predict(x_test)

# 假设就只有两层，那么最后预测：

final_model = xgb()  # 假设第二层我们还是用xgb吧,这个地方也是简写，仅仅表示这个地方是xgb模型
final_model.fit(train_stack, y_train)

pre = final_model.predict(test_stack)
print(pre)


# 更多资料

# https://github.com/akshaykumarvikram/kaggle-advanced-regression-algos/blob/master/notebook.ipynb
# https://blog.csdn.net/willduan1/article/details/73618677
# https://blog.csdn.net/qq1483661204/article/details/80157365
# https://zhuanlan.zhihu.com/p/26890738
# https://github.com/wanlida/2018_diantou_PhotovoltaicPowerStation/blob/master/history_code_please_ignore/wan_8%E6%9C%884%E5%8F%B722%E7%82%B911%E5%88%86.ipynb


class Stacker(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    # X: 原始训练集, y: 原始训练集真实值, predict_data: 原始待预测数据
    def fit_predict(self, X, y, predict_data):
        X = np.array(X)
        y = np.array(y)
        T = np.array(predict_data)

        folds = list(KFold(n_splits=self.n_splits, shuffle=False,
                           random_state=2018).split(X, y))

        # 以基学习器预测结果为特征的 stacker的训练数据 与 stacker预测数据
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_predict = np.zeros((T.shape[0], len(self.base_models)))

        for i, regr in enumerate(self.base_models):
            print(i + 1, 'Base model:', str(regr).split('(')[0])
            S_predict_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                # 将X分为训练集与测试集
                X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                print('Fit fold', (j + 1), '...')
                regr.fit(X_train, y_train)
                y_pred = regr.predict(X_test)
                S_train[test_idx, i] = y_pred
                S_predict_i[:, j] = regr.predict(T)

            S_predict[:, i] = S_predict_i.mean(axis=1)

        nmse_score = cross_val_score(
            self.stacker, S_train, y, cv=5, scoring='neg_mean_squared_error')
        print('CV MSE:', -nmse_score)
        print('Stacker AVG MSE:', -nmse_score.mean(), 'Stacker AVG Score:',
              np.mean(np.divide(1, 1 + np.sqrt(-nmse_score))))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_predict)
        return res, S_train, S_predict


# regrs = [
#     xgbt1, gbdt1, forest1, lgb1,
#     xgbt2, gbdt2, forest2, lgb2,
#     xgbt3, gbdt3, forest3, lgb3
# ]
#
#
#
#
# # stacking_mode1 = Ridge(alpha=0.008, copy_X=True, fit_intercept=False, solver='auto', random_state=2)
# stacking_model = SVR(C=100, gamma=0.01, epsilon=0.01)
# stacker = Stacker(5, stacking_model, regrs)
# pred_stack, S_train_data, S_predict_data = stacker.fit_predict(all_X_train, all_y_train, sub_data)


# https://tianchi.aliyun.com/notebook-ai/detail?postId=41822

# 将lgb和xgb的结果进行stacking
train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10

mean_squared_error(target.values, oof_stack)
