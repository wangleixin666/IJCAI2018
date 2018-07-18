#!/usr/bin/env python    # -*- coding: utf-8 -*

import time
import pandas as pd
from kaggle.XGBoost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import log_loss
import warnings
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold

# 对Xgboost和lightbgm进行融合

"""
采用最优的xgboost结果为
线下0.08161
准确率0.6874
线上结果为0.08288
"""

warnings.filterwarnings("ignore")


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])
    return data


if __name__ == "__main__":

    data = pd.read_csv('al_train.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)

    train = data.loc[data.day < 24]
    test = data.loc[data.day == 24]

    features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                ]
    target = ['is_trade']

    X_train = train[features]
    X_test = test[features]
    Y_train = train[target]
    Y_test = test[target]
    """X_train,Y_train已经是部分特征的值了"""
    # print X_train.shape # (420693, 22)
    # print Y_train.shape # (420693, 1)
    # print X_test.shape  # (57418, 22)

    """模型融合中要用到的模型"""
    clfs = [XGBClassifier(n_estimators=400, max_depth=3, min_child_weight=2, subsample=0.7, colsample_bytree=0.7),
            lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)]

    dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))
    # dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
    #  dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))
    # print dataset_blend_train.shape # (420693L, 2L)
    # print dataset_blend_test.shape # (57418L, 2L)

    """5折融合"""
    kf = KFold(Y_train, 5)
    kf.split(Y_train, 5)
    for j, clf in enumerate(clfs):
        '''依次训练各个单模型'''
        print(j, clf)
        dataset_blend_test_j = np.zeros((X_test.shape[0], len(kf)))
        for i, (train, test) in enumerate(kf):
            '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
            # print("Fold", i)
            x_train, y_train, x_test, y_test = X_train[train], Y_train[train], X_train[test], Y_train[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(x_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:, 1]
        '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
        print('test:', log_loss(Y_test, dataset_blend_test[:, j]))
        print "Accuracy : %.4f" % metrics.roc_auc_score(Y_test, dataset_blend_test[:, j])

"""
stacking 融合就类似交叉验证。
将数据集分为K个部分，共有n个模型。
for i in xrange(n):
​ for i in xrange(k):
​用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。
​对于测试集，直接用这k个模型的预测值均值作为新的特征。
这样k次下来，整个数据集都获得了这个模型构建的New Feature，n个模型训练下来，这个模型就有n个New Features
把New Features和label作为新的分类器的输入进行训练。然后输入测试集的New Features输入模型获得最终的预测结果
"""

"""
融合的条件
Base Model 之间的相关性要尽可能的小
这就是为什么非 Tree-based Model 往往表现不是最好但还是要将它们包括在 Ensemble 里面的原因
Ensemble 的 Diversity 越大，最终 Model 的 Bias 就越低
Base Model 之间的性能表现不能差距太大
这其实是一个 Trade-off，在实际中很有可能表现相近的 Model 只有寥寥几个而且它们之间相关性还不低
但是实践告诉我们即使在这种情况下 Ensemble 还是能大幅提高成绩
"""

# skf = StratifiedKFold(n_splits=5).get_n_splits(Y_train)
# print type(StratifiedKFold(n_splits=5).get_n_splits(Y_train)) # <type 'int'>
# skf = list(StratifiedKFold(n_splits=5).get_n_splits(Y_train))
# print type(skf)
# 查了下，原来不能直接用int进行迭代，而必须加个range.
# skf = StratifiedKFold(n_splits=5)
# print skf.split(Y_train, 5)
# TypeError: Singleton array array(5) cannot be considered a valid collection.
# skf = KFold(n_splits=5)
# print skf.split(Y_train)
