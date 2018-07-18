#!/usr/bin/env python    # -*- coding: utf-8 -*

import time
import pandas as pd
from kaggle.XGBoost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import log_loss
import warnings
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

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


    def SelectModel(modelname):
        model = None
        if modelname == "XGBC":
            model = XGBClassifier(n_estimators=400, max_depth=3, min_child_weight=2, subsample=0.7, colsample_bytree=0.7)
        elif modelname == "LGB":
            model = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        else:
            pass
        return model


    def get_oof(clf, n_folds, x_train, y_train, x_test):
        ntrain = x_train.shape[0]
        ntest = x_test.shape[0]
        classnum = len(np.unique(y_train))
        kf = KFold(n_splits=n_folds)
        oof_train = np.zeros((ntrain, classnum))
        oof_test = np.zeros((ntest, classnum))

        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            kf_x_train = x_train[train_index]  # 数据
            kf_y_train = y_train[train_index]  # 标签

            kf_x_test = x_train[test_index]  # k-fold的验证集

            clf.fit(kf_x_train, kf_y_train)
            oof_train[test_index] = clf.predict_proba(kf_x_test)

            oof_test += clf.predict_proba(X_test)
        oof_test = oof_test / float(n_folds)
        return oof_train, oof_test


    # 使用stacking方法的时候
    # 第一级，重构特征当做第二级的训练集
    modelist = ['XGBC', 'LGB']
    newfeature_list = []
    newtestdata_list = []
    for modelname in modelist:
        clf_first = SelectModel(modelname)
        oof_train_, oof_test_ = get_oof(clf=clf_first, n_folds=5, x_train=X_train, y_train=Y_train, x_test=X_test)
        newfeature_list.append(oof_train_)
        newtestdata_list.append(oof_test_)

    # 特征组合
    newfeature = reduce(lambda x, y: np.concatenate((x, y), axis=1), newfeature_list)
    newtestdata = reduce(lambda x, y: np.concatenate((x, y), axis=1), newtestdata_list)

    # 第二级，使用上一级输出的当做训练集
    clf_second1 = RandomForestClassifier()
    clf_second1.fit(newfeature, Y_train)
    pred = clf_second1.predict(newtestdata)
    accuracy = log_loss(Y_test, pred)
    print accuracy

# KeyError: '[ 42070  42071  42072 ..., 420690 420691 420692] not in index'
# 怎么都报错。。。
"""
#这里kf.split(X)返回的是X中进行分裂后train和test的索引值
令X中数据集的索引为0，1，2，3；
第一次分裂，先选择test，索引为0和1的数据集为test,剩下索引为2和3的数据集为train；
第二次分裂，先选择test，索引为2和3的数据集为test,剩下索引为0和1的数据集为train。
"""