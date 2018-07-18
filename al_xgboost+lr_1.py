#!/usr/bin/env python    # -*- coding: utf-8 -*

import time
import pandas as pd
from kaggle.XGBoost import XGBClassifier
from sklearn.metrics import log_loss
import warnings
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
# import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse.construct import hstack

"""对Xgboost进行调参"""

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

    train = data.loc[data.day < 24]  # 18,19,20,21,22,23 # 一共420693行，32列
    test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集 # 一共420693

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
    """
    xgbc = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        min_child_weight=2,
        subsample=0.7,
        colsample_bytree=0.7
    )
    # clf = lgb.LGBMClassifier()
    xgbc.fit(X_train, Y_train)
    
    Y_predict = xgboost.predict_proba(X_test, )[:, 1]
    print('test:', log_loss(Y_test, Y_predict))
    print "Accuracy : %.4f" % metrics.roc_auc_score(Y_test, Y_predict)
    """

    # 定义GBDT模型
    gbdt = GradientBoostingClassifier()
    # 训练学习
    gbdt.fit(X_train, Y_train)

    """
    # 预测及AUC评测
    Y_predict_gbdt = gbdt.predict_proba(X_test)[:, 1]
    print('test:', log_loss(Y_test, Y_predict_gbdt))
    print "Accuracy : %.4f" % metrics.roc_auc_score(Y_test, Y_predict_gbdt)
    # 原始结果为：('test:', 0.081939773937662927)
    # Accuracy : 0.6848
    """

    """
    # lr对原始特征样本模型训练
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)  # 预测及AUC评测
    Y_predict_LR = lr.predict_proba(X_test)[:, 1]
    print('test:', log_loss(Y_test, Y_predict_LR))
    print "before Accuracy : %.4f" % metrics.roc_auc_score(Y_test, Y_predict_LR)
    # ('test:', 0.095052240862119497)
    # before Accuracy : 0.5413
    """

    # GBDT编码原有特征
    X_train_leaves = gbdt.apply(X_train)[:, :, 0]
    X_test_leaves = gbdt.apply(X_test)[:, :, 0]
    # apply方法只有gdbt里面才有
    # xgboost里没有。。

    # 对所有特征进行ont-hot编码
    (train_rows, cols) = X_train_leaves.shape

    gbdtenc = OneHotEncoder()
    X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))
    # print X_trans.shape # (478111, 797)

    """
        # 定义LR模型
        lr = LogisticRegression(n_jobs=1)
        # lr对gbdt特征编码后的样本模型训练
        lr.fit(X_trans[:train_rows, :], Y_train)
        # 预测及AUC评测
        Y_predict_gbdtlr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
        print('test:', log_loss(Y_test, Y_predict_gbdtlr1))
        print "before Accuracy : %.4f" % metrics.roc_auc_score(Y_test, Y_predict_gbdtlr1)
    
        # ('test:', 0.082230433929968566)
        # before Accuracy: 0.6818
        """
    # 定义LR模型
    lr = LogisticRegression(n_jobs=1)

    # 组合特征
    X_train_ext = hstack([X_trans[:train_rows, :], X_train])
    X_test_ext = hstack([X_trans[train_rows:, :], X_test])

    # lr对组合特征的样本模型训练
    lr.fit(X_train_ext, Y_train)
    Y_predict_gbdtlr2 = lr.predict_proba(X_test_ext)[:, 1]
    print('test:', log_loss(Y_test, Y_predict_gbdtlr2))
    print "before Accuracy : %.4f" % metrics.roc_auc_score(Y_test, Y_predict_gbdtlr2)
    # ('test:', 0.095052240862119497)
    # before Accuracy : 0.5413

    """
    
    # 合并编码后的训练数据和测试数据
    All_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    All_leaves = All_leaves.astype(np.int32)

    # 对所有特征进行ont-hot编码
    xgbenc = OneHotEncoder()
    X_trans = xgbenc.fit_transform(All_leaves)

    (train_rows, cols) = X_train_leaves.shape

    # 定义LR模型
    lr = LogisticRegression()
    # lr对xgboost特征编码后的样本模型训练
    lr.fit(X_trans[:train_rows, :], y_train)
    # 预测及AUC评测
    y_pred_xgblr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
    xgb_lr_auc1 = roc_auc_score(y_test, y_pred_xgblr1)
    print('基于Xgb特征编码后的LR AUC: %.5f' % xgb_lr_auc1)

    # 定义LR模型
    lr = LogisticRegression(n_jobs=-1)
    # 组合特征
    X_train_ext = hstack([X_trans[:train_rows, :], X_train])
    X_test_ext = hstack([X_trans[train_rows:, :], X_test])

    # lr对组合特征的样本模型训练
    lr.fit(X_train_ext, y_train)

    # 预测及AUC评测
    y_pred_xgblr2 = lr.predict_proba(X_test_ext)[:, 1]
    xgb_lr_auc2 = roc_auc_score(y_test, y_pred_xgblr2)
    print('基于组合特征的LR AUC: %.5f' % xgb_lr_auc2)

    """