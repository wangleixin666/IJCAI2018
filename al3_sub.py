#!/usr/bin/env python    # -*- coding: utf-8 -*

import time
import pandas as pd
from kaggle.XGBoost import XGBClassifier
import warnings

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

    train = data.copy()
    test = pd.read_csv('D:\kaggle\\alimm\\round1_ijcai_18_test_a_20180301.txt', sep=' ')
    test = convert_data(test)

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

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        min_child_weight=2,
        subsample=0.7,
        colsample_bytree=0.7
    )
    clf.fit(X_train, Y_train)
    Y_predict = clf.predict_proba(X_test, )[:, 1]

    pd.DataFrame({'instance_id': test['instance_id'], 'predicted_score': Y_predict}). \
        to_csv('D:\kaggle\\alimm\\baseline_05.csv', index=False, sep=' ')

# 第一赛季 1094 / 0.08288
# 第一次进步，容易吗我
# 线下对数损失0.08161589381677363
# 结果与最优的lightbgm相比线上线下都差不多
# 接下来就是模型融合，可以试下xgboost和lr或者lightb融合一波