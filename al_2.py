#!/usr/bin/env python    # -*- coding: utf-8 -*
# 所有用的参数搭配已经是最优了
# 接下来考虑特征工程，利用PCA，或者别的方法对特征进行降维

import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
import warnings
from sklearn import metrics

# 进行特征选取，不能全部应用特征了
from sklearn.feature_selection import SelectFromModel
# 对于树常用的

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
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    return data


if __name__ == "__main__":
    data = pd.read_csv('D:\kaggle\\alimm\\round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)

    """
    data['user_age_level'].replace(-1, 1003, inplace=True)
    data['user_gender_id'].replace(-1, 0, inplace=True)
    data['user_occupation_id'].replace(-1, 2005, inplace=True)
    data['user_star_level'].replace(-1, 3006, inplace=True)
    data['item_sales_level'].replace(-1, 12, inplace=True)
    """

    train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
    test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集

    # print train.info()        # 32*478111
    # print test.info()         # 31*18371
    # test数据中缺少一行is_trade，也就是我们要训练的是否购买的信息

    features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                ]
    # lgb具有指定类别特征的功能，具体哪些特征要作为类别特征，还需要大家自己考虑哈~~
    target = ['is_trade']

    clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
    clf.fit(train[features], train[target], feature_name=features,
            categorical_feature=['user_gender_id', ])

    test['lgb_predict'] = clf.predict_proba(test[features], )[:, 1]
    train['lgb_predict'] = clf.predict_proba(train[features], )[:, 1]
    print('test:', log_loss(test[target], test['lgb_predict']))
    print('train:', log_loss(train[target], train['lgb_predict']))

"""
num_leaves=62, max_depth=6, n_estimators=122
0.0819673332636
0.0831694291858
num_leaves=62, max_depth=7, n_estimators=82
('test:', 0.081988824927125892)
('train:', 0.084097475327490387)
num_leaves=100, max_depth=7, n_estimators=82
('test:', 0.082141071940912996)
('train:', 0.082469249880585696)
num_leaves=150, max_depth=7, n_estimators=82
('test:', 0.082016997218832335)
('train:', 0.082253941014037754)
num_leaves=100, max_depth=7, n_estimators=100
('test:', 0.082175973657571774)
('train:', 0.081192812698486716)
明显过拟合了，准确率并不好
num_leaves=150, max_depth=7, n_estimators=82, min_data_in_leaf=100, n_jobs=20
('test:', 0.081959635716969723)
('train:', 0.083780574614552739)
num_leaves=110, max_depth=7, n_estimators=82
('test:', 0.082067432954116309)
('train:', 0.082437326080272946)
实测结果：0.08324
不补充缺失值的话
('test:', 0.082057555810688743)
('train:', 0.082362479981964956)
num_leaves=63, max_depth=7, n_estimators=80
('test:', 0.08183320942298164)
('train:', 0.084157527086740125)
实测0.08310
"""
"""
n_estimators RF最大的决策树个数
n_estimators太小，容易欠拟合，n_estimators太大，计算量会太大
并且n_estimators到一定的数量后，再增大n_estimators获得的模型提升会很小，所以一般选择一个适中的数值,默认是100
mx_depth是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本，一般不用考虑
num_leaves = 2^(max_depth),就可以完成depth-wise tree growth和leaf-wise growth的转换
但是在实践中这种简单的转换并不能够有好的结果，设置相同的时候，leaf-wise growth树会比depth-wise tree growth要深很多，容易过拟合
所以在调这个参数的时候，策略就是num_leaves < 2^(max_depth)，一般比100大
较大的max_bin(但会是模型速度下降)能提高精度
使用较小max_bin.能避免过拟合
使用较小 num_leaves 能避免过拟合
min_data_in_leaf设置的大则可以避免建立过于深的树，但是会造成欠拟合，一般设置100以上比较合适。
"""

"""
lightGBM使用leaf-wise growth算法，该算法收敛速度要快很多，但同时容易过拟合
"""