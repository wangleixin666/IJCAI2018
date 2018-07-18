#!/usr/bin/env python    # -*- coding: utf-8 -*

import time
import pandas as pd
import lightgbm as lgb
# from sklearn.feature_selection import SelectFromModel
# 进行树的特征选择
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


# 格式化输出当地时间
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    # strftime返回以可读字符串表示的当地时间
    return dt
    # %Y 四位数的年份表示 %m 月份（01-12）%d 月内中的一天（0-31）
    # %H 24小时制小时数（0-23）%M 分钟数（00=59）%S 秒（00-59）
    # 返回形如200x-0x-0x 14:43:54 几几年几几月几几天 小时：分钟：秒


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
    # drop_duplicate方法是对DataFrame格式的数据，去除特定列下面的重复行
    data = convert_data(data)
    # print data.info()
    # 一共478110组数据
    # 32个特征

    """
    直接对data数据进行缺失值统计和补充(主要针对非商品信息,item一般没有缺失)
    user_age_level用户年龄等级       -1       964
    user_gender_id用户性别编号       -1     12902
    user_occupation_id用户职业编号   -1       964
    user_star_level用户星级          -1       964
    item_sales_level 商品销售登等级   -1       913
    """

    # 接下来对缺失值进行处理
    # 然而xgboost会自己处理缺失值
    # 把缺失值分别放到左叶子节点和右叶子节点中，计算增益，哪个增益大就放到哪个叶子节点

    data['user_age_level'].replace(-1, 1003, inplace=True)
    data['user_gender_id'].replace(-1, 0, inplace=True)
    data['user_occupation_id'].replace(-1, 2005, inplace=True)
    data['user_star_level'].replace(-1, 3006, inplace=True)
    data['item_sales_level'].replace(-1, 12, inplace=True)

    train = data.loc[data.day < 24]  # 18,19,20,21,22,23 # 一共420693行，32列
    test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集 # 一共420693

    # print train.info()
    # 进行特征选取

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

    """进行特征工程
    首先是选择方差较大的特征，进行选择，利用树常用的SelectFromModel
    print X_train.shape # (420693, 22)
    model = SelectFromModel(clf, prefit=True)
    train_new = model.transform(train)
    print clf.feature_importances_
    方差越大越重要，我们选取比均值的1.25倍还大的作为特征
    model = SelectFromModel(clf, prefit=True)
    print model
    model.transform(X_train)
    print X_train.shape # 还是没有变化啊 (420693, 22)
    只能说明特征都相关，没有干扰特征
    """

    """接下来考虑给特征进行降维"""
    sc = StandardScaler()
    X_train_new = sc.fit_transform(X_train)
    X_test_new = sc.fit_transform(X_test)
    # print X_train.shape
    # 对数据进行PCA降维之前需要先进行标准化

    estimator = PCA(n_components=19)
    # 或者直接用n_components=0.x选择前百分之几的特征
    X_train_new_next = estimator.fit_transform(X_train_new)
    X_test_new_next = estimator.fit_transform(X_test_new)

    clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
    clf.fit(X_train_new_next, Y_train)
    Y_predict = clf.predict_proba(X_test_new_next, )[:, 1]
    print('test:', log_loss(Y_test, Y_predict))

    # clf.fit(X_train, Y_train, categorical_feature=['user_gender_id', ])
    # Y_predict = clf.predict_proba(X_test, )[:, 1]
    # print Y_predict.shape
    # 没有后缀的话(57418L, 2L),但是我们需要的是1列，所以加上[:, 1]

"""
PCA维数的差别
22 ('test:', 0.084121393248075146)
21 ('test:', 0.082767857059916281)
20 ('test:', 0.082845616801928423)
19 ('test:', 0.082579656212406163)
18 ('test:', 0.083220139876504409)
然而如果不补充缺失值的话
22 ('test:', 0.084121393248075146)
19 ('test:', 0.084762049133869685)
然后是模型参数问题
# num_leaves=110, max_depth=7, n_estimators=82
num_leaves=63, max_depth=7, n_estimators=80
22 ('test:', 0.082714299772016928)
19 ('test:', 0.082786916600928753)
"""

"""
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        learning_rate=0.1, max_depth=7, min_child_samples=20,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=82,
        n_jobs=20, num_leaves=110, objective=None, random_state=None,
        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=1),
        norm_order=1, prefit=True, threshold='1.25*mean')     
('test:', 0.082057555810688743)
('train:', 0.082362479981964956)
"""

"""
PCA和LDA有很多的相似点，其本质是要将原始的样本映射到维度更低的样本空间中
但是PCA和LDA的映射目标不一样：PCA是为了让映射后的样本具有最大的发散性；而LDA是为了让映射后的样本有最好的分类性能
所以说PCA是一种无监督的降维方法，而LDA是一种有监督的降维方法。
"""