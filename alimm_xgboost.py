# -*- coding: utf-8 -*

from kaggle.XGBoost import XGBClassifier
import time
import pandas as pd
# from sklearn.metrics import log_loss
from sklearn import metrics
# 引入打分机制
from sklearn.model_selection import GridSearchCV
# 引入调参

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
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    return data


if __name__ == "__main__":
    # online = False
    # 这里用来标记是 线下验证 还是 在线提交

    data = pd.read_csv('D:\kaggle\\alimm\\round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)

    train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
    test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集

    features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                ]

    # train = train[features]
    target = ['is_trade']
    # 类似于y_train

    # xgbc = XGBClassifier()
    xgbc1 = XGBClassifier(
        learning_rate=0.1,
        max_depth=3,
        seed=100,
        n_estimators=140,
        # min_child_weight=1,
        # scale_pos_weight=1
        # 因为样本类别不均
    )

    param_test1 = {
        'n_estimators': range(100, 600, 200),
        # 'min_chlid_weight': range(1, 6, 2)
    }
    gs = GridSearchCV(xgbc1, param_test1, cv=5)

    gs.fit(train[features], train[target])
    print gs.best_params_

    test['gs_predict'] = gs.predict(test[features])
    print metrics.accuracy_score(test[target], test['gs_predict'])

    """
    调参过程
    选择较高的学习速率(learning rate)。一般情况下，学习速率的值为0.1。但是，对于不同的问题，理想的学习速率有时候会在0.05到0.3之间波动
    # 选择对应于此学习速率的理想决策树数量。XGBoost有一个很有用的函数“cv”，这个函数可以在每一次迭代中使用交叉验证，并返回理想的决策树数量。
    对于给定的学习速率和决策树数量，进行决策树特定参数调优(max_depth, min_child_weight, gamma, subsample, colsample_bytree)
    在确定一棵树的过程中，我们可以选择不同的参数，待会儿我会举例说明。
    xgboost的正则化参数的调优。(lambda, alpha)。这些参数可以降低模型的复杂度，从而提高模型的表现。
    降低学习速率，确定理想参数
    """

    # xgbc1.fit(train[features], train[target])

    # test['xgbc_predict'] = xgbc1.predict(test[features])
    # print metrics.accuracy_score(test[target], test['xgbc_predict'])

# xgbc1 0.983141175241
# xgbc 0.983141175241
# xgbc2{'max_depth': 3} 0.983141175241
"""
    for date in range(18, 25, 1):

        test = data.loc[data.day == date]  # 暂时先使用第24天作为验证集

        features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                    'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                    'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                    'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                    'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                    ]

        # train = train[features]
        target = ['is_trade']
        # 类似于y_train

        xgbc = XGBClassifier()
        xgbc.fit(train[features], train[target])

        test['xgbc_predict'] = xgbc.predict_proba(test[features], )[:, 1]

        loss = log_loss(test[target], test['xgbc_predict'])

        list_loss.append(loss)

    for j in range(len(list_loss)):
        summary += float(list_loss[j])

    print summary / len(list_loss)
    
    xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
 base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=140, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=1000, silent=True, subsample=1
"""
# 初始xgboost结果为 0.088105707276
# 初始lightgbm结果为 0.0841988667303
