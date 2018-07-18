#!/usr/bin/env python    # -*- coding: utf-8 -*
# 考虑利用网格搜索找到最优的lightgbm最优的参数搭配
# 本地测试从0.083到0.0828
# 感觉过拟合了，测试从0.08310到了0.08340，考虑换种方法

import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
# 使用并行网格搜索的方式寻找更好的超参数组合，期待进一步提高lightgbmClassifier的性能
# 发现本地测试如果用第23天的话准确率接近在线提交的0.0831，所以有可能不是最优的参数组合
# 还是需要更新参数的
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

    list_loss = []
    summary = 0

    for date in range(18, 25, 1):

        test = data.loc[data.day == date]  # 暂时先使用第24天作为验证集

        features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                    'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                    'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                    'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                    'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                    ]
        target = ['is_trade']

        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=6, n_estimators=122, n_jobs=20)

        # clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=82, n_jobs=20)
        # 此时的参数组合，本地准确率为0.0831

        clf.fit(train[features], train[target], feature_name=features,
                categorical_feature=['user_gender_id', ])

        test['lgb_predict'] = clf.predict_proba(test[features], )[:, 1]
        loss = log_loss(test[target], test['lgb_predict'])

        list_loss.append(loss)

    for j in range(len(list_loss)):
        summary += float(list_loss[j])

    print summary / len(list_loss)

"""
交叉验证结果
num_leaves=62, max_depth=6, n_estimators=122   0.082853
num_leaves=63, max_depth=7, n_estimators=82    0.0835185255839
num_leaves=63, max_depth=6, n_estimators=82    0.0846498814085

7层明显慢很多的，结果很明显过拟合了
"""

"""
发现正负样本数不均衡
"""

"""
依次验证下6层没问题，122个n_estimators，不过节点个数2^6是64
60, 0.082965083901661671
61, 0.082930202392248256
62, 0.082808151161040344
63, 0.082872591418784822
64, 0.083029658057337058
65, 0.083029658057337058
"""

"""
n_estimators间隔为10的时候
70,0.0829526973093
80,0.0829526973093
90 0.082959299909879894
100 0.082952697309292556
110 0.08285927401901777
120 0.082814997345373759
130 0.082831350552556374
140 0.082854744072588596
大致范围在110到130之间，间隔变为5
115, 0.082837873913491511
120, 0.082814997345373759
125, 0.082825706394878185
判断在120左右间隔变为2
118, 0.082827889115900874
120, 0.082814997345373759
122, 0.082808151161040344
间隔变为1
121, 0.082809533999878948
122, 0.082808151161040344
123, 0.082809579875808004
"""

"""
学习率从0.1变一下
0.05的话0.0859198028795，0.15的话0.0830454566613
所以还是用默认的比较好
"""

"""
在深度为6的时候，节点个数从64开始，准确率为0.0831515984328
65就是0.0831515984328.。最大就是2^max_depth
63的时候变成0.0830366736545
62的时候0.0829961849687
61的时候0.0830119877698
此时的最优组合为深度为6，节点个数为62
"""

"""
params = {
    # "learning_rate": [0.1, 0.2, 0.3, 0.4],
    # 'feature_fraction': (0.5, 0.1, 0.9),
    # 'num_leaves': [60, 63],
    # 'max_depth': [6, 7, 8]
}

clf_best = GridSearchCV(clf, param_grid=params, cv=3, n_jobs=20)
# 3折交叉验证
clf_best.fit(train[features], train[target], feature_name=features, categorical_feature=['user_gender_id', ])

print clf_best.best_params_

调参过程
1、是7的话结果是0.083152277448，max_depth = 8的话，准确率为0.0830853309564，有所提高
当最深层变成6时，准确率同样是0.0830366736545，5的话0.0830124198792
'max_depth': 3,准确率是0.0834907249694
# 不过层数变多的话也可能造成过拟合，所以结果不可信
2、学习率也就是梯度下降法每次的变化的步幅，不设置的话默认为0.1

test['lgb_best_predict'] = clf_best.predict_proba(test[features], )[:, 1]
print(log_loss(test[target], test['lgb_best_predict']))
"""

"""

params = {
    # "learning_rate": [0.1, 0.2, 0.3, 0.4],
    # 'feature_fraction': (0.5, 0.1, 0.9),
    'num_leaves': [60, 63],
    # 'max_depth': [5, 6, 7, 8],
    'n_estimators': [80, 85]
}

clf_best = GridSearchCV(clf, param_grid=params, cv=3, n_jobs=20)
# 3折交叉验证
clf_best.fit(train[features], train[target], feature_name=features, categorical_feature=['user_gender_id', ])

print clf_best.best_params_

test['lgb_best_predict'] = clf_best.predict_proba(test[features], )[:, 1]
print(log_loss(test[target], test['lgb_best_predict']))
# 叶子节点数设为63，深度设置为7，学习率未设置，为默认的0.1，测试发现最优的
# 因为决策树较容易过拟合，深度有待考虑
# 此处用的参数已经是经过网格搜索后的结果，已经比较好
"""

"""
params = {
# "learning_rate": [0.1, 0.2, 0.3, 0.4],
# 'feature_fraction': (0.5, 0.1, 0.9),
# 'num_leaves': [60, 63],
# 'max_depth': [6, 7, 8]
}

clf_best = GridSearchCV(clf, param_grid=params, cv=3, n_jobs=20)
# 3折交叉验证
clf_best.fit(train[features], train[target], feature_name=features, categorical_feature=['user_gender_id', ])

print clf_best.best_params_

"""