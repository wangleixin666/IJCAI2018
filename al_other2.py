# coding:utf-8

import pandas as pd
import time
import warnings
import numpy as np

np.random.seed(2018)
warnings.filterwarnings("ignore")


# 时间处理
def time2cov(time_):
    '''
    时间是根据天数推移，所以日期为脱敏，但是时间本身不脱敏
    :param time_:
    :return:
    '''
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_))


train = pd.read_csv('D:\kaggle\\alimm\\round1_ijcai_18_train_20180301.txt', sep=" ")
train = train.drop_duplicates(['instance_id'])
train = train.reset_index(drop=True)
test_a = pd.read_csv('D:\kaggle\\alimm\\round1_ijcai_18_test_a_20180301.txt', sep=" ")

all_data = pd.concat([train, test_a])
all_data['real_time'] = pd.to_datetime(all_data['context_timestamp'].apply(time2cov))
all_data['real_hour'] = all_data['real_time'].dt.hour
all_data['real_day'] = all_data['real_time'].dt.day


def time_change(hour):
    hour = hour - 1
    if hour == -1:
        hour = 23
    return hour


def time_change_1(hour):
    hour = hour + 1
    if hour == 24:
        hour = 0
    return hour


all_data['hour_before'] = all_data['real_hour'].apply(time_change)
all_data['hour_after'] = all_data['real_hour'].apply(time_change_1)

# 18 21 19 20 22 23 24 | 25
# print(all_data['real_day'].unique())


def c_log_loss(y_t, y_p):
    tmp = np.array(y_t) * np.log(np.array(y_p)) + (1 - np.array(y_t)) * np.log(1 - np.array(y_p))
    return -np.sum(tmp) / len(y_t), False


# 获取当前时间之前的前x天的转化率特征
def get_before_cov_radio(all_data, label_data,
                         cov_list=list(['shop_id', 'item_id', 'real_hour', 'item_pv_level', 'item_sales_level']),
                         day_list=list([1, 2, 3])):
    result = []
    r = pd.DataFrame()
    label_data_time = label_data['real_day'].min()
    label_data_time_set = label_data['real_day'].unique()
    # print('label set day', label_data_time_set)
    for cov in cov_list:
        for d in day_list:
            feat_set = all_data[
                (all_data['real_day'] >= label_data_time - d) & (all_data['real_day'] < label_data_time)
                ]
            # print("cov feature", feat_set['real_day'].unique())
            # print("cov time", cov)

            tmp = feat_set.groupby([cov], as_index=False).is_trade.agg({'mean': np.mean, 'count': 'count'}).add_suffix(
                "_%s_before_%d_day" % (cov, d))

            tmp.rename(columns={'%s_%s_before_%d_day' % (cov, cov, d): cov}, inplace=True)

            if d == 1:
                r = tmp
            else:
                r = pd.merge(r, tmp, on=[cov], how='outer').fillna(0)

        result.append(r)
    return result


def calc_categry_feat(data):
    data['item_category_list_1'] = data['item_category_list'].apply(lambda x: int(x.split(';')[0]))
    data['item_category_list_2'] = data['item_category_list'].apply(lambda x: int(x.split(';')[1]))
    data['item_property_list_0'] = data['item_property_list'].apply(lambda x: int(x.split(';')[0]))
    data['item_property_list_1'] = data['item_property_list'].apply(lambda x: int(x.split(';')[1]))
    data['item_property_list_2'] = data['item_property_list'].apply(lambda x: int(x.split(';')[2]))

    for i in range(3):
        data['predict_category_%d' % (i)] = data['predict_category_property'].apply(
            lambda x: int(str(x.split(";")[i]).split(":")[0]) if len(x.split(";")) > i else -1
        )

    for item_cate in ['item_category_list_1', 'item_category_list_2']:
        for pre_item_cate in ['predict_category_0', 'predict_category_1', 'predict_category_2']:
            data['%s_%s' % (item_cate, pre_item_cate)] = data[item_cate] == data[pre_item_cate]
            data['%s_%s' % (item_cate, pre_item_cate)] = data['%s_%s' % (item_cate, pre_item_cate)].astype(int)

    del data['item_category_list']
    del data['item_property_list']
    del data['predict_category_property']
    return data


take_columns = ['instance_id', 'item_id', 'shop_id', 'user_id', 'is_trade']

shop_current_col = [
    'shop_score_description', 'shop_score_delivery', 'shop_score_service',
    'shop_star_level', 'shop_review_positive_rate', 'shop_review_num_level'
]

user_col = [
    'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level'
]

item_col = [
    'item_brand_id', 'item_city_id', 'item_price_level',
    'item_sales_level', 'item_collected_level', 'item_pv_level',
    'item_category_list', 'item_property_list'
]
time_feat = ['real_hour', 'hour_before', 'hour_after', 'context_timestamp', 'real_day']

context_col = ['predict_category_property', 'context_page_id']

feat = take_columns + shop_current_col + time_feat + user_col + item_col + context_col


def get_history_user_feat(all_data, data):
    label_data_time = data['real_day'].min()
    # print(label_data_time)

    tmp = all_data[all_data['real_day'] < label_data_time]
    # print(tmp['real_day'].unique())

    user_time = tmp.groupby(['user_id'], as_index=False).context_timestamp.agg({'day_begin': 'min', 'day_end': 'max'})
    user_time['alive'] = user_time['day_end'] - user_time['day_begin']
    user_time['s_alive'] = label_data_time - user_time['day_begin']
    user_time['alive/s_alive'] = user_time['alive'] / user_time['s_alive']

    user_time_cov = tmp[tmp['is_trade'] == 1]
    user_time_cov = user_time_cov.groupby(['user_id'], as_index=False).context_timestamp.agg({'day_end_cov': 'max'})

    user_time_cov = pd.DataFrame(user_time_cov).drop_duplicates(['user_id', 'day_end_cov'])

    data = pd.merge(data, user_time[['user_id', 'alive', 's_alive', 'alive/s_alive', 'day_begin', 'day_end']],
                    on=['user_id'], how='left')

    data = pd.merge(data, user_time_cov, on=['user_id'], how='left')
    data['day_end_cov'] = data['day_end_cov'].fillna(data['day_end'])

    data['alive_cov'] = data['day_end_cov'] - data['day_begin']
    data['alive/alive_cov'] = data['alive'] / data['alive_cov']
    # data['s_alive/alive_cov'] = data['s_alive'] / data['alive_cov']

    del data['day_end_cov']
    del data['day_end']
    del data['day_begin']

    return data.fillna(-1)


def get_history_shop_feat(all_data, data):
    label_data_time = data['real_day'].min()
    # print(label_data_time)
    for i in [1, 2, 3]:
        tmp = all_data[(all_data['real_day'] < label_data_time) & (all_data['real_day'] >= label_data_time - i)]

        shop_score_service_hour = tmp.groupby(['real_hour'], as_index=False)[
            'shop_score_service'] \
            .agg({
            'shop_score_service_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_score_service_hour, 'left', on=['real_hour'])

        shop_score_delivery = tmp.groupby(['real_hour'], as_index=False)[
            'shop_score_delivery'] \
            .agg({
            'shop_score_delivery_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_score_delivery, 'left', on=['real_hour'])

        shop_score_service_hour = tmp.groupby(['real_hour'], as_index=False)[
            'shop_score_description'] \
            .agg({
            'shop_score_description_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_score_service_hour, 'left', on=['real_hour'])

        shop_review_positive_rate = tmp.groupby(['real_hour'], as_index=False)[
            'shop_review_positive_rate'] \
            .agg({
            'shop_review_positive_rate_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_review_positive_rate, 'left', on=['real_hour'])

        shop_star_level = tmp.groupby(['real_hour'], as_index=False)[
            'shop_star_level'] \
            .agg({
            'shop_star_level_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_star_level, 'left', on=['real_hour'])

        shop_review_num_level = tmp.groupby(['real_hour'], as_index=False)[
            'shop_review_num_level'] \
            .agg({
            'shop_review_num_level_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_review_num_level, 'left', on=['real_hour'])

        shop_query_day_hour = tmp.groupby(['shop_id', 'real_hour']).size().reset_index().rename(
            columns={0: 'shop_query_day_hour_%d' % (i)})
        data = pd.merge(data, shop_query_day_hour, 'left', on=['shop_id', 'real_hour'])

    return data


def get_history_item_feat(all_data, data):
    for i in [1, 2, 3]:
        tmp = all_data[
            (all_data['real_day'] < data['real_day'].min()) & (all_data['real_day'] >= data['real_day'].min() - i)]

        item_brand_id_day = tmp.groupby(['item_city_id', 'real_hour']).size().reset_index().rename(
            columns={0: 'item_brand_id_day_%d' % (i)})
        data = pd.merge(data, item_brand_id_day, 'left', on=['item_city_id', 'real_hour'])

        item_brand_id_hour = tmp.groupby(['item_brand_id', 'real_hour']).size().reset_index().rename(
            columns={0: 'item_brand_id_hour_%d' % (i)})
        data = pd.merge(data, item_brand_id_hour, 'left', on=['item_brand_id', 'real_hour'])
        item_pv_level_hour = tmp.groupby(['item_pv_level', 'real_hour']).size().reset_index().rename(
            columns={0: 'item_pv_level_hour_%d' % (i)})
        data = pd.merge(data, item_pv_level_hour, 'left', on=['item_pv_level', 'real_hour'])
        #
        # item_pv_level_day = data.groupby(['real_day','real_hour'], as_index=False)['item_pv_level'] \
        #     .agg({'item_pv_level_day_mean_%d'%(i): 'mean',
        #           'item_pv_level_day_median_%d'%(i): 'median',
        #           'item_pv_level_day_std_%d'%(i): 'std'
        #           })
        # data = pd.merge(data, item_pv_level_day, 'left', on=['real_day','real_hour'])
    return data


# print('make feat')


def make_feat(data, feat):
    '''
    :param data: 标签数据，当前时刻的用户特征
    :param feat: 特征数据，统计的用户特征
    :return: 拼接后的特征
    '''

    data = calc_categry_feat(data)
    data = get_history_user_feat(all_data, data)
    data = get_history_shop_feat(all_data, data)
    data = get_history_item_feat(all_data, data)

    for f in feat:
        data = pd.merge(data, f, on=[f.columns[0]], how='left')

    return data.fillna(0)


test_a = all_data[train.shape[0]:]

train = all_data[:train.shape[0]]
val_a = train[train['real_day'] == 24]
train_a = train[train['real_day'] == 23]
train_b = train[train['real_day'] == 22]
train_c = train[train['real_day'] == 21]

# 传入全部数据和当前标签数据
test_cov_feat = get_before_cov_radio(all_data, test_a)
val_cov_feat = get_before_cov_radio(all_data, val_a)

train_cov_feat_a = get_before_cov_radio(all_data, train_a)
train_cov_feat_b = get_before_cov_radio(all_data, train_b)
train_cov_feat_c = get_before_cov_radio(all_data, train_c)

train_a = make_feat(train_a[feat], train_cov_feat_a)
train_b = make_feat(train_b[feat], train_cov_feat_b)
train_c = make_feat(train_c[feat], train_cov_feat_c)

test_a = make_feat(test_a[feat], test_cov_feat)
val_a = make_feat(val_a[feat], val_cov_feat)

train = pd.concat([train_a, train_b])
train = pd.concat([train, train_c])

# print(train.shape)
# train = pd.concat([train,val_a])
# print(train.shape)

y_train = train.pop('is_trade')
train_index = train.pop('instance_id')
X_train = train

y_test = test_a.pop('is_trade')
test_index = test_a.pop('instance_id')
X_test = test_a

y_val = val_a.pop('is_trade')
val_index = val_a.pop('instance_id')
X_val = val_a

# print(train.head())

category_list = [
    'item_id', 'shop_id', 'user_id', 'user_gender_id', 'user_age_level',
    'user_occupation_id', 'user_star_level',
    'item_brand_id', 'item_city_id', 'item_price_level',
    'item_sales_level', 'item_collected_level', 'item_pv_level',
    'shop_review_num_level', 'shop_star_level', 'item_category_list_1', 'item_category_list_2',
    'item_property_list_0', 'item_property_list_1', 'item_property_list_2',
    'predict_category_0', 'predict_category_1', 'predict_category_2', 'context_page_id'
]


def make_cat(data):
    for i in category_list:
        data[i] = data[i].astype('category')
    return data


train_test_val = pd.concat([X_train, X_test])
train_test_val = pd.concat([train_test_val, X_val])
train_test_val = train_test_val.reset_index(drop=True)

# train_test_val = make_cat(train_test_val)
#
# X_train = train_test_val[:X_train.shape[0]]
# X_test = train_test_val[X_train.shape[0]:X_train.shape[0]+X_test.shape[0]]
# X_val = train_test_val[X_train.shape[0]+X_test.shape[0]:]

X_train = make_cat(X_train)
X_test = make_cat(X_test)
X_val = make_cat(X_val)

# print(X_train.shape)
# (203112, 104)
# print(X_test.shape)
# (18371, 104)
# print(X_val.shape)
# (57411, 104)

# X_test = make_cat(X_test)
# X_val = make_cat(X_val)

del X_train['hour_before']
del X_test['hour_before']
del X_val['hour_before']

del X_train['hour_after']
del X_test['hour_after']
del X_val['hour_after']

del X_train['real_day']
del X_test['real_day']
del X_val['real_day']

# print(X_train.dtypes)

del X_train['context_timestamp']
del X_test['context_timestamp']
del X_val['context_timestamp']

X_train = X_train[X_train.columns]
X_test = X_test[X_train.columns]
X_val = X_val[X_train.columns]

import lightgbm as lgb
from sklearn.metrics import log_loss
# 线下学习
gbm = lgb.LGBMClassifier(max_depth=2)

gbm.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['binary_logloss'],
        early_stopping_rounds=50)

# y_pred_1 = gbm.predict(X_val, num_iteration=gbm.best_iteration_)

y_pred_1 = gbm.predict_proba(X_val)

print(log_loss(y_val, y_pred_1))

"""
gbm classifier默认的结果为
0.0832652213537
max_depth默认是-1
(2, 0.082622634083773544)
(4, 0.083531443179165596)
(6, 0.084490071779603493)
learnint_rate
(0.05, 0.083419310870096783)
(0.15, 0.082692228738167881)
(0.2, 0.082619060542378789)
(20, 0.12412446275665344)
(100, 0.082622634083773544)
(200, 0.082622634083773544)
"""
"""
默认为0.0830834230403
调参后0.082622634083773544
"""

"""
8                     shop_review_num_level    0
98      count_item_sales_level_before_3_day    0
86             count_real_hour_before_3_day    0
18                     item_collected_level    0
67                      item_brand_id_day_3    0
19                            item_pv_level    0
17                         item_sales_level    0
21                     item_category_list_1    0
23                     item_property_list_0    0
24                     item_property_list_1    0
25                     item_property_list_2    0
16                         item_price_level    0
27                       predict_category_1    0
15                             item_city_id    0
29  item_category_list_1_predict_category_0    0
30  item_category_list_1_predict_category_1    0
31  item_category_list_1_predict_category_2    0
32  item_category_list_2_predict_category_0    0
33  item_category_list_2_predict_category_1    0
34  item_category_list_2_predict_category_2    0
37                            alive/s_alive    0
38                                alive_cov    0
39                          alive/alive_cov    0
41           shop_score_delivery_hour_std_1    0
45         shop_review_num_level_hour_std_1    0
46                    shop_query_day_hour_1    0
47            shop_score_service_hour_std_2    0
51               shop_star_level_hour_std_2    0
54            shop_score_service_hour_std_3    0
99       mean_item_sales_level_before_3_day    0
这些特征没啥用

0                                   item_id  105
1                                   shop_id   39
2                                   user_id   15
14                            item_brand_id   14
75                mean_shop_id_before_3_day   12
87              mean_real_hour_before_3_day   10
79                mean_item_id_before_2_day    8
28                       predict_category_2    6
40            shop_score_service_hour_std_1    6
76               count_item_id_before_1_day    5
81                mean_item_id_before_3_day    5
3                    shop_score_description    5
26                       predict_category_0    5
73                mean_shop_id_before_2_day    4
57     shop_review_positive_rate_hour_std_3    4
20                          context_page_id    4
85              mean_real_hour_before_2_day    4
50     shop_review_positive_rate_hour_std_2    4
13                          user_star_level    4
53                    shop_query_day_hour_2    3
48           shop_score_delivery_hour_std_2    3
11                           user_age_level    3
69                     item_pv_level_hour_3    3
43     shop_review_positive_rate_hour_std_1    3
60                    shop_query_day_hour_3    3
35                                    alive    3
59         shop_review_num_level_hour_std_3    2
77                mean_item_id_before_1_day    2
44               shop_star_level_hour_std_1    2
42        shop_score_description_hour_std_1    2
"""

# print('Start predicting...')
# predict

# print(log_loss(y_train, y_tt))


"""
0.0831469157809
0.0783179459843
"""
"""
# 线上提交

gbm_sub = lgb.LGBMRegressor(objective='binary',
                            num_leaves=32,
                            learning_rate=0.01,
                            n_estimators=gbm.best_iteration_ + 1,
                            colsample_bytree=0.65,
                            subsample=0.65,
                            seed=0
                            )

X_train = pd.concat([X_train, X_val])
y_train = pd.concat([y_train, y_val])

X_train = make_cat(X_train)

X_train = X_train[X_train.columns]

gbm_sub.fit(X_train, y_train,
            eval_set=[(X_train, y_train)],
            eval_metric=['binary_logloss'])

y_sub_1 = gbm_sub.predict(X_test)

y_tt = gbm_sub.predict(X_train, num_iteration=gbm_sub.best_iteration_)

from sklearn.metrics import log_loss

print(log_loss(y_train, y_tt))

sub = pd.DataFrame()
sub['instance_id'] = list(test_index)

sub['predicted_score'] = list(y_sub_1)

sub.to_csv('D:\kaggle\\alimm\\baseline_07.csv', sep=" ", index=False)
"""