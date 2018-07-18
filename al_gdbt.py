#!/usr/bin/env python    # -*- coding: utf-8 -*
import time
import pandas as pd
from sklearn.metrics import log_loss
import warnings
from sklearn import metrics
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

"""对GDBT进行调参"""

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

    # 定义GBDT模型
    gbdt = GradientBoostingClassifier(n_estimators=170, min_samples_split=3, min_samples_leaf=8)
    # 调参之后的GBDT模型

    # 训练学习
    gbdt.fit(X_train, Y_train)
    # 预测及AUC评测
    Y_predict_gbdt = gbdt.predict_proba(X_test)[:, 1]
    print('test:', log_loss(Y_test, Y_predict_gbdt))
    print "Accuracy : %.4f" % metrics.roc_auc_score(Y_test, Y_predict_gbdt)

    """
    # 原始结果为：('test:', 0.081939773937662927)
    # Accuracy : 0.6848
    目前GDBT结果
    ('test:', 3, 0.081778211935922926)
    Accuracy : 0.6864
    线上结果为：0.08344
    目前最优的对数损失0.08161589381677363
    准确率0.6874
    """

    """
    n_estimators默认是100：
    ('test:', 160, 0.081822593361136106)
    Accuracy : 0.6855
    ('test:', 170, 0.081778956639371334)
    Accuracy : 0.6864
    ('test:', 180, 0.081777704065840603)
    Accuracy : 0.6863
    ('test:', 190, 0.081774009578415477)
    Accuracy : 0.6859
    感觉有点过拟合了，按照170
    我们搜索接下来的参数
    max_depth默认是3，搜索2,4为可能的
    内部节点再划分所需最小样本数min_samples_split,默认是2，我们用100,200测试
    ('test:', 3, 100, 0.081839874888073177)
    Accuracy : 0.6863
    ('test:', 3, 200, 0.085286685353935299)
    Accuracy : 0.6863
    ('test:', 4, 100, 0.082243048544814762)
    Accuracy : 0.6845
    ('test:', 4, 200, 0.082227519751189859)
    Accuracy : 0.6828
    判断深度为3时候的min_samples_split，测试20-100之间的结果
    ('test:', 20, 0.084960941064984863)
    Accuracy : 0.6857
    ('test:', 40, 0.085264816856556083)
    Accuracy : 0.6852
    ('test:', 60, 0.085264816856556083)
    Accuracy : 0.6852
    ('test:', 80, 0.085080009776700646)
    Accuracy : 0.6859
    换成1到10了
    ('test:', 3, 0.081778211935922926)
    Accuracy : 0.6864
    ('test:', 5, 0.081793534448781596)
    Accuracy : 0.6861
    发现3的比默认的2好一点点
    对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
    ('test:', 2, 0.081779423184746222)
    Accuracy : 0.6853
    ('test:', 4, 0.081788267825988262)
    Accuracy : 0.6854
    ('test:', 6, 0.081728317969095035)
    Accuracy : 0.6851
    ('test:', 8, 0.081691332870021324)
    Accuracy : 0.6862
    """

    """
    调参
    随着树的增加，GBM不会overfit，但如果learning_rate的值较大，会overfiting
    如果减小learning_rate、并增加树的个数，在个人电脑上计算开销会很大
    1.选择一个相对高的learning_rate。缺省值为0.1，通常在0.05到0.2之间都应有效
    2.根据这个learning_rate，去优化树的数目。这个范围在[40,70]之间。记住，选择可以让你的电脑计算相对较快的值。因为结合多种情况才能决定树的参数.
    3.调整树参数，来决定learning_rate和树的数目。注意，我们可以选择不同的参数来定义树。
    4.调低learning_rate，增加estimator的数目，来得到更健壮的模型
    应首先调节对结果有较大影响的参数。例如，首先调整max_depth 和 min_samples_split，它们对结果影响更大
    """