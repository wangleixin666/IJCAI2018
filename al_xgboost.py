#!/usr/bin/env python    # -*- coding: utf-8 -*

import time
import pandas as pd
from kaggle.XGBoost import XGBClassifier
from sklearn.metrics import log_loss
import warnings
from sklearn import metrics

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

    # print X_train.shape # (420693, 22)
    # print Y_train.shape # (420693, 1)
    # print Y_test.shape # (57418, 1)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        min_child_weight=2,
        subsample=0.7,
        colsample_bytree=0.7
    )
    # clf = lgb.LGBMClassifier()
    clf.fit(X_train, Y_train)
    Y_predict = clf.predict_proba(X_test, )[:, 1]
    print('test:', log_loss(Y_test, Y_predict))
    print "Accuracy : %.4f" % metrics.roc_auc_score(Y_test, Y_predict)
"""
XGboost参数黄金搭配
learning_rate=0.05, max_depth=6, n_estimators=500
('test:', 0.081942890328163834)
Accuracy : 0.6822
xgboost默认参数（不调参的话）
('test:', 0.081944846441379712)
Accuracy : 0.6835
先调整 最佳迭代次数 n_estimators：
('test:', 100, 0.081944846441379712)
Accuracy : 0.6835
('test:', 200, 0.081717427233321677)
Accuracy : 0.6859
('test:', 300, 0.081672002245841077)
Accuracy : 0.6859
('test:', 400, 0.081659112599510877)
Accuracy : 0.6862
('test:', 500, 0.081696718870915314)
Accuracy : 0.6848
('test:', 600, 0.081714378988268374)
Accuracy : 0.6843
缩小间隔为50到350--450附近的范围
('test:', 350, 0.081663509052391434)
Accuracy : 0.6862
('test:', 400, 0.081659112599510877)
Accuracy : 0.6862
('test:', 450, 0.081678557042738145)
Accuracy : 0.6854
然后就不用更精确了，选定400了
目前对数损失结果为：0.081659112599510877
准确度结果为：0.6862

接下来是max_depth，默认是6？？？？？？
默认就是3。。。。。。。。谁说的6！！！：
min_child_weight默认是1，然后两个参数一起调
max_depth=6, min_child_weight=1
('test:', 0.082441350049658649)
Accuracy : 0.6792
max_depth=3, min_child_weight=1
('test:', 0.081659112599510877)
Accuracy : 0.6862
('test:', 4, 0.081894044071160119)
Accuracy : 0.6832
('test:', 5, 0.082105962337208446)
Accuracy : 0.6806
至于子树最小权重
('test:', 3, 2, 0.081647512150225157)
Accuracy : 0.6870
('test:', 3, 3, 0.081694937908247778)
Accuracy : 0.6859
('test:', 3, 4, 0.081658234954534101)
Accuracy : 0.6865
('test:', 4, 2, 0.081860369247522838)
Accuracy : 0.6833
('test:', 4, 3, 0.081896896086544588)
Accuracy : 0.6818
('test:', 4, 4, 0.081914971126082148)
Accuracy : 0.6825
('test:', 5, 2, 0.082174787064929811)
Accuracy : 0.6801
所以最优选择为max_depth=3, min_child_weight=2
目前对数损失结果为：0.081647512150225157
准确度结果为：0.6870

接下来是参数：gamma，默认为0
通常使用的范围为0——0.5之间:
('test:', 0, 0.081647512150225157)
Accuracy : 0.6870
('test:', 2, 0.081669524490561501)
Accuracy : 0.6860
('test:', 0.1, 0.081693451312264007)
Accuracy : 0.6862
('test:', 0.2, 0.081693451312264007)
Accuracy : 0.6862
('test:', 0.3, 0.081683825541184599)
Accuracy : 0.6867
('test:', 0.4, 0.081683825541184599)
Accuracy : 0.6867
('test:', 0.5, 0.081683825541184599)
Accuracy : 0.6867
可见gamma变大并没有能够提升结果，所以选用0

接着是subsample以及colsample_bytree:
默认是1，选取所有特征训练子树，测试0.7——0.9：
('test:', 0.7, 0.7, 0.08161589381677363)
Accuracy : 0.6874
('test:', 0.7, 0.8, 0.081749037053922821)
Accuracy : 0.6850
('test:', 0.7, 0.9, 0.081743130046072879)
Accuracy : 0.6857
('test:', 0.8, 0.7, 0.081682724492239012)
Accuracy : 0.6864
('test:', 0.8, 0.8, 0.081783772025286222)
Accuracy : 0.6841
('test:', 0.8, 0.9, 0.081744046821754401)
Accuracy : 0.6858
('test:', 0.9, 0.7, 0.081748687498675032)
Accuracy : 0.6857
('test:', 0.9, 0.8, 0.081717270934502512)
Accuracy : 0.6860
('test:', 0.9, 0.9, 0.081715755218702918)
Accuracy : 0.6853
选择0.7,0.7的组合
对数损失0.08161589381677363
准确率0.6874

接下来是reg_alpha以及reg_lambda：
正则化的惩罚系数，默认均为reg_alpha=0, reg_lambda=1：
('test:', 0.1, 0.1, 0.081819681177558765)
Accuracy : 0.6840
('test:', 0.1, 1, 0.081754109792519819)
Accuracy : 0.6844
('test:', 1, 0.1, 0.081665086564691611)
Accuracy : 0.6866
('test:', 1, 1, 0.081807432453108092)
Accuracy : 0.6831
('test:' 0, 0, 0.081756997901030859)
Accuracy : 0.6846
默认并不是全0或者全1
目前最优的对数损失0.08161589381677363
准确率0.6874

接下来就剩下learning_rate了，默认为0.1，我们尝试0.01，0.05，0.15, 0.2
('test:', 0.01, 0.085789335498625163)
Accuracy : 0.6655
('test:', 0.05, 0.081690564310606076)
Accuracy : 0.6870
('test:', 0.15, 0.08172966114097667)
Accuracy : 0.6866
('test:', 0.2, 0.082037218582848556)
Accuracy : 0.6796
看到0.05附近有可能有比0.1高的准确率，接下来测试
('test:', 0.04, 0.081754200774442246)
Accuracy : 0.6866
('test:', 0.06, 0.081682863633953232)
Accuracy : 0.6872
('test:', 0.07, 0.081673577188871846)
Accuracy : 0.6874
('test:', 0.08, 0.081708945995339149)
Accuracy : 0.6861
('test:', 0.09, 0.081746684317079751)
Accuracy : 0.6854
可见并没有比0.1默认的更好的参数了，所以选0.1

调参结束了，选好的参数有：
n_estimators=400, max_depth=3, min_child_weight=2, subsample=0.7, colsample_bytree=0.7
"""

"""
默认lightgbm的话
('test:', 0.082012928974854973)
Accuracy : 0.6796

num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20
('test:', 0.081917488577883016)
Accuracy : 0.6807

feature_name=features, categorical_feature=['user_gender_id', ]
('test:', 0.08183320942298164)
Accuracy : 0.6837

('test:', 0.082037503444690019)
Accuracy : 0.6800
比没有feature_name的话要好一点点

可见xgboost还没有调参的结果已经有很好的结果了，需要稍微改进下参数，有希望超过lightbgm
首先是max_depth，默认是3，太小了容易欠拟合，而太大了有可能过拟合，所以我们取2，4,6之间
('test:', 2, 0.082309828229161289)
Accuracy : 0.6789
('test:', 4, 0.081877331373628986)
Accuracy : 0.6836
('test:', 6, 0.081944981987105786)
Accuracy : 0.6813
可见4的时候是要比默认的3要好的，然后试一下5
('test:', 0.081803178246747904)
Accuracy : 0.6840
很明显max_depth为5的时候结果已经比lightbgm好了，不过有可能过拟合
接下来我们以4,5为结果，测试别的参数

learning_rate 默认是0.1,如果比0.1小的话
太小的话，训练速度太慢，而且容易陷入局部最优点。通常是0.0001到0.1之间
选取0.05试一下
('test:', 0.082925687657709801)
Accuracy : 0.6770
结果并不理想
learning_rate 为0.2时
('test:', 0.081918452460409755)
Accuracy : 0.6838
要比默认的0.1结果稍好
0.3结果要差，所以考虑用0.1和0.2作为参数搜索
"""
"""
parameters = {
    'learning_rate': (0.1, 0.2),
    'max_depth': (4, 5),
    'n_estimators': (500, 550, 600),
    'subsample': (0.6, 0.7, 0.8),
    'colsample_bytree': (0.6, 0.7, 0.8)
最佳迭代次数：n_estimators一般在500左右
没有参数：Accuracy : 0.6835
有参数Accuracy : 0.6829
0.3的学习率Accuracy : 0.6846
5的深度Accuracy : 0.6840
"""

"""
调参过程
初始0.1： ('test:', 0.081944846441379712)
学习率变成0.3时('test:', 0.08178744239306944)
max_depth=3, seed=100, n_estimators=140
"""
"""
max_depth [default=6] 树的最大深度，缺省值为6
subsample [default=1] 用于训练模型的子样本占整个样本集合的比例
如果设置为0.5则意味着XGBoost将随机的从整个样本集合中随机的抽取出50%的子样本建立树模型，这能够防止过拟合
lambda [default=0]  L2 正则的惩罚系数
alpha [default=0]  L1 正则的惩罚系数
eta [default=0.3] 为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重
eta通过缩减特征的权重使提升计算过程更加保守
eta –> learning_rate
lambda –> reg_lambda
alpha –> reg_alpha
min_child_weight就是叶子上的最小样本数啦，值越大，算法越保守？？
gamma在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。
这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。
"""
"""
learning_rate=0.1,
        # n_estimators=1000,
        max_depth=3,
        # gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        # objective='binary:logistic',
        # nthread=4,
        seed=27
        
parameters = {
    'n_estimators': (200, 400, 600),
    'learning_rate': (0.2, 0.3),
    'subsample': (0.6, 0.7, 0.8),
    'colsample_bytree': (0.6, 0.7, 0.8)
}

Grid Search尽管比较全面，但是太慢了，尤其是对于老年机，多种组合方式，尽管参数个数并不多，而且取值可能性也只有部分

Random Search是从所有的组合中随机选出k种组合，进行交叉验证，比较它们的表现，从中选出最佳的
虽然Random Search的结果不如Grid Search，但是Random Search通常以很小的代价获得了与Grid Search相媲美的优化结果
所以实际调参中，特别是高维调参中，Random Search更为实用。

learning_rate = 0.05
max_depth = 6
n_estimators = 500
"""
# 调参对于模型准确率的提高有一定的帮助，但这是有限的
# 最重要的还是要通过数据清洗，特征选择，特征融合，模型融合等手段来进行改进！