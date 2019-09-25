# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 3train.py
# @time  : 2019/8/9
"""
文件说明：lightgbm模型
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

sub_path = 'F:/数据集/1908天翼/初赛/'
feature_path = 'F:/项目相关/1908天翼/feature/'

data_all = pd.read_csv(open(feature_path+"data_plus_score0809.csv",encoding="utf8"))
# 方案1：去除缺失较多的行
# data_all = data_all.loc[np.isnan(data_all["category_rank"])==0] #去除阶段性考试
data_all = data_all.loc[np.isnan(data_all["his_category_mean"])==0] #去除his_category空值
data_all = data_all.loc[np.isnan(data_all["score_mean05"])==0] #去除05空值
data_all = data_all.loc[np.isnan(data_all["score_mean07"])==0] #去除07空值
# data_all = data_all.loc[np.isnan(data_all["score_mean10"])==0] #去除10空值

data_all = data_all.reset_index(drop=True)

data_all_temp = data_all.copy()
le = LabelEncoder()
data_all_temp["le_course"] = le.fit_transform(data_all_temp["course"])
le = LabelEncoder()
data_all_temp["le_exam_id"] = le.fit_transform(data_all_temp["exam_id"])
le = LabelEncoder()
data_all_temp["le_course_class"] = le.fit_transform(data_all_temp["course_class"])
le = LabelEncoder()
data_all_temp["le_category"] = le.fit_transform(data_all_temp["category"])

y = "score"

features = ["gender","exam_time","knowledge_cnt","category_rank"]\
           +["hardvalue_sum","hardvalue_max","hardvalue_rank"] \
           + [i for i in data_all_temp.columns if "his_category_" in i] \
           +["le_course","le_course_class","le_category"] \
           + [i for i in data_all_temp.columns if "05" in i] \
           + [i for i in data_all_temp.columns if "07" in i] \
           + [i for i in data_all_temp.columns if "last" in i] \
           # + [i for i in data_all_temp.columns if "10" in i] \

"""
5.134/8.013
hardvalue_+his_category_   去空 5.08
hardvalue_+his_category_+categorical   去空 5.07
hardvalue_+his_category_+categorical+05+07   去空 4.725/7.874
hardvalue_+his_category_+categorical+05+07+last123   去空 4.719/7.629
hardvalue_+his_category_+categorical+05+07+last123   去空 调参 4.714/7.655
"""

# categorical_feats = []
categorical_feats = ["le_course","le_course_class","le_category","gender"]
train = data_all_temp[data_all.is_train==1]
test = data_all_temp[data_all.is_train==0]

folds = KFold(n_splits=5, shuffle=True, random_state=2333)

oof = np.zeros(len(train))
predictions = np.zeros(len(test))
train_x = train[features]
test_x = test[features]
train_y = train[y]

param = {'objective': 'regression',
         'num_leaves': 2**5, #2**5
         'min_data_in_leaf': 25,#25
         'max_depth': 5,  #5
         'learning_rate': 0.02, #
         'lambda_l1': 0.13,#0.13
         "boosting": "gbdt",
         "feature_fraction": 0.7,#0.7线上最优-0.5线下最优
         'bagging_freq': 8,#8
         "bagging_fraction": 0.9, #0.9
         "metric": 'rmse',
         "verbosity": -1,
         "random_state": 2333,
         "num_threads" : 50}
# lgb
model = "lgb"
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x.values, train_y.values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train_x.iloc[trn_idx][features],
                           label=train_y.iloc[trn_idx],
                           categorical_feature=categorical_feats
                           )
    val_data = lgb.Dataset(train_x.iloc[val_idx][features],
                           label=train_y.iloc[val_idx],
                           categorical_feature=categorical_feats
                           )

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=50,early_stopping_rounds=100)
    #n*6矩阵
    oof[val_idx] = clf.predict(train_x.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits
feature_importance = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
ascending=False)

from sklearn import metrics
from math import log10
rmse = round((metrics.mean_squared_error(oof, train_y) ** 0.5),3)
print("CV score: {:<8.5f}".format(10*log10(rmse)))

#提交答案
test["pred"] = predictions

sub =pd.read_csv(open(sub_path+"submission_s1_sample.csv",encoding='utf8'))
del sub["pred"]
sub = sub.merge(test[["student_id", "course", "exam_id","pred"]],how='left',on=["student_id", "course", "exam_id"])
sub.to_csv("F:/项目相关/1908天翼/out/"+"result{}.csv".format(rmse),index=False)