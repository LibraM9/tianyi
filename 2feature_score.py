# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 2feature.py
# @time  : 2019/8/8
"""
文件说明：feature 按分数聚合
"""
import pandas as pd
import numpy as np
from tqdm import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb


train_path = 'F:/数据集/1908天翼/初赛/train_s1/'
test_path = 'F:/数据集/1908天翼/初赛//test_s1/'
feature_path = 'F:/项目相关/1908天翼/feature/'
all_knowledge = pd.read_csv(open(train_path + 'all_knowledge.csv',encoding='utf8'))
course1_exams = pd.read_csv(open(train_path + 'course1_exams.csv',encoding='utf8'))#85,369 考点依次向后，最后为综合性考试
course2_exams = pd.read_csv(open(train_path + 'course2_exams.csv',encoding='utf8')) #94,368
course3_exams = pd.read_csv(open(train_path + 'course3_exams.csv',encoding='utf8'))#70，263
course4_exams = pd.read_csv(open(train_path + 'course4_exams.csv',encoding='utf8'))#68,220
course5_exams = pd.read_csv(open(train_path + 'course5_exams.csv',encoding='utf8'))#90,336
course6_exams = pd.read_csv(open(train_path + 'course6_exams.csv',encoding='utf8'))#54,147
course7_exams = pd.read_csv(open(train_path + 'course7_exams.csv',encoding='utf8'))#78,317
course8_exams = pd.read_csv(open(train_path + 'course8_exams.csv',encoding='utf8'))#78,224
train = pd.read_csv(open(train_path + 'exam_score.csv',encoding='utf8')) #共8门课 65500,4
student = pd.read_csv(open(train_path + 'student.csv',encoding='utf8'))#500,2
course = pd.read_csv(open(train_path + 'course.csv',encoding='utf8'))#8,2
test = pd.read_csv(open(test_path + 'submission_s1.csv',encoding='utf8'))#8000,5

exams = pd.DataFrame({"exam_id":[],"level_1":[],0:[],"course":[]})
for i in range(1,9):
    s=0
    exec("""s = course{}_exams.set_index("exam_id").stack().to_frame().reset_index()""".format(i))
    exam_list = s["exam_id"].drop_duplicates().values
    s_temp = pd.DataFrame({"exam_id":exam_list,"exam_time":range(len(exam_list))})
    s = s.merge(s_temp,how='left',on="exam_id")
    s["course"] = "course{}".format(i)
    exams = exams.append(s)

exams = exams.rename(columns={0:"score"})
# 合并所有和考试相关的信息
exams = exams.merge(all_knowledge,how='left',left_on=["course","level_1"],right_on=["course","knowledge_point"])
exams = exams.merge(course,how='left',on='course')
exams["hardvalue"] = exams["score"]*exams["complexity"]
exams["rank"] = exams.groupby("exam_id")["score"].rank(ascending=False,method='first')
exams["score"] = exams["score"].replace(0,np.nan)

#将所有考试信息归到一维
agg = {
    "score":["count"],#考了几个知识点
    "hardvalue":["sum","max"] #难度值、难度最高的知识点有多高
}
exams_group = exams.groupby(["exam_id","exam_time","course","course_class"],as_index=False).agg(agg) #617,2
exams_group.columns=["exam_id","exam_time","course","course_class","knowledge_cnt","hardvalue_sum","hardvalue_max"]
exams_group = exams_group.merge(exams.loc[exams["rank"]==1][["exam_id","section","category"]],how="left",on=["exam_id"])#本次考试的知识点所属领域
#知识点过多的是阶段性考试(超过12个？),每章节最多9个知识点
"""
5    159
4    159
3     79
6     73
7     11
8      5
2      2
9      1
"""
exams_group = exams_group.sort_values(by=["course_class","course","exam_time"]).reset_index(drop=True)

#学生维度
train["is_train"]=1
test["is_train"]=0
test.rename(columns = {'pred':'score'},inplace=True)
data_all = train.append(test)
data_all = data_all.merge(student,how='left',on='student_id')
data_all = data_all.merge(exams_group,how='left',on=["exam_id","course"])
data_all["effotvalue"] = data_all["score"]/data_all["hardvalue_sum"]
bin_labels = [0,1,2,3,4,5,6,7]
data_all["hardvalue_rank"]=pd.cut(data_all["hardvalue_sum"], 8, right=True, labels=bin_labels, retbins=False, precision=3, include_lowest=False)
#当前章节第几次考试,测试集中3/16次考试更换了知识点,category_rank为nan则说明为阶段性考试
data_tmp = data_all.sort_values(by=["student_id","course","section","knowledge_cnt"])
data_tmp = data_tmp.drop_duplicates(["student_id","course","section"])
data_tmp["category_rank"] = data_tmp.groupby(["student_id","course","category"])["exam_time"].rank(method="first")
data_all = data_all.merge(data_tmp[["student_id","course","exam_time","category_rank"]],how='left',on=["student_id","course","exam_time"])
#当前数据输出
data_all.to_csv(feature_path + "data0809.csv", index=False)

#聚合
exam_his={"course1":18,"course2":21,"course3":14,"course4":14,
          "course5":22,"course6":10,"course7":16,"course8":16}
data_all["exam_time_tmp"] = data_all["course"].replace(exam_his)
data_all["exam_time05"] =data_all["exam_time"]-(0.5*data_all["exam_time_tmp"]).astype(int)
data_all["exam_time07"] =data_all["exam_time"]-(0.7*data_all["exam_time_tmp"]).astype(int)
data_all["exam_time10"] =data_all["exam_time"]-(data_all["exam_time_tmp"]).astype(int)

#对历史努力值进行统计
agg = {
    "score":["mean","median","std","max","min"],
    "category_his":["count"]
}
for i in ["05","07","10"]:
    print(i)
    data_union = data_all[["student_id","course","exam_id","exam_time","exam_time{}".format(i),"category"]]\
        .merge(data_all[["student_id","course","exam_time","category","score","hardvalue_sum"]].rename(columns={"exam_time":"exam_time_his","category":"category_his"}),
               how='left',on=["student_id","course"])
    data_union = data_union.loc[(data_union["exam_time"+i]<data_union["exam_time_his"])&(data_union["exam_time_his"]<data_union["exam_time"])]
    data_union["score"] = data_union["score"].replace(0,np.nan) #将缺考的努力值置为空
    data_tmp = data_union.groupby(["student_id","course","exam_id"],as_index=False).agg(agg)
    data_tmp.columns = [j[0]+"_"+j[1]+i for j in data_tmp.columns ]
    data_tmp.columns = ["student_id","course","exam_id"]+list(data_tmp.columns[3:])
    #计算应该对几条数据做聚合，筛选数据不足的行,如：第二次考试仅可以获取第一次考试的信息，信息量少于0.5/0.7/1，故舍弃
    data_tmp["cnt"] = (data_tmp["course"].replace(exam_his)*(int(i[0])+0.1*int(i[1]))).astype(int)-1
    data_tmp = data_tmp.loc[data_tmp["category_his_count"+i]==data_tmp["cnt"]]
    del data_tmp["cnt"]
    del data_tmp["category_his_count"+i]
    data_all = data_all.merge(data_tmp, how='left', on=["student_id", "course", "exam_id"])
    if i=="10":
        #筛选近期考点属于同一大类的考试进行统计
        data_tmp2 = data_union.loc[data_union["category"]==data_union["category_his"]]
        data_tmp2 = data_tmp2.groupby(["student_id", "course", "exam_id"], as_index=False)["score"].agg(["mean","median","std","max","min"])
        data_tmp2 = data_tmp2.reset_index()
        data_tmp2.columns = list(data_tmp2.columns[:3])+["his_category_"+j for j in data_tmp2.columns[3:]]
        data_all = data_all.merge(data_tmp2,how='left',on=["student_id", "course", "exam_id"])
        #近3次同类考试的分数和难度值
        data_tmp2 = data_union.loc[data_union["category"] == data_union["category_his"]]
        data_tmp2["his_rank"]=data_tmp2.groupby(["student_id", "course", "exam_id"])["exam_time_his"].rank(ascending=False,method='first')
        for k in [1,2,3]:
            data_tmp3 = data_tmp2.loc[data_tmp2.his_rank==k]
            data_tmp3 = data_tmp3[["student_id", "course", "exam_id","score","hardvalue_sum"]].rename(columns={"score":"score_last{}".format(k),"hardvalue_sum":"hardvalue_last{}".format(k)})
            data_all=data_all.merge(data_tmp3,how='left',on=["student_id", "course", "exam_id"])
del data_all["exam_time_tmp"],data_all["exam_time05"],data_all["exam_time07"],data_all["exam_time10"]
data_all.to_csv(feature_path+"data_plus_score0809.csv", index=False)


