#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : gupeng_feature.py
# @Author: Feng
# @Date  : 2019/8/8
# @Desc  :
import pandas as pd
import numpy as np
from tqdm import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import gc
from sklearn import preprocessing
# train_path='C:\\Users\\gupeng\\Desktop\\prediction-of-students-exam-scores\\data\\train_s1\\'
# test_path='C:\\Users\\gupeng\\Desktop\\prediction-of-students-exam-scores\\data\\test_s1\\'
# out_path = 'C:\\Users\\gupeng\\Desktop\\prediction-of-students-exam-scores\\out\\'
train_path = 'F:/数据集/1908天翼/初赛/train_s1/'
test_path = 'F:/数据集/1908天翼/初赛//test_s1/'
out_path = 'F:/项目相关/1908天翼/out/'
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
exams = pd.DataFrame({'exam_id':[],"level_1":[],0:[],"course":[]})
for i in range(1,9):
    s = 0
    exec("""s = course{}_exams.set_index("exam_id").stack().to_frame().reset_index()""".format(i))
    exam_list = s['exam_id'].drop_duplicates().values
    s_temp = pd.DataFrame({'exam_id':exam_list,'exam_time':range(len(exam_list))})
    s =s.merge(s_temp,how = 'left',on = 'exam_id')
    s["course"] = "course{}".format(i)
    exams = exams.append(s)
for i in range(1,9):
    exec('''s{} = course{}_exams.stack().to_frame().reset_index().loc[course{}_exams.stack().to_frame().reset_index().level_1.str.contains('K:')]'''.format(i,i,i))
    exec('''s{}.columns = ['row','col','hardvalue']'''.format(i))
    exec('''hardness{} = s{}.groupby('row')['hardvalue'].apply(lambda x : np.dot(x,all_knowledge.loc[all_knowledge.course == 'course{}']['complexity']))'''.format(i,i,i))
    exec('''course{}_exams = pd.concat([course{}_exams,hardness{}],axis=1)'''.format(i,i,i))
result1 = course1_exams.copy(deep = True)
for i in range(2,9):
    exec('''result1 = pd.concat([result1,course{}_exams])'''.format(i))
    exec('''del hardness{}'''.format(i-1))
    exec('''del s{}'''.format(i-1))
    exec('''del course{}_exams'''.format(i - 1))

result1 = result1[['exam_id', 'hardvalue']].copy(deep=True).reset_index()
train['is_train'] = 1
test['is_train'] = 0
test.rename(columns = {'pred':'score'},inplace=True)
data_all = pd.concat([train,test])
data_all = data_all.merge(result1[['exam_id', 'hardvalue']],on = 'exam_id',how = 'left')
exams = exams.rename(columns = {0:"score"})
exams = exams.merge(all_knowledge,how = 'left',left_on = ['course','level_1'],right_on=['course','knowledge_point'])
exams = exams.merge(course,how = 'left',on = 'course')
# exams = exams.merge(result1[['exam_id','hardvalue']],on = 'exam_id',how = 'left')

data_all.columns = ['student_id','train_course','exam_id','train_score','is_train','hardvalue']
temp = data_all.merge(exams,on = 'exam_id',how = 'left')
gc.collect()
#使用数字替换类别
replace = {'course1':1,'course2':2,'course3':3,'course4':4,'course5':5,'course6':6,'course7':7,'course8':8}
temp['train_course'] = temp['train_course'].replace(replace)
temp['course'] = temp['course'].replace(replace)

le = preprocessing.LabelEncoder()
temp["level_1"] = le.fit_transform(temp["level_1"])
alert_type = {index: label for index, label in enumerate(le.classes_)}
alert_type = {v:k for k,v in alert_type.items()}
temp['knowledge_point'] = temp['knowledge_point'].replace(alert_type)
le = preprocessing.LabelEncoder()
temp["section"] = le.fit_transform(temp["section"])
section_type = {index: label for index, label in enumerate(le.classes_)}
le = preprocessing.LabelEncoder()
temp["category"] = le.fit_transform(temp["category"])
category_type = {index: label for index, label in enumerate(le.classes_)}
replace = {'course_class1':1,'course_class2':2}
temp['course_class'] = temp['course_class'].replace(replace)

#计算难度
# 当前考试
#1. 知识点所属section
def countSection(df):
    tmp = df.value_counts()
train = train.merge(temp.groupby(['student_id','exam_id','is_train'])['section'].agg({'section_part':'count'}),on = ['student_id','exam_id','is_trian'],how = 'left')

# 2.知识点所属的category
# train = train.merge(temp.groupby(['student_id','exam_id'])['section'].agg({'section_part':'value_counts'}),on = ['student_id','exam_id'],how = 'left')
# 3. 难度值
# train = train.merge(temp.groupby(['student_id','exam_id'])['section'].agg({'hardness':'max'}),on = ['student_id','exam_id'],how = 'left')
#历史考试

#学生维度


#同一个知识点学生的掌握程度