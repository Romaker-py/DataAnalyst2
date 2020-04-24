#!/usr/bin/env python
# coding: utf-8


# 【项目】风险预警模型

# #### 目录

# 一、数据获取
# 
# 二、数据理解
# 
# 三、数据预览
# 
# 四、数据处理
# 
# 五、数据建模
# 
# 六、模型应用

# ### 一、数据获取

# _温馨提醒：服务器端需提前安装pymysql包。_

# #### 连接数据库

# In[2]:


import pymysql.cursors
import pandas as pd
import numpy as np

db = pymysql.connect(host='localhost',user='root',password='root',db='tb',charset='utf8',cursorclass=pymysql.cursors.DictCursor)
cursor = db.cursor()

print('连接成功！')


#  - 备注：连接成功

# #### 预览数据库中的数据

# --// `预处理` //-- < 01 /  >

# In[3]:


sql_1 = '''
SELECT * FROM prewarning;
'''
cursor.execute(sql_1)

result = cursor.fetchall()
df_1 = pd.DataFrame(list(result))
#db.close()    # 最后再统一关闭连接

df_1


#  - 备注：特征变量不多，且都是数值型，处理起来相对容易些。

# ### 二、数据理解

# #### 字段业务含义

# In[3]:


sql_2 = '''
SELECT
    COLUMN_NAME AS name
    ,COLUMN_COMMENT AS comments
    ,COLUMN_TYPE AS type
FROM information_schema.COLUMNS
WHERE TABLE_NAME='prewarning'
ORDER BY ORDINAL_POSITION;
'''
cursor.execute(sql_2)

result = cursor.fetchall()
df_type = pd.DataFrame(list(result))

df_type


#  - 备注：主要特征为日期、话量及人力。

# #### 字段类型和DataFrame类型对比

# In[4]:


df_type['dataframeDtype'] = pd.DataFrame(df_1.dtypes).reset_index().iloc[:,1:2]
df_type


#  - 备注：数据由数据库转成DataFrame后，日期的DataFrame类型是object，后续可能需要处理，一次接通率也需要确认。

# ### 三、数据预览

# _【说明】“数据预览”部分，只做缺失值、异常值、重复值检验，至于判断合理处理方式、具体实现部分，统一放到“数据处理”进行处理。_

# #### 缺失值检验：NaN

# In[5]:


df_1.isna().sum()


#  - 备注：无缺失值NaN

# #### 缺失值检验：0

# In[6]:


(df_1 == 0).sum()


#  - 备注：部分类型人员存在0的情况。

# #### 缺失值检验：‘’

# In[7]:


(df_1 == '').sum()


#  - 备注：无空字符串。

# #### 缺失值检验：'0000-00-00 00:00:00'

# In[8]:


(df_1['callDate'] == '0000-00-00 00:00:00').any()


#  - 备注：无空日期。由于上述发现，callDate转成DataFrame后，数据类型为object，待转换后需再次检验是否存在'0000-00-00 00:00:00'

# #### 异常值检验：整体预览

# // 整体预览：

# In[9]:


df_1.describe(include='all')


#  - 备注：呼入总量及呼损最大值需确认是否为异常值；其他暂无发现异常。

# // 数值型变量预览：

# In[10]:


df_1.describe(include=np.number)


#  - 备注：

# // 非数值型预览（string、categorical）：

# In[11]:


df_1.describe(exclude=np.number)


#  - 备注：暂无发现异常。

# #### 异常值检验：字符型变量逐个预览

#  - 备注：无需要检验变量。

# #### 重复值检验

# // 查看整体是否存在重复记录：

# In[12]:


df_1.duplicated().any()


#  - 备注：无重复记录。

# // 进一步查看重复的记录：

#  - 备注：无需查看。

# ### 四、数据处理

# ### （一）特征工程

# _【说明】“特征工程”部分，只做特征的思路准备，具体实现部分，统一放到“生成数据集”进行处理。_

# In[13]:


df_1


# #### 特征选择

# 根据业务场景，使用实际进线的人数、实际产生的话务来建模，最后根据预测的话务和对应的安排人数进行预测。尽管预测话务和安排人数跟实际值有差异，但模型本身具有泛化能力。

#  - 建模和预测需要分开不同的数据。

# #### 特征构造

#  - 上中下旬
#  - 星期
#  - 呼损级别

# #### 特征提取

# _【说明】在“生成数据集”部分进行多重共线性检验（相关性检验）及PCA降维处理。_

# ### （二）预处理

# _【说明】“预处理”部分，判断各类缺失值、异常值、重复值的合理处理方式，具体实现部分，统一放到“生成数据集”进行处理。_

# #### 缺失值处理

# // callDate转日期类型后再次检验空日期：

# In[5]:


(pd.to_datetime(df_1['callDate']) == '0000-00-00 00:00:00').any()


#  - callDate不存在缺失值，无需处理。

# #### 异常值处理

# In[9]:


df_1[df_1['failCall'] >=1500]


#  - 备注：呼损超1500的，可视为系统故障或其他超级大灾，虽转化成3级呼损可以正常入模，但根据实际业务场景，还是需要剔除再建模。

# #### 重复值处理

#  - 备注：无重复值需要处理。

# ### （三）生成数据集

# *该部分生成df_2*

# #### 预处理部分

# // 呼损异常值处理：

# --// `预处理` //-- 

# In[4]:


df_2 = df_1[df_1['failCall'] < 1500]
df_2


# // 话务日期格式转换：

# --// `预处理` //-- 

# In[5]:


df_2['callDate'] = pd.to_datetime(df_2['callDate'])


# #### 特征工程部分

# // 特征选择、特征构造：

#  - 接通率：删除
#  - 上中下旬：新增
#  - 星期：新增
#  - 呼损级别：新增（0-99:1级，100-299:2级，300及以上：3级）

# 删除接通率：

# --// `预处理` //-- 

# In[6]:


del df_2['handleRate']
df_2


# 增加上中下旬字段：

# --// `预处理` //-- 

# In[7]:


def periodOfMonth(a):
    if a.day <= 10:
        result = '上旬'
    elif a.day >= 21:
        result = '下旬'
    else:
        result = '中旬'
    return result

df_2['periodOfMonth'] = df_2.apply(lambda x: periodOfMonth(x['callDate']), axis=1)
df_2['periodOfMonth'].value_counts()


# 增加星期字段：

# --// `预处理` //-- 

# In[8]:


df_2['dayOfWeek'] = df_2['callDate'].map(lambda x: str(x.dayofweek + 1))    # 原周一到周日为：0-6，现改为1-7
df_2['dayOfWeek'].value_counts()


#  - 部分异常值被删除，导致星期的类别数量出现不一致的情况。

# 增加呼损级别：（0-99:1级，100-299:2级，300及以上：3级）

# --// `预处理` //-- 

# In[9]:


def failLevel(a):
    if a <=99:
        level = 1
    elif a >=300:
        level = 3
    else:
        level = 2
    return level

df_2['failLevel'] = df_2.apply(lambda x: failLevel(x['failCall']), axis=1)
df_2['failLevel'].value_counts()


# // 多重共线性检验（相关性检验）：

# In[24]:


df_2.corr().applymap(lambda x: '' if x < 0.8 else x)


#  - 以上相关系数大于0.8的字段对，均不会同时出现。

#  - 实际总人力和实际排班坐席相关系数为0.81，相关性较强。由于实际总人力由各类坐席人数相加得来，故可以删除实际总人力。

# // PCA降维：

#  - 备注：根据相关系数矩阵可知，不需要降维。

# #### 准备建模数据和待预测数据

# // 建模数据集：

# 删除相关字段：

# --// `预处理` //-- 

# In[10]:


df_model = df_2.drop(['historyCall', 'predictCall', 'failCall', 'prepareTotal', 'prepareSchedule', 'prepareUrgent', 'prepareCallback', 'prepareNew', 'realTotal'], axis=1)
df_model


# 删除相关记录：

# --// `预处理` //-- 

# In[12]:


df_model = df_model[df_model['callDate'] < '2019-02']
df_model


# // 预测数据集：

# 删除相关记录：

# --// `预处理` //-- 

# In[13]:


df_predict = df_2[df_2['callDate'] >= '2019-02']
df_predict


# 删除相关字段：

# --// `预处理` //-- 

# In[14]:


df_predict = df_predict.drop(['historyCall', 'realCall', 'failCall', 'prepareTotal', 'realTotal', 'realSchedule', 'realUrgent', 'realCallback', 'realNew'], axis=1)
df_predict


# ### （四）入模处理

# *该部分生成df_3*

# #### 连续变量分箱

#  - 备注：除了呼损等级，其他暂时无需分箱处理。

# #### 变量数值化

# // 去掉字符型变量（非特征部分）：

# // 删除：callDate

# --// `预处理` //-- 

# In[15]:


df_3_1 = df_model.drop('callDate', axis=1)
df_3_2 = df_predict.drop('callDate', axis=1)


# // 生成哑变量：

# --// `预处理` //-- 

# In[16]:


df_3_1 = pd.get_dummies(df_3_1)
df_3_2 = pd.get_dummies(df_3_2)
df_3_1


#  - 备注：

# // 获取变量名称（后续备用）：

# In[17]:


feature_names = df_3_1.drop('failLevel', axis=1).columns    # 特征字段名称不含y字段
feature_names


# #### 样本均衡

# // 样本均衡检验：

# In[110]:


df_3_1['failLevel'].value_counts()


# In[54]:


df_3_1['failLevel'].value_counts(normalize=True)


#  - 样本均衡，无须处理。

# // 样本均衡处理：

#  - 样本均衡，无须处理。

# #### 标准化数据

#  - 使用XGBoost可不用标准化

# #### 各数据集名称及范围汇总

# - df_1:从数据库中读取的数据集。* rows × * columns
# 
# 
# - df_2:完成数据预处理、特征工程之后的数据集。* rows × * columns
# 
# 
# - df_model:建模数据集（未进行入模处理）。* rows × * columns
# 
# 
# - df_predict:预测数据集（未进行入模处理）。* rows × * columns
# 
# 
# - df_3:删除非特征部分的字符型字段。* rows × * columns
# 
# 
# - df_3_3:
# 
# 
# - df_4:生成哑变量之后的数据集。* rows × * columns
# 
# 
# - df_5_1:保留有评价内容的记录。* rows × * columns
# - df_5_2:无评价内容的记录。* rows × * columns
# 
# 
# - X:df_5_1的X部分。* rows ×* columns
# - y:df_5_1的y部分。* rows × * columns
# 
# 
# - X_train:训练集有*个样本
# - X_test:测试集有*个
# 
# 
# - X_resampled:* rows × * columns
# - y_resampled:* rows × * columns
# 
# 
# - X_train_res:SMOTE后训练集有*个样本
# - X_test_res:SMOTE后测试集有*个样本
# 
# 
# - df_predict_result:

# ### 五、数据建模

# #### 建模数据：切分训练集和测试集

# // 准备X, y：

# --// `预处理` //-- 

# In[18]:


X = df_3_1.drop('failLevel', axis=1)
y = df_3_1['failLevel']


# // 开始切分：

# --// `预处理` //-- 

# In[19]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 显示切分结果
print('训练集有%d个样本.' % X_train.shape[0])
print('测试集有%d个样本.' % X_test.shape[0])


# #### 建模调参

# // 网格搜索：

# --// `预处理` //-- 

# In[29]:


from datetime import datetime

# 开始计时
start_time = datetime.now()
start_time_str = start_time.strftime("%Y{}%m{}%d{} %H{}%M{}%S{}").format('年','月','日','时','分','秒')
print('开始时间：%s' %start_time_str)


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

xgbc = XGBClassifier(random_state=42, n_jobs=-1)

parameters = {
    'max_depth': [7, 10, 12],
    'learning_rate': [0.003, 0.005, 0.01],
    'n_estimators': [800, 1000, 2500],
    'min_child_weight': [0, 5, 20]
    'max_delta_step': [0, 0.6, 2],
    'subsample': [0.6, 0.8, 0.95],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [0.2, 0.6, 1],
    'scale_pos_weight': [0.2, 0.6, 1]
}
scorings = ['accuracy']
for scoring in scorings:
    print('===========================================================================')
    print('此次调参的scoring为：%s' % scoring)
    print()
    clf = GridSearchCV(xgbc, parameters, scoring=scoring, cv=3)
    clf.fit(X_train, y_train)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    params = clf.cv_results_['params']
    print('【交叉验证】过程如下：')
    for mean, std, param in zip(means, stds, params):
        print('%s得分为 %0.3f(+-%0.3f)时，参数为：%r' % (scoring, mean, std * 2, param))
    print()
    print('【训练集】最佳参数为：%r' % clf.best_params_)
    print('【训练集】最佳参数得分为：%0.3f' % clf.best_score_)
    print()
    print('【测试集】%s得分为：%0.3f' % (scoring, clf.score(X_test, y_test)))    # 直接获取测试集的得分
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    reports = classification_report(y_true, y_pred)
    print('【测试集中实际值与预测值对比】模型报告如下：')
    print(reports)
    
    
# 结束计时
end_time = datetime.now()
end_time_str = end_time.strftime("%Y{}%m{}%d{} %H{}%M{}%S{}").format('年','月','日','时','分','秒')
total_time = end_time - start_time

# 提醒总耗时
print('【耗时情况】')
print('开始时间：%s' %start_time_str)
print('结束时间：%s' %end_time_str)
print('总耗时为：%s' %total_time)


# // SMOTE

# In[191]:


get_ipython().run_line_magic('pinfo', 'SMOTE')


# In[190]:


y.value_counts()


# In[192]:


from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE(sampling_strategy='not minority', random_state=42).fit_resample(X, y)    # sampling_strategy='auto'

print('X_resampled为：%d rows × %d columns' % (X_resampled.shape[0], X_resampled.shape[1]))
print()
print('y_resampled各类样本量为：\n %r' % y_resampled.value_counts())
print()
print('y_resampled各类样本占比为：\n %r' % y_resampled.value_counts(normalize=True))


# // 重新切分训练集合测试集：

# In[193]:


from sklearn.model_selection import train_test_split

X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_resampled, y_resampled, test_size=0.33, random_state=42)

# 显示切分结果
print('训练集有%d个样本.' % X_train_res.shape[0])
print('测试集有%d个样本.' % X_test_res.shape[0])


# // 最优参数建模：

# In[75]:


get_ipython().run_line_magic('pinfo', 'XGBClassifier')


# In[30]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report

clf = XGBClassifier(learning_rate=0.05, max_depth=10, n_estimators=1000, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print('【测试集】得分为：%0.3f' % clf.score(X_test, y_test))    # 直接获取测试集的得分
print()
y_true, y_pred = y_test, clf.predict(X_test)
reports = classification_report(y_true, y_pred)
print('【测试集中实际值与预测值对比】模型报告如下：')
print(reports)


#  - 备注：3级呼损召回率较低，可尝试其他算法。

# // XGBClassifier含SMOTE：【最终采用模型】

# In[221]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report

clf = XGBClassifier(learning_rate=0.01, max_depth=20, n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_train_res, y_train_res)
print('【测试集】得分为：%0.3f' % clf.score(X_test_res, y_test_res))    # 直接获取测试集的得分
print()
y_true, y_pred = y_test, clf.predict(X_test)
reports = classification_report(y_true, y_pred)
print('【测试集中实际值与预测值对比】模型报告如下：')
print(reports)


# // GradientBoostClassifier：

# In[152]:


get_ipython().run_line_magic('pinfo', 'GradientBoostingClassifier')


# In[82]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, random_state=42)
clf.fit(X_train, y_train)
print('【测试集】得分为：%0.3f' % clf.score(X_test, y_test))    # 直接获取测试集的得分
print()
y_true, y_pred = y_test, clf.predict(X_test)
reports = classification_report(y_true, y_pred)
print('【测试集中实际值与预测值对比】模型报告如下：')
print(reports)


#  - 3级呼损召回率仍然不够理想。

# // GradientBoostingClassifier含SMOTE：

# In[201]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=300, random_state=42)
clf.fit(X_train_res, y_train_res)
print('【测试集】得分为：%0.3f' % clf.score(X_test_res, y_test_res))    # 直接获取测试集的得分
print()
y_true, y_pred = y_test, clf.predict(X_test)
reports = classification_report(y_true, y_pred)
print('【测试集中实际值与预测值对比】模型报告如下：')
print(reports)


# // RandomForest：

# In[65]:


get_ipython().run_line_magic('pinfo', 'RandomForestClassifier')


# In[83]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

clf = RandomForestClassifier(n_estimators=800, class_weight={1:1, 2:1, 3:10}, random_state=42)
clf.fit(X_train, y_train)
print('【测试集】得分为：%0.3f' % clf.score(X_test, y_test))    # 直接获取测试集的得分
print()
y_true, y_pred = y_test, clf.predict(X_test)
reports = classification_report(y_true, y_pred)
print('【测试集中实际值与预测值对比】模型报告如下：')
print(reports)


# // RandomForestClassifier含SMOTE：

# In[214]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

clf = RandomForestClassifier(n_estimators=800, class_weight={1:5, 2:3, 3:1}, random_state=42)
clf.fit(X_train_res, y_train_res)
print('【测试集】得分为：%0.3f' % clf.score(X_test_res, y_test_res))    # 直接获取测试集的得分
print()
y_true, y_pred = y_test, clf.predict(X_test)
reports = classification_report(y_true, y_pred)
print('【测试集中实际值与预测值对比】模型报告如下：')
print(reports)


# #### 模型属性

# // 特征重要性：

# In[37]:


from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

result = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)    # 跑不动时可减少n_repeates次数
sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.barh(tree_indices, clf.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(feature_names[tree_importance_sorted_idx])
ax1.set_yticks(tree_indices)
ax1.set_ylim((0, len(clf.feature_importances_)))
ax2.boxplot(result.importances[sorted_idx].T, vert=False, labels=feature_names[sorted_idx])
fig.tight_layout()

#plt.savefig(r"D:\Jupyter\Data\importances2.png",dpi=800)
plt.show()


#  - 备注：星期及应急人数、话务量都很重要。

# #### 模型业务理解

#  - 备注：模型结果跟实际业务相符，可以落地。

# #### 模型效果评估：指标

#  - 备注：详见建模评估指标

# #### 模型效果评估：画图

# // 混淆矩阵：

# --// `预处理` //-- 

# In[195]:


from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))
plot_confusion_matrix(clf, X_test, y_test, ax=ax, cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.show()


# // ROC曲线图：

#  - 备注：多分类问题无ROC曲线。

# ### 六、模型应用

# #### 数据风险评估

#  - 数据预测风险：建模用的是实际发生的数据，预测用的是预测话量及安排人力，故预测结果受话务预测及人力结果安排影响。

#  - 预测结果为低级别呼损时，有可能存在系统故障，故应急举措仍然需要到位。

# #### 利用模型进行预测

# // 生成预测数据集：

# --// `预处理` //-- 

# In[89]:


df_3_3 = df_3_2.drop('failLevel', axis=1)
df_3_3


# 需要在上旬之后插入下旬（periodOfMonth_下旬），且值为0：

# --// `预处理` //-- 

# In[120]:


X_resampled.columns


# In[116]:


df_3_3.insert(6, 'periodOfMonth_下旬', 0)
df_3_3


# 修改待预测数据集字段名称：

# In[121]:


df_3_3.columns = list(X_resampled.columns)
df_3_3


# // 模型预测：

# --// `预处理` //-- 

# In[222]:


predict_class = clf.predict(df_3_3)
predict_class


# // 预测结果：

# --// `预处理` //-- 

# In[223]:


df_predict_result = pd.concat([df_predict.reset_index().drop('index', axis=1),
                               pd.DataFrame(predict_class, columns=['predictClass'])], axis=1)
df_predict_result


# 增加预测结果对比：

# --// `预处理` //-- 

# In[224]:


def resultCompare(a, b):
    if a == b:
        result = '预测正确'
    elif a > b:
        result = '实际更严重'
    else:
        result = '预测过于严重'
    return result

df_predict_result['predictResult'] = df_predict_result.apply(lambda x: resultCompare(x['failLevel'], x['predictClass']), axis=1)
df_predict_result


# In[225]:


df_predict_result['predictResult'].value_counts()


# In[226]:


df_predict_result.groupby(['failLevel', 'predictClass', 'predictResult'])['predictResult'].count()


# #### 输出模型结果

# // 保存到本地：

# --// `预处理` //-- 

# In[131]:


outputpath='D:/Jupyter/Data/prewarning_result.csv'
df_predict_result.to_csv(outputpath, sep=',', index=False, header=True, encoding='gb18030')

print("导出成功！")


# // 存入数据库：

#  - 无须存入数据库。

# #### 结束：关闭数据库连接

# In[38]:


db.close()

print('已关闭数据库连接！')

