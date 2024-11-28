'''
集成学习（Ensemble Learning）————房屋价格预测分析

1.数据及分析对象
'housing.csv'，是scikit-learn的内置数据集————波士顿房价数据集（Boston House Prices Dataset），数据内容来自卡内基梅隆大学的StatLib Library库。该数据集有506行，14个属性（列）。主要属性如下：
（1）CRIM: 城镇人均犯罪率（per capita crime rate by town）。
（2）ZN: 超过 25,000 平方英尺的住宅用地的比例（proportion of residential land zoned for lots over 25,000 sq.ft.）。
（3）INDUS: 城镇中非住宅用地所占比例（proportion of non-retail business acres per town）。
（4）CHAS: Charles River 虚拟变量(= 1，如果道路沿河而行; 否则为0)（Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)） 。
（5）NOX: 一氧化氮浓度（百万分之几）（nitric oxides concentration (parts per 10 million)）。
（6）RM: 每栋住宅的平均房间数（average number of rooms per dwelling）。
（7）AGE: 1940 年以前建成的自住单位的比例（proportion of owner-occupied units built prior to 1940）。
（8）DIS: 距离5个波士顿的就业中心的加权距离（weighted distances to five Boston employment centres）。
（9）RAD: 距离高速公路的便利指数（index of accessibility to radial highways）。
（10）TAX: 每一万美元的不动产税率（full-value property-tax rate per $10,000）。
（11）PTRATIO: 城镇中的教师学生比例（pupil-teacher ratio by town）。
（12）B: 城镇中的黑人比例（1000(Bk - 0.63)^2 where Bk is the proportion of blacks   by town）。
（13）LSTAT: 地区中有多少房东属于低收入人群（% lower status of the population）。
（14）MEDV: 自住房屋房价中位数（Median value of owner-occupied homes in $1000's）。


2.目的及分析任务
理解机器学习方法在数据分析中的应用——采用随机森林方法进行回归分析：
（1）划分训练集与测试集，利用随机森林算法进行模型训练，回归分析；
（2）根据随机森林模型预测测试集的“MEDV”值；
（3）将随机森林模型给出的“MEDV”预测值与测试集自带的实际“MEDV”值进行对比分析，验证随机森林建模的有效性。

3.方法及工具：sklearn

'''
#%%                       1.业务理解
'''
根据住房的历史交易信息对波士顿房价进行预测。
'''
#%%                       2.数据读取
import pandas as pd
boston = pd.read_csv('D:/desktop/ML/集成学习/housing.csv',header=0) #'header=0'表示第0行为列名

#%%% 查看缺失值
boston.isnull().any()  # all 'False'

#%%                       3.数据准备
'''
因为随机森林作为树形模型的一种，主要关心变量的分布和变量之间的条件概率而非变量本身的取值，所以不需要进行数据标准化。
'''






#%%                       4.模型训练


#%%                       5.模型评价
































