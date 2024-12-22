'''
集成学习（Ensemble Learning）————房屋价格预测分析
XGBoost

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
理解机器学习方法在数据分析中的应用——采用XGBoost方法进行回归分析：
（1）划分训练集与测试集，利用XGBoost算法进行模型训练，回归分析；
（2）模型评价，调整模型参数；
（3）用调参后的模型进行预测，得出的结果与测试集结果进行对比分析，验证XGBoost的建模有效性。

3.方法及工具：sklearn包、XGBoost包

'''
#%%                       1.业务理解
'''
根据住房的历史交易信息对波士顿房价进行预测。
'''
#%%                       2.数据读取
import pandas as pd
boston = pd.read_csv('D:/desktop/ML/集成学习/housing.csv',header=0, index_col=0) #'header=0'表示第0行为列名

#%%                       3.数据准备
'''
因为XGBoost的工作原理基于决策树，而决策树对特征的分布和量纲并不敏感，主要关心变量的分布和变量之间的条件概率而非变量本身的取值，所以不需要进行数据标准化。
'''
#%%% 指定特征变量和目标变量
X = boston.drop(columns='MEDV')
y = boston['MEDV']

#%%% 分割训练集与测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

#%%                       4.模型训练
#用默认参数拟合
from xgboost import XGBRegressor
xgb = XGBRegressor() 
#'n_estimators'为随机森林的决策树数量，'max_depth'为树的最大深度（限制深度可以防止过拟合，但可能降低性能）
xgb.fit(X_train, y_train)

#输出R方，看下模型拟合效果
print(xgb.score(X_train, y_train))  #0.9999977306758547
print(xgb.score(X_test, y_test))    #0.8729544676981909

#%%                       5.模型评价
y_pred = xgb.predict(X_test)

# 评估模型，分别用平均绝对误差MAE、均方误差MSE、均方根误差RMSE和平均绝对误差百分比MAPE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
print("MAE", mean_absolute_error(y_test, y_pred))
print("MSE：", mean_squared_error(y_test, y_pred))
print("RMSE：", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAPE：", np.mean(np.abs((y_test-y_pred)/y_test)) * 100)
print("R方：", r2_score(y_test, y_pred))
# =============================================================================
# MAE 2.110999335113325
# MSE： 9.466555889875238
# RMSE： 3.076776867092451
# MAPE： 10.835497043292195
# R方： 0.8729544676981909
# =============================================================================

#%%                       6.调参
'''
核心参数
n_estimators: 树的数量，更多的树可能提高性能，但会增加计算时间。
learning_rate: 学习率，较小的值需要更多的树来收敛。
max_depth: 树的最大深度，较深的树可以捕获更多细节，但可能过拟合。
subsample: 每棵树使用的样本比例，默认值为 1。
colsample_bytree: 每棵树随机选择的特征比例，默认值为 1。

learning_rate、max_depth、min_child_weight这三个参数为控制过拟合的重要参数，例如，增加max_depth会使模型更复杂，更容易过拟合。
n_estimators作为弱学习器的最大迭代次数，n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合。
在实际调参的过程中，常将n_estimators和learning_rate一起考虑。
除了n_estimators外，colsample_bytree、subsample也是控制速度的重要参数。
reg_alpha、reg_lambda分别表示L1、L2正则化项的影响。
'''
#%%%   (1)初始模型搭建,从默认参数或简单的参数配置开始
from sklearn.model_selection import cross_val_score
xgb = XGBRegressor()

# 交叉验证,评估模型的基准性能。
scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Baseline RMSE: {(-scores.mean()) ** 0.5:.2f}")
# Baseline RMSE: 3.75

#%%%    (2)调节 n_estimators
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.8, 0.9, 1],
    'colsample_bytree': [0.8, 0.9, 1]
}

grid_search = GridSearchCV(xgb, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
# =============================================================================
# Best parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.15, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}
# =============================================================================

#%%%     (3)代入
xgb_tuned = XGBRegressor(n_estimators=200, learning_rate=0.15, max_depth=3, subsample=0.8,colsample_bytree=0.8)
scores_tuned = cross_val_score(xgb_tuned, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Baseline RMSE: {(-scores_tuned.mean()) ** 0.5:.2f}")

xgb_tuned.fit(X_train, y_train)
#输出R方，看下模型拟合效果
print(xgb_tuned.score(X_train, y_train))  #0.9898811466044236
print(xgb_tuned.score(X_test, y_test))    #0.8230225377158351

y_pred_tuned = xgb_tuned.predict(X_test)
# 评估模型，分别用平均绝对误差MAE、均方误差MSE、均方根误差RMSE和平均绝对误差百分比MAPE
print("MAE", mean_absolute_error(y_test, y_pred_tuned))
print("MSE：", mean_squared_error(y_test, y_pred_tuned))
print("RMSE：", np.sqrt(mean_squared_error(y_test, y_pred_tuned)))
print("MAPE：", np.mean(np.abs((y_test-y_pred_tuned)/y_test)) * 100)
print("R方：", r2_score(y_test, y_pred_tuned))

# =============================================================================
# Baseline RMSE: 3.36
# 0.9956000288640228
# 0.8981499753323682
# MAE 1.9338408507798848
# MSE： 7.589160621648867
# RMSE： 2.7548431210595035
# MAPE： 9.998405131613945
# R方： 0.8981499753323682
# =============================================================================


# 模型回归效果有提升













































