'''
集成学习（Ensemble Learning）————房屋价格预测分析
随机森林

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
boston = pd.read_csv('D:/desktop/ML/集成学习/housing.csv',header=0, index_col=0) #'header=0'表示第0行为列名

#%%% 查看缺失值
boston.isnull().any()  # all 'False'

#%%                       3.数据准备
'''
因为随机森林的工作原理基于决策树，而决策树对特征的分布和量纲并不敏感，主要关心变量的分布和变量之间的条件概率而非变量本身的取值，所以不需要进行数据标准化。
'''
#%%% 指定特征变量和目标变量
X = boston.drop(columns='MEDV')
y = boston['MEDV']

#%%% 分割训练集与测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)

#%%                       4.模型训练
'随机森林可以用来解决回归和分类问题，sklearn包中分别用sklearn.ensemble里的 RandomForestRegressor()和RandomForestClassifier()，现在用于回归。'
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=0) 
#'n_estimators'为随机森林的决策树数量，'max_depth'为树的最大深度（限制深度可以防止过拟合，但可能降低性能）
rf.fit(X_train, y_train)

#输出R方，看下模型拟合效果
print(rf.score(X_train, y_train))  #0.8786018078350182
print(rf.score(X_test, y_test))    #0.7604643026107618

#%%                       5.模型评价
y_pred = rf.predict(X_test)

# 评估模型，分别用平均绝对误差MAE、均方误差MSE、均方根误差RMSE和平均绝对误差百分比MAPE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
print("MAE", mean_absolute_error(y_test, y_pred))
print("MSE：", mean_squared_error(y_test, y_pred))
print("RMSE：", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAPE：", np.mean(np.abs((y_test-y_pred)/y_test)) * 100)
print("R方：", r2_score(y_test, y_pred))

# =============================================================================
# MAE 2.914318123370007
# MSE： 19.945060004936796
# RMSE： 4.465989252666961
# MAPE： 15.30735907168058
# R方： 0.7604643026107618
# =============================================================================
#明显模型回归效果不佳，要调参

#%%                       6.调参
'''
以下是随机森林中最常调节的参数：

核心参数
n_estimators: 森林中的树的数量。默认值通常是 100，增加此值会提高模型稳定性但增加计算成本。
max_depth: 树的最大深度，较深的树捕捉更多信息，但可能导致过拟合。
min_samples_split: 内部节点再划分所需的最小样本数。默认值是 2，增大此值可以减少过拟合。
min_samples_leaf: 叶节点所需的最小样本数。较大值能减少过拟合。
max_features: 每棵树分裂时考虑的最大特征数。可以是整数、浮点数或特定策略（auto、sqrt、log2）。

性能优化参数
bootstrap：是否启用有放回采样。
max_leaf_nodes：限制最大叶节点数量（减少过拟合）。
n_jobs：并行化核心数，设为 -1 使用所有可用核心。
oob_score：是否使用袋外样本评估模型性能。
'''
#%%%   (1)初始模型搭建,从默认参数或简单的参数配置开始
from sklearn.model_selection import cross_val_score
rf = RandomForestRegressor(random_state=0, n_jobs=-1)

# 交叉验证,评估模型的基准性能。
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Baseline RMSE: {(-scores.mean()) ** 0.5:.2f}")
# Baseline RMSE: 3.65

#%%% 单个参数调整
#%%%% n_setimators
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100, 150, 200, 300, 400, 500, 600]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
#200

#%%%%% max_depth
param_grid = {'max_depth': [3, 5, 7, 10, 20]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
#20

#%%%% max_features
param_grid = {'max_features': ['sqrt', 'log2', None, 0.4, 0.5, 0.6, 0.7]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
# 0.4

#%%%% min_samples_split
param_grid = {'min_samples_split': [2, 3, 4, 5]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
# 3


#%%%%  min_samples_leaf
param_grid = {'min_samples_leaf': [1, 2, 3, 4, 5]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
# 1

#%%%    (2)grid search
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [7, 10, 20,],
    'max_features': [0.3, 0.4, 0.5, 0.6],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=0, n_jobs=-1),
                           param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

#=====================================================================
# Best parameters: {'max_depth': 20, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 200}
#=====================================================================


#%%%    (3)代入
from sklearn.model_selection import cross_val_score
rf_tuned= RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=3, min_samples_leaf=1, max_features=0.5, random_state=0, n_jobs=-1)

# 交叉验证,评估模型的基准性能。
scores_tuned = cross_val_score(rf_tuned, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Tuned RMSE: {(-scores_tuned.mean()) ** 0.5:.2f}")

rf_tuned.fit(X_train, y_train)
#输出R方，看下模型拟合效果
print(rf_tuned.score(X_train, y_train))  #0.9831856505283192
print(rf_tuned.score(X_test, y_test))    #0.7883451752380384


y_pred_tuned = rf_tuned.predict(X_test)
# 评估模型，分别用平均绝对误差MAE、均方误差MSE、均方根误差RMSE和平均绝对误差百分比MAPE
print("MAE", mean_absolute_error(y_test, y_pred_tuned))
print("MSE：", mean_squared_error(y_test, y_pred_tuned))
print("RMSE：", np.sqrt(mean_squared_error(y_test, y_pred_tuned)))
print("MAPE：", np.mean(np.abs((y_test-y_pred_tuned)/y_test)) * 100)
print("R方：", r2_score(y_test, y_pred_tuned))

# =============================================================================
# 500, 20, 3, 1, 0.5
# Tuned RMSE: 3.29
# 0.985184682208376
# 0.7947656035755518
# MAE 2.5170809254670745
# MSE： 17.088944973036458
# RMSE： 4.133877716265499
# MAPE： 12.265905382754346
# R方： 0.7947656035755519
# =============================================================================

# 调参后虽有提升，但是改进不明显啊，随机森林不适合吗

#%%%    (4)特征重要度分析
rf_tuned.feature_importances_
# =============================================================================
# array([0.04884996, 0.00207117, 0.04312868, 0.00157286, 0.02998566,
#        0.3629743 , 0.02229104, 0.03997945, 0.00399712, 0.01936558,
#        0.05484436, 0.01388249, 0.35705733])
# =============================================================================

X.columns
# 按照重要度递增对特征进行排序
feature_names = X.columns
feature_importance = rf_tuned.feature_importances_
indices = np.argsort(feature_importance)[::-1] # 从高到低排序
sorted_features = [feature_names[i] for i in indices]

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance[indices], align='center')
plt.xticks(range(len(feature_importance)), sorted_features, rotation=90)
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# 从条形图可看出，特征‘RM（每栋住宅的平均房间数）’和‘LSTAT（地区中有多少房东属于低收入人群）’对目标变量即房价的重要程度最大，显著超过其它特征对房价的影响力。

# =============================================================================
# sidenote:可以根据重要性分析的结果优化模型：
# 1.删除无关特征：如果某些特征的重要性接近0，可以尝试去掉它们以简化模型。
# 2.重新构造重要特征：对于重要性较高的特征，可以检查是否有非线性关系、交互项等需要进一步建模。
# =============================================================================

#%%                       7.模型预测
prediction = pd.DataFrame(y_pred_tuned, columns=['prediction'])
MEDV = pd.DataFrame(y_test, columns=['MEDV']).reset_index()
comparison = pd.concat([prediction,MEDV],axis=1).drop('index',axis=1)
# 计算误差
comparison['Absolute Error'] = abs(comparison['MEDV'] - comparison['prediction'])
comparison['Relative Error (%)'] = (comparison['Absolute Error'] / comparison['MEDV']) * 100

# 散点图可视化  当散点接近在对角线上，则说明预测值和观察值接近
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_tuned, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs. Predicted')
plt.show()



























