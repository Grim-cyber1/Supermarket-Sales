#import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
!gdown https://drive.google.com/uc?id=1R-K2mRfPkstCB7EJTyNDC-bNJv2xm2i2
df = pd.read_csv("Supermarket_sales - Sheet1.csv")
df.head()
df.info()
df.head()
df.head()
df[df['Gender'] == 'Male']['Product line'].value_counts()
df[df['Gender'] == 'Female']['Product line'].value_counts()
df.groupby(['Gender', 'Product line', 'Customer type']).agg({'Total':['median', 'mean','max', 'min'], 'Unit price':['median', 'mean','max', 'min']}, numeric_only=True)
df['DateTime'] = df['Date'] + ' ' + df['Time']
df.head()
df['DateTime'] = pd.to_datetime(df['DateTime'])

df['year'] = df['DateTime'].dt.year
df['month'] = df['DateTime'].dt.month
df['week'] = df['DateTime'].dt.weekday
df['week year'] = df['DateTime'].dt.isocalendar().week
df['day'] = df['DateTime'].dt.day
df['hour'] = df['DateTime'].dt.hour
df['minute'] = df['DateTime'].dt.minute
df.drop(columns=['Date', 'Time'], inplace=True)

plt.figure(figsize=(12, 3), dpi=1000)
plt.title('Иллюстрация стоимости', color='red', fontsize=18)
plt.plot(df['Total'], color='black', linestyle='-', linewidth=0.8, label='Итоговая стоимость')
plt.xlabel('Точки наблюдения', color='grey', fontsize=12)
plt.ylabel('Итоговая стоимость', color='blue', fontsize=12, rotation=90)
plt.xticks(np.arange(0, 1020, 50),fontsize=9, color='green', rotation=90)
plt.yticks(np.arange(0, 2250, 500),fontsize=9, color='green', rotation=45)
plt.grid(color='red', alpha=0.3)
plt.legend(loc=0, fontsize=12, framealpha=0.9)
plt.savefig('Рисунок 1.png')
plt.show()

plt.figure(figsize=(12, 4))

plt.bar(df['City'].value_counts().index, df['City'].value_counts(), color='b')

plt.title('CIty', fontsize=16, color='black')
plt.xlabel('City', fontsize=12, color='grey')
plt.xticks(fontsize=12, color='red', rotation=0)
plt.yticks(fontsize=12, color='red', rotation=0)
plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.3)
plt.savefig('Рисунок 8. Bar plot for "Okrug.jpeg', dpi=300)
plt.show()

df

plt.figure(figsize=(12, 3))
plt.title('Цена с налогом', color='red', fontsize=18)


plt.scatter(x=df['Unit price'], y=df['Tax 5%'], color='black', )


plt.xlabel('Цена за штуку', color='blue', fontsize=12)
plt.ylabel('Налог 5%', color='blue', fontsize=12, rotation=90)
plt.xticks(fontsize=9, color='green', rotation=90)
plt.yticks(fontsize=9, color='green', rotation=45)
plt.grid(color='red', alpha=0.3)
plt.legend(loc=0, fontsize=12, framealpha=0.9)
plt.savefig('Рисунок 1.png')
plt.show()

df.info()

df.groupby(['Gender', 'Product line']).sum(numeric_only=True)

df.info()

df['Branch'].value_counts()

df[['A', 'B', 'C']] = pd.get_dummies(df['Branch'], dtype=int)

df.drop(columns=['Branch'], inplace=True)

df['City'].value_counts()

columns_list = pd.get_dummies(df['City'], dtype=int).columns

df[columns_list] = pd.get_dummies(df['City'], dtype=int)
df.corr(numeric_only=True)

df.head()
df.drop(columns=['City'], inplace=True)

columns_list = pd.get_dummies(df['Customer type'], dtype=int).columns

df[columns_list] = pd.get_dummies(df['Customer type'], dtype=int)

df.drop(columns=['Customer type'], inplace=True)

columns_list = pd.get_dummies(df['Gender'], dtype=int).columns

df[columns_list] = pd.get_dummies(df['Gender'], dtype=int)

df.drop(columns=['Gender'], inplace=True)

columns_list = pd.get_dummies(df['Product line'], dtype=int).columns

df[columns_list] = pd.get_dummies(df['Product line'], dtype=int)

df.drop(columns=['Product line'], inplace=True)

columns_list = pd.get_dummies(df['Payment'], dtype=int).columns

df[columns_list] = pd.get_dummies(df['Payment'], dtype=int)

df.drop(columns=['Payment'], inplace=True)

df['Ewallet'].value_counts()/len(df)*100
df.columns

df.info()

df.isnull().sum()

X_columns = df.drop(columns=['Total', 'Invoice ID', 'DateTime', 'cogs', 'Tax 5%', 'gross income']).columns

X = df[X_columns]

Y = df['Total']

X_columns

df.head()

df.corr(numeric_only=True)



X.shape, Y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.10, random_state=10)

X_train.shape, X_test.shape

X_train

X.describe()



from sklearn.preprocessing import LabelEncoder


labelencoder = LabelEncoder()

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error


def all_reg_scores(model, name_model, X_test, Test_y):

    MAE = round(mean_absolute_error(Y_test, model.predict(X_test)), 2)
    MSE = round(mean_squared_error(Y_test, model.predict(X_test)), 2)
    RMSE = round(np.sqrt(mean_squared_error(Y_test, model.predict(X_test))), 2)
    MAPE = round(mean_absolute_percentage_error(Y_test, model.predict(X_test))*100, 4)
    R2 = round(r2_score(Y_test, model.predict(X_test)), 4)

    print(f'{name_model} model: \n', '      r2_score: {0}     MAPE (%): {1}     MAE: {2}     RMSE: {3}     MSE: {4}'.format(R2, MAPE, MAE, RMSE, MSE))
    df.describe()

    from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR.fit(X_train, Y_train)

LR.score(X_test, Y_test)


y_pred = LR.predict(X_test)

all_reg_scores(LR, 'LR', X_test, Y_test)

plt.figure(figsize=(17, 3))
plt.plot(y_pred, color='r', label='прогнозные')
plt.plot(Y_test, color='g', label='реальные')
plt.grid()
plt.legend()

plt.figure(figsize=(17, 3))
plt.plot(pd.DataFrame(y_pred).sort_values(0).reset_index().drop('index', axis=1), color='r', label='прогнозные')
plt.plot(pd.DataFrame(Y_test).sort_values(0).reset_index().drop('index', axis=1), color='g', label='реальные')
plt.grid()
plt.legend()

from sklearn.neighbors import KNeighborsRegressor

KNNR = KNeighborsRegressor(n_neighbors=6)

KNNR.fit(X_train, Y_train)

KNNR.score(X_test, Y_test)

all_reg_scores(KNNR, 'KNNR', X_test, Y_test)
from sklearn.tree import DecisionTreeRegressor

DTR = DecisionTreeRegressor(max_depth=15, random_state=10)

DTR.fit(X_train, Y_train)

DTR.score(X_test, Y_test)

all_reg_scores(DTR, 'DTR', X_test, Y_test)
from sklearn.ensemble import BaggingRegressor

BGR = BaggingRegressor(base_estimator=LinearRegression(),
                        n_estimators=10,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        oob_score=False,
                        warm_start=False,
                        n_jobs=None,
                        random_state=10,
                        verbose=0,)

BGR.fit(X_train, Y_train.ravel())
BGR.score(X_test, Y_test.ravel())

all_reg_scores(BGR, 'BGR', X_test, Y_test)

Pred_BGR = BGR.predict(X_test)
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=100,
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=4,
                            min_weight_fraction_leaf=0.0,
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            bootstrap=True,
                            oob_score=False,
                            n_jobs=None,
                            random_state=0,
                            verbose=0,
                            warm_start=False,
                            ccp_alpha=0.0,
                            max_samples=None,)

RFR.fit(X_train, Y_train.ravel())
RFR.score(X_test, Y_test.ravel())

all_reg_scores(RFR, 'RFR', X_test, Y_test)

columns = X.columns

sorted_idx = RFR.feature_importances_.argsort()
plt.barh(columns[sorted_idx], RFR.feature_importances_[sorted_idx])
plt.xlabel("RFR Feature Importance")
from sklearn.ensemble import ExtraTreesRegressor

ExTR = ExtraTreesRegressor(n_estimators=100,
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            bootstrap=False,
                            oob_score=False,
                            n_jobs=None,
                            random_state=66,
                            verbose=0,
                            warm_start=False,
                            ccp_alpha=0.0,
                            max_samples=None,)

ExTR.fit(X_train, Y_train.ravel())

ExTR.score(X_test, Y_test.ravel())

all_reg_scores(ExTR, 'ExTR', X_test, Y_test)

y_pred_ExTR = ExTR.predict(X_test)

sorted_idx = ExTR.feature_importances_.argsort()
plt.barh(columns[sorted_idx], ExTR.feature_importances_[sorted_idx])
plt.xlabel("ExTR Feature Importance")
from sklearn.ensemble import AdaBoostRegressor

AdBR = AdaBoostRegressor(random_state=0, n_estimators=100)

AdBR.fit(X_train, Y_train.ravel())

AdBR.score(X_test, Y_test.ravel())

all_reg_scores(AdBR, 'AdBR', X_test, Y_test)
from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(
                                learning_rate=0.1,
                                n_estimators=100,
                                subsample=1.0,
                                criterion='friedman_mse',
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_depth=3,
                                min_impurity_decrease=0.0,
                                init=None,
                                random_state=21,
                                max_features=None,
                                alpha=0.9,
                                verbose=0,
                                max_leaf_nodes=None,
                                warm_start=False,
                                validation_fraction=0.1,
                                n_iter_no_change=None,
                                tol=0.0001,
                                ccp_alpha=0.0,)

GBR.fit(X_train, Y_train.ravel())
GBR.score(X_test, Y_test.ravel())

all_reg_scores(GBR, 'GBR', X_test, Y_test)
from xgboost import XGBRegressor

XGBR = XGBRegressor(max_depth=10,
                   learning_rate=0.1,
                   n_estimators=1000,
                   reg_alpha=0.001,
                   reg_lambda=0.000001,
                   n_jobs=-1,
                   min_child_weight=3)

XGBR.fit(X_train, Y_train)

XGBR.score(X_test, Y_test)

all_reg_scores(XGBR, 'XGBR', X_test, Y_test)
import lightgbm as ltb

lgbm = ltb.LGBMRegressor()

#Defining a dictionary containing all the releveant parameters
param_grid = {
    "boosting_type": ['gbdt'],
    "num_leaves": [9, 19],  #[ 19, 31, 37, 47],
    "max_depth": [29], #[7, 15, 29, 37, 47, 53],
    "learning_rate": [0.1, 0.15],
    "n_estimators": [1000], #[500, 1000, 2000],
    "subsample_for_bin": [200000], #[20000, 200000, 2000000],
    "objective": ["regression"],
    "min_child_weight": [0.01], #[0.001, 0.01],
    "min_child_samples":[100, 200], #[20, 50, 100],
    "subsample":[1.0],
    "subsample_freq":[0],
    "colsample_bytree":[1.0],
    "reg_alpha":[0.0],
    "reg_lambda":[0.0]
}

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import model_selection

model_lgbm = model_selection.RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_grid,
            n_iter=100,
            scoring="neg_root_mean_squared_error",
            verbose=10,
            n_jobs=-1,
            cv=5
        )

model_lgbm.fit(X_train, Y_train)

print(f"Best score: {model_lgbm.best_score_}")
print("Best parameters from the RandomSearchCV:")
best_parameters = model_lgbm.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print(f"\t{param_name}: {best_parameters[param_name]}")

# Get best model
LGBMR = model_lgbm.best_estimator_
Y_pred_lgb = LGBMR.predict(X_test)

r2_score(Y_pred_lgb, Y_test)

all_reg_scores(LGBMR, 'LGBMR', X_test, Y_test)
all_reg_scores(LR, 'LR', X_test, Y_test)
all_reg_scores(KNNR, 'KNNR', X_test, Y_test)
all_reg_scores(DTR, 'DTR', X_test, Y_test)
all_reg_scores(BGR, 'BGR', X_test, Y_test)
all_reg_scores(RFR, 'RFR', X_test, Y_test)
all_reg_scores(ExTR, 'ExTR', X_test, Y_test)
all_reg_scores(AdBR, 'AdBR', X_test, Y_test)
all_reg_scores(GBR, 'GBR', X_test, Y_test)
all_reg_scores(XGBR, 'XGBR', X_test, Y_test)
all_reg_scores(LGBMR, 'LGBMR', X_test, Y_test)
