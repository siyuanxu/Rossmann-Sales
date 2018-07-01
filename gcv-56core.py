from sklearn.metrics import make_scorer

def RMSPE(test,pred):
    # 在特征工程中，Sales被取对数，这里计算评分时要还原
    test = np.e**test
    pred = np.e**pred
    return np.sqrt(np.mean(((test-pred)/pred)**2))
scorer = make_scorer(RMSPE)

# 引用必要的库
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# 读入数据集
cesar_train = pd.read_csv('data/cesar_train.csv',index_col=False)
cesar_test = pd.read_csv('data/cesar_test.csv',index_col=False)
X = cesar_train.drop(['Sales'],axis=1)
y = cesar_train.Sales

from sklearn.model_selection import GridSearchCV, KFold

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.05, random_state=42)

if __name__ == '__main__':
    RF_cherry = RandomForestRegressor(random_state=42)
    parameters = {'n_estimators': np.arange(125, 176, 5),
                 'max_features': np.arange(0.15, 0.21, 0.02),
                 'max_depth': np.arange(18, 25, 1)}
    kfold = KFold(n_splits=10)

    grid = GridSearchCV(RF_cherry, parameters, scorer,
                        cv=kfold, n_jobs=-1, verbose=1)
    grid = grid.fit(X_train, y_train)

    print(grid.best_params_)
    print(grid.best_score_)

    gridcv_results = pd.DataFrame(grid.cv_results_).T

    gridcv_results.to_csv('gcv.csv')