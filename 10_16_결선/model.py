from models import *

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno

from sklearn.model_selection import train_test_split

import pycaret

def classifier(log=None):
    print("="*50,"Start XGB","="*50)
    model_xgb = xgb.XGBClassifier(random_state=1120) 
    model_xgb.fit(X_train, y_train)
    models_list.append(model_xgb)
    y_pred_xgb = model_xgb.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_xgb)
    print('XGB accuracy 값: {0:.4f}'.format(accuracy))
    
    print("="*50,"Start LGBM","="*50)
    model_lgb = lgb.LGBMClassifier(random_state=1120)
    model_lgb.fit(X_train, y_train)
    models_list.append(model_lgb)
    y_pred_lgb = model_lgb.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_lgb)
    print('LGBM accuracy 값: {0:.4f}'.format(accuracy))

    print("="*50,"Start Random Forest","="*50)
    model_rf = RandomForestClassifier(random_state=1120)
    model_rf.fit(X_train, y_train)
    models_list.append(model_rf)
    y_pred_rf = model_rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_rf)
    print('Random Forest accuracy 값: {0:.4f}'.format(accuracy))
    
def regressor( met, log=None):
    
    print("="*50,"Start XGB","="*50)
    model_xgb = xgb.XGBRegressor(random_state=1120, metric=met) 
    model_xgb.fit(X_train, y_train)
    models_list.append(model_xgb)
    if log==True:
        y_pred_xgb = np.expm1(model_xgb.predict(X_val))
        score = np.sqrt(mse(np.expm1(y_val), y_pred_xgb))
        print('XGB RMSE 값: {0:.4f}'.format(score))
    elif log==False:
        y_pred_xgb = model_xgb.predict(X_val)
        score = np.sqrt(mse(y_val, y_pred_xgb))
        print('XGB RMSE 값: {0:.4f}'.format(score))
    
    print("="*50,"Start LGBM","="*50)
    model_lgb = lgb.LGBMRegressor(random_state=1120, metric='rmse')
    model_lgb.fit(X_train, y_train)
    models_list.append(model_lgb)
    if log==True:
        y_pred_lgb = np.expm1(model_lgb.predict(X_val))
        score = np.sqrt(mse(np.expm1(y_val), y_pred_lgb))
        print('LGBM RMSE 값: {0:.4f}'.format(score))
    elif log==False:
        y_pred_lgb = model_lgb.predict(X_val)
        score = np.sqrt(mse(y_val, y_pred_lgb))
        print('LGBM RMSE 값: {0:.4f}'.format(score))

    print("="*50,"Start Ridge","="*50)
    model_ridge=Ridge(alpha=10)
    model_ridge.fit(X_train, y_train)
    models_list.append(model_ridge)
    if log==True:
        y_pred_ridge = np.expm1(model_ridge.predict(X_val))
        score = np.sqrt(mse(np.expm1(y_val), y_pred_ridge))
        print('Ridge RMSE 값: {0:.4f}'.format(score))
    elif log==False:
        y_pred_ridge = model_ridge.predict(X_val)
        score = np.sqrt(mse(y_val, y_pred_ridge))
        print('Ridge RMSE 값: {0:.4f}'.format(score))

def get_avg_rmse_cv(models=models_list, folds):
    for model in models:
        score_list = cross_val_score(model, X_train, y_train, cv = folds)
        score_avg = np.mean(score_list)
        print('\n{0} CV accuracy: {1}'.format( model.__class__.__name__, np.round(score_list, 3)))
        print('{0} CV 평균 accuracy: {1}'.format( model.__class__.__name__, np.round(score_avg, 3)))
        
def get_best_params(model, params, folds):
    grid_cv = GridSearchCV(model, param_grid=params , cv=folds, n_jobs=-1 )
    grid_cv.fit(X_train , y_train)
    score = grid_model.best_score_
    print('{0} 5 CV 시 최적 평균 accuracy: {1}, 최적 하이퍼파라미터:{2}'.format(model.__class__.__name__,
                                        np.round(score, 4), grid_model.best_params_))
    return grid_model.best_estimator_