from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

def model(model, X_train, y_train, X_val, y_val) :
    if model == "xgb":
        model_xgb = xgb.XGBRegressor(random_state=1120, metric='rmse') 
        model_xgb.fit(X_train, y_train)
        y_pred_xgb = np.expm1(model_xgb.predict(X_val))
        score = np.sqrt(mse(np.expm1(y_val), y_pred_xgb))
        print('LGBM RMSE 값: {0:.4f}'.format(score))

    # LGBM
    if model == "lgbm":
        model_lgb = lgb.LGBMRegressor(random_state=1120, metric='rmse')
        model_lgb.fit(X_train, y_train)
        y_pred_lgb = np.expm1(model_lgb.predict(X_val))
        score = np.sqrt(mse(np.expm1(y_val), y_pred_lgb))
        print('LGBM RMSE 값: {0:.4f}'.format(score))


    if model == "ridge":
        model_ridge=Ridge(alpha=10)
        model_ridge.fit(X_train, y_train)
        y_pred_ridge=np.expm1(model_ridge.predict(X_val))
        score = np.sqrt(mse(np.expm1(y_val), y_pred_ridge))
        print('Ridge RMSE 값: {0:.4f}'.format(score))
    
    return model