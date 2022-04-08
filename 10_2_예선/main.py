## 미리 import 할 함수를 정리해봅시다!!
# !pip install missingno
# !pip install pycaret
from models import *

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno

from sklearn.model_selection import train_test_split

import pycaret

def main():
    df = pd.read_csv('../data/dacon/comp1/train.csv', header=0, index_col=0)
    df_test = pd.read_csv('../data/dacon/comp1/train.csv', header=0, index_col=0)
    
    # 함수 사용해서 이상치 값 삭제 ''에 변수명
    oulier_idx = get_outlier(df=df, column='', weight=1.5)
    df.drop(outlier_idx, axis=0, inplace=True)
    
    model = model(args.model)
    train, valid = train_test_split(df, test_size=0.2, random_state=42)
    train_scaled, test_scaled = standard_scaler(train_df, test_df)
    
    train_droplist = ['컬럼명1' , '컬럼명2',.. ]
    test_droplist = ['컬럼명1' , '컬럼명2',.. ]
    train_df = train.drop(train_droplist, axis=1)
    test_df = test.drop(test_droplist, axis=1)

    train_notscaled = train[train_droplist]
    test_notscaled = test[test_droplist]
    train = pd.concat([train_scaled, train_notscaled], axis =1)
    test = pd.concat([test_scaled, test_notscaled], axis =1)
    
    train_log, test_log = logtransform(train_df, test_df)

    train_notscaled = train[train_droplist]
    test_notscaled = test[test_droplist]
    train = pd.concat([train_log, train_notscaled], axis =1)
    test = pd.concat([test_log, test_notscaled], axis =1)
    
    train = preprocess(train)
    
    
    model(train, valid)
    predict = model.prediction(df_test)
    pd.out_csv(predict, '../data/dacon/comp1/sample_submission.csv')
    

def __init__():
    print("예선 과제 시작")
    main()
    
