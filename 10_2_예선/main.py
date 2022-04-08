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
    model = model(args.model)
    train, valid = train_test_split(df, test_size=0.2, random_state=42)
    train = preprocess(train)
    model(train, valid)
    predict = model.prediction(df_test)
    pd.out_csv(predict, '../data/dacon/comp1/sample_submission.csv')
    

def __init__():
    print("예선 과제 시작")
    main()
    
