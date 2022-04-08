import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('../data/dacon/comp1/train.csv', header=0, index_col=0)
label = df['label']

train_x, train_y, valid_x, valid_y = train_test_split(df, label, test_size=0.2, random_state=42)

train_df = train.drop(['id','price', 'date', 'sqft_living', 'sqft_lot', 'waterfront', 'yr_built',	'yr_renovated',	'zipcode',	'lat',	'long', 'log_price'	], axis=1)
test_df = test.drop(['id', 'date', 'sqft_living', 'sqft_lot', 'waterfront', 'yr_built',	'yr_renovated',	'zipcode',	'lat',	'long'	], axis=1)
train_df.head()

scaler = StandardScaler()
scaler.fit(train_df)
train_scaled = scaler.transform(train_df)
test_scaled = scaler.transform(test_df)#테스트, 트레인데이터 스탠다드 스케일링 완료

# 스케일링 된 피처 분포보기
pd.DataFrame(train_scaled).hist(bins=50, figsize=(13,13))
plt.show()

#로그 변환
train_log = np.log1p(train_df)
test_log = np.log1p(test_df)
train_log.head()