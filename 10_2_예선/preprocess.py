import numpy as np

def preprocess(train, test):
    limit = np.quantile(train['sqft_living15'],0.99)
    train.drop(train[(train.sqft_living15 > limit)].index, axis=0, inplace=True)
    
    for x in range(train.shape[0]):
        if train.iloc[x,15] != 0:
            train.iloc[x, 14] = train.iloc[x, 15]

    train['re_built_year'] = [0 for _ in range(train.shape[0])]
    
    for x in range(train.shape[0]):
        if train.iloc[x,14] < 1930:
            train['re_built_year'].iloc[x] = 0
        elif train.iloc[x,14] < 1960:
            train['re_built_year'].iloc[x] = 1
        elif train.iloc[x,14] < 1990:
            train['re_built_year'].iloc[x] = 2
        else:
            train['re_built_year'].iloc[x] = 3
    
    for data in [train,test]:
        data['new_condition'] = 0
        data.loc[(data['condition']==1) | (data['condition']==2)  ,['new_condition']]= 0
        data.loc[(data['condition']>=3)  ,['new_condition']]= 1
    
    # 7이상 베드룸은 다 7개로 처리, 베드룸 개수 0은 1로 편입 train 데이터에서 0인거 3개, test에서 1개
    for data in [train,test]:
        data.loc[data['bedrooms'] >= 7,['bedrooms']]= 7
        data.loc[data['bedrooms'] == 0,['bedrooms']]= 1
        
    # 베스룸 데이터가 소수점이라 버림 처리 통해 다 정수화
    for data in [train,test]:
        data.loc[data['bathrooms']< 1 ,['bathrooms']]= 0
        data.loc[(data['bathrooms']>=1) &(data['bathrooms']< 2) ,['bathrooms']]= 1
        data.loc[(data['bathrooms']>=2) & (data['bathrooms']< 3) ,['bathrooms']]= 2
        data.loc[(data['bathrooms']>=3) & (data['bathrooms']< 4) ,['bathrooms']]= 3
        data.loc[(data['bathrooms']>=4) & (data['bathrooms']< 5) ,['bathrooms']]= 4
        data.loc[(data['bathrooms']>=5) & (data['bathrooms']< 6) ,['bathrooms']]= 5
        data.loc[(data['bathrooms']>=6) & (data['bathrooms']< 7) ,['bathrooms']]= 6
        data.loc[(data['bathrooms']>=7) & (data['bathrooms']< 8) ,['bathrooms']]= 7
        data.loc[data['bathrooms']>=8 ,['bathrooms']]= 8
    # 베스룸 5이상인건 다 5로 처리
    for data in [train,test]:
        data.loc[data['bathrooms']>=5 ,['bathrooms']]= 5
    
    activation = np.ones(train.shape[0])
    train['activation'] = activation
    train.loc[train['month'].isin([11,12,1,2]),['activation']] = 0

    activation = np.ones(test.shape[0])
    test['activation'] = activation
    test.loc[test['month'].isin([11,12,1,2]),['activation']] = 0
    
    train['renovation'] = train['sqft_lot']-train['sqft_lot15']
    test['renovation'] = test['sqft_lot']-test['sqft_lot15']
    
    return train