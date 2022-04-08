import missingno as msno

## 시각화

def check(df, train_df):
    msno.matrix(train_df)

    ## 수치로 확인
    df.info()

def imputation(df,columns,method):
##############################################
##  columns: 결측치가 존재하는 컬럼명 리스트
##  method: 결측치를 채우는 방법을 string 형태로
##           - mean, mode, interpolate 
##############################################
	## 평균으로 결측치 채우기
	if method == "mean":
		df[columns] = df[columns].fillna(df.mean())
	## 최빈값으로 결측치 채우기
	elif method == "mode":
		df[columns] = df[columns].fillna(df.mode())
	elif method == "interpolate":
	## 보간법으로 결측치 채우기
		df[columns] = df[columns].interpolate(method='linear')
	else: 
     assert False, "method not supported"
	
    return df 

def preproces(train, test):
    # 작동 방식
    # train_droplist = ['컬럼명1' , '컬럼명2',.. ]
    # test_droplist = ['컬럼명1' , '컬럼명2',.. ]
    # train_df = train.drop(train_droplist, axis=1)
    # test_df = test.drop(test_droplist, axis=1)
    
    # 스탠다드 스케일러
    def standard_scaler(train_df, test_df):  #인자로는 스케일링 시킬것만 남긴 df 넣기
        from sklearn.preprocessing import StandardScaler #원 데이터프레임에서 스케일링 안할건 drop
        scaler = StandardScaler()
        scaler.fit(train_df)
        train_scaled = scaler.transform(train_df)
        test_scaled = scaler.transform(test_df)
        train_scaled = pd.DataFrame(data= train_scaled, columns= train_df.columns)
        test_scaled = pd.DataFrame(data = test_scaled, columns = test_df.columns)
        return train_scaled, test_scaled # 스케일링 할 피처들만 다 스케일링 된 데이터 프레임
                                                                            # 스케일링 필요없는 피처들 데이터프레임과 concat 필요
    # 로그변환
    def logtransform(train_df, test_df):  #인자로는 스케일링 시킬것만 남긴 df 넣기
        import numpy as np
        train_log = np.log1p(train_df)
        test_log = np.log1p(test_df)
        return train_log, test_log
    
    import numpy as np

    def get_outlier(df=None, column=None, weight=1.5):
    # target 값과 상관관계가 높은 열을 우선적으로 진행
        quantile_25 = np.percentile(df[column].values, 25)
        quantile_75 = np.percentile(df[column].values, 75)

        IQR = quantile_75 - quantile_25
        IQR_weight = IQR*weight
        
        lowest = quantile_25 - IQR_weight
        highest = quantile_75 + IQR_weight
        
        outlier_idx = df[column][ (df[column] < lowest) | (df[column] > highest) ].index
        return outlier_idx

