import missingno as msno

## 시각화
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
	else: assert True "존재하지 않는 imputation 방법입니다..ㅠ" 
	return df 