train_labellist = ['컬럼명 1', '컬럼명2',...]
test_labellist = ['컬럼명 1', '컬럼명2',...]
train_label = train[train_labellist]
test_label = test[test_labellist]

# 라벨 인코더
def labelencoding(train_label, test_label):
	from sklearn.preprocessing import LabelEncoder
	encoder = LabelEncoder()
	encoder.fit(train_label)
	train_labels = encoder.transform(train_label)
	test_labels = encoder.transform(test_label)
	return train_labels, test_labels

train_labels, test_labels = labelencoding(train_label, test_label)
train_notencoding = train.drop(train_labellist, axis =1)
test_notencoding = test.drop(test_labellist, axis =1)
train_afterencoding = pd.concat([train_labels, train_notencoding], axis =1)
test_afterencoding = pd.concat([test_labels, test_notencoding], axis =1)

train_onehotlist = ['컬럼명1', '컬럼명 2',...]
test_onehotlist = ['컬럼명1', '컬럼명 2',...]
train_onehot = train[train_onehotlist]
test_onehot = test[test_onehotlist]

#원핫 인코딩
def onehotencoding(train_onehot, test_onehot):
	import pandas as pd 
	train_aftonehot = pd.get_dummies(train_onehot)
	test_aftonehot = pd.get_dummies(test_onehot)
	return train_afteronehot, test_afteronehot

train_afteronehot, test_afteronehot = onehotencoding(train_onehot, test_onehot)
train_notonehot = train.drop(train_onehotlist, axis =1)
test_notonehot = test.drop(test_onehotlist, axis =1)
train_aftonehotencoding = pd.concat([train_afteronehot, train_notonehot], axis =1)
test_aftonehotencoding = pd.concat([test_afteronehot, test_notonehot], axis =1)