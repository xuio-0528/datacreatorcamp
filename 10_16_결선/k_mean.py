from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
# '' 안에는 clustering시킬 변수명 넣기

def change_n_clusters(n_clusters, data):
    sum_of_squared_distance = []
    for n_cluster in n_clusters:
        kmeans = KMeans(n_clusters=n_cluster)
        kmeans.fit(data)
        sum_of_squared_distance.append(kmeans.inertia_)
        
    plt.figure(1 , figsize = (12, 6))
    plt.plot(n_clusters , sum_of_squared_distance , 'o')
    plt.plot(n_clusters , sum_of_squared_distance , '-' , alpha = 0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    change_n_clusters([2,3,4,5,6,7,8,9,10,11], train)


    model = KMeans(n_clusters = 4, random_state = 42)
    pred = model.fit_predict(train[['', '', '']])

    train['km_cluster'] = pred
    test['km_cluster'] = model.predict(test[['', '', '']])

    marker0_ind = train[ train['km_cluster']==0].index
    marker1_ind = train[ train['km_cluster']==1].index
    marker2_ind = train[ train['km_cluster']==2].index
    marker3_ind = train[ train['km_cluster']==3].index


    # cluster값 0, 1, 2에 해당하는 Index로 각 cluster 레벨의 pca_x, pca_y 값 추출. o, s, ^ 로 marker 표시
    plt.scatter(x=train.loc[marker0_ind,''], y=train.loc[marker0_ind,''], marker='o') 
    plt.scatter(x=train.loc[marker1_ind,''], y=train.loc[marker1_ind,''], marker='s')
    plt.scatter(x=train.loc[marker2_ind,''], y=train.loc[marker2_ind,''], marker='^')
    plt.scatter(x=train.loc[marker3_ind,''], y=train.loc[marker3_ind,''], marker='x')


    plt.xlabel('')
    plt.ylabel('')
    plt.title('4 Clusters Visualization by 2 Components')
    plt.show()