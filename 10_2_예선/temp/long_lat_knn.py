from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def lon_lat(train):
    long = train['long']
    lat = train['lat']

    test_long = test['long']
    test_lat = test['lat']

    cl_df = pd.concat([long, lat], axis=1)
    test_cl_df = pd.concat([test_long, test_lat], axis=1)

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

    change_n_clusters([2,3,4,5,6,7,8,9,10,11], cl_df.iloc[:,1:])


    model = KMeans(n_clusters = 5, random_state = 42)
    pred = model.fit_predict(cl_df.iloc[:, 1:])
    test_pred = model.fit_predict(test_cl_df.iloc[:,1:])

    cl_df['km_cluster'] = pred

    train_cl = pd.merge(train, cl_df[['km_cluster']], how='left', on=train.index)
    return train_cl, test_pred