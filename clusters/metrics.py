from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from pyClusterTend.hopkins import *
from pyClusterTend.metric import *
from pyClusterTend.visual_assessment_of_tendency import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

iris = datasets.load_iris()
cluster1 = scale(iris["data"])
cluster2 = scale(np.random.rand(150,4)) #matrix with 150 rows and 4 columns of random data

#inertia evaluates the quality of the grouping and is generated by the model itself
#it's a measure of how compact the clusters formed by the data are
inertia = []
for i in range(1,8):
    kmeans = KMeans(n_clusters=i, n_init="auto")
    kmeans.fit(cluster1)
    inertia.append(kmeans.inertia_)
plt.plot(range(1,8), inertia)
plt.title("Elbow")
plt.show() #in this graph, we can see that the ideal number of clusters would be 3


inertia = []
for i in range(1,8):
    kmeans = KMeans(n_clusters=i, n_init="auto")
    kmeans.fit(cluster2)
    inertia.append(kmeans.inertia_)
plt.plot(range(1,8), inertia)
plt.title("Elbow")
plt.show() #in this graph, the number of clusters will vary because the data is random


#hopkings stats: the closer to zero, the greater the tendency to find clusters in the data
hopkins_cluster1 = hopkins(cluster1, 150)
hopkins_cluster2 = hopkins(cluster2, 150)
print("hopkins cluster 1: ", hopkins_cluster1)
print("hopkins cluster 2: ", hopkins_cluster2)

vat(cluster1)
vat(cluster2)

ivat(cluster1)
ivat(cluster2)

#for this data, the ideal number of clusters apperars to be 2
n = assess_tendency_by_metric(cluster1, "silhouette", 5) #5 if the number of clusters I want to evaluate
print(n) #(2, array([0.58175005, 0.45994824, 0.38349897, 0.3471606 ])) -> this means that 0.58175005 is for 2 clusters

n2 = assess_tendency_by_metric(cluster1, "davies_bouldin", 5)
print(n2) #(2, array([0.59331269, 0.83359495, 0.869779  , 0.95619058]))

n3 = assess_tendency_by_metric(cluster1, "calinski_harabasz", 5)
print(n3) #(2, array([251.34933946, 241.9044017 , 207.26660627, 203.26741933]))



#for this data, the ideal number of clusters apperars to be 5
n = assess_tendency_by_metric(cluster2, "silhouette", 5)
print(n) #(5, array([0.21928933, 0.21197609, 0.2279224 , 0.24679333]))

n2 = assess_tendency_by_metric(cluster2, "davies_bouldin", 5)
print(n2) #(5, array([1.72621591, 1.51475105, 1.40357406, 1.24599214]))

n3 = assess_tendency_by_metric(cluster2, "calinski_harabasz", 5)
print(n3) #(2, array([46.29403093, 42.03079098, 41.91938305, 43.1976191 ]))


n = assess_tendency_by_mean_metric_score(cluster1, 5) #this gets the mean of the 3 methods (silhouette, davies_bouldin, calinski_harabasz)
print(n) #2.0

n2 = assess_tendency_by_mean_metric_score(cluster2, 5)
print(n2) #4.333333333333333