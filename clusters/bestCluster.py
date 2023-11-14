import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

def compare_algorithms(X, max_clusters):
    results = []
    cluster_range = range(2, max_clusters + 1)

    #KMeans
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        cluster = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster)
        results.append(("KMeans", n_clusters, silhouette_avg))

    #Agglomerative clustering
    for n_clusters in cluster_range:
        agglomerative_cluster = AgglomerativeClustering(n_clusters=n_clusters)
        cluster = agglomerative_cluster.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster)
        results.append(("Agglomerative", n_clusters, silhouette_avg))

    #DBSCAN
    eps_values = np.arange(0.1, 0.9, 0.1)
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        cluster = dbscan.fit_predict(X)

        if len(set(cluster)) > 1:
            silhouette_avg = silhouette_score(X, cluster)
            results.append(("DBSCAN", eps, silhouette_avg))

    return results

iris = datasets.load_iris()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(iris["data"])

results = compare_algorithms(scaled_data, 10)
df = pd.DataFrame(results, columns=["Agrupador", "Clusters", "Score"])
print(df)

max_score_index = df["Score"].idxmax()
print(df.iloc[max_score_index])
