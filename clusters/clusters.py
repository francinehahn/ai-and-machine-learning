import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import confusion_matrix

def plot_clusters(data, labels, title):
    colors = ["red", "green", "purple", "black"] #black represents noise
    plt.figure(figsize=(8,4))
    for i, c, l in zip(range(-1,3), colors, ["Noise", "Setosa", "Versicolor", "Virginica"]):
        if i == -1:
            plt.scatter(data[labels == i, 0], data[labels == i, 3], c = colors[i], label = l, alpha=0.5, s=50, marker="x")
        else:
            plt.scatter(data[labels == i, 0], data[labels == i, 3], c = colors[i], label = l, alpha=0.5, s=50)

    plt.legend()
    plt.title(title)
    plt.xlabel("Sepal length")
    plt.ylabel("Petal width")
    plt.show()

iris = datasets.load_iris()

kmeans = KMeans(n_clusters=3, n_init="auto")
kmeans.fit(iris["data"])
print("kmens labels", kmeans.labels_)

#METRICS:
cm = confusion_matrix(iris["target"], kmeans.labels_)
print("cm: ", cm)

plot_clusters(iris["data"], kmeans.labels_, "Cluster Kmeans")

#the higher the eps, the lower the number os clusters.
#the higher the min samples, the lower the noise
dbscan = DBSCAN(eps=0.5, min_samples=3) #min samples is the minimum number of samples to be considered a cluster. Otherwise, it's just noise
dbscan_labels = dbscan.fit_predict(iris["data"])
print("dbscan labels ", dbscan_labels) #-1 is noise

plot_clusters(iris["data"], dbscan_labels, "Cluster DBSCAN")


agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(iris["data"])
print("agglo labels: ", agglo_labels)

plot_clusters(iris["data"], agglo_labels, "Hierarchical Cluster")

plt.figure(figsize=(12,6))
plt.title("Hierarchical cluster dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")
linkage_matrix = linkage(iris["data"], method="ward")
dendrogram(linkage_matrix, truncate_mode="lastp", p=15)
plt.axhline(y=7, c="gray", lw=1, linestyle="dashed")
plt.show()