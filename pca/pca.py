#Principal Component Analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

iris = datasets.load_iris()
forecasters = iris["data"]
iris_class = iris["target"]

print("forecasters: ", forecasters)

sc = StandardScaler()
forecasters = sc.fit_transform(forecasters)

print("forecasters: ", forecasters)

X_train, X_test, y_train, y_test = train_test_split(forecasters, iris_class, test_size=0.3, random_state=123)

forest = RandomForestClassifier(n_estimators=100, random_state=1234)
forest.fit(X_train, y_train)

predictions = forest.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("accuracy: ", accuracy)

pca = PCA(n_components=3) #number of components must be less than the number of attributes, which is 4 in this case
forecasters = pca.fit_transform(forecasters)

print("forecasters: ", forecasters) #characteristics artificially created from the original attributes (petal and sepal measurements)

X_train, X_test, y_train, y_test = train_test_split(forecasters, iris_class, test_size=0.3, random_state=123)

forest = RandomForestClassifier(n_estimators=100, random_state=1234)
forest.fit(X_train, y_train)

predictions = forest.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("accuracy: ", accuracy)

