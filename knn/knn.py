import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

mtcars = pd.read_csv('files/mt_cars.csv')

#accident is de dependent variable and it is in column 7
y = mtcars['cyl'].values
X = mtcars[['mpg', 'hp']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3) #it's better to use an odd number to avoid a tie
model = knn.fit(X_train, y_train)

y_predict = model.predict(X_test)
print("predict y: ", y_predict)

#METRICS:
accuracy = accuracy_score(y_test, y_predict)
print("accuracy: ", accuracy)

precision = precision_score(y_test, y_predict, average="weighted")
print("precision: ", precision)

recall = recall_score(y_test, y_predict, average="weighted")
print("recall: ", recall)

f1 = f1_score(y_test, y_predict, average="weighted")
print("f1 score: ", f1)

cm = confusion_matrix(y_test, y_predict)
print("cm: ", cm)
