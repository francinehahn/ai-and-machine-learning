import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2, SelectKBest

ad = pd.read_csv("files/ad.data", header=None)
print(ad.shape)

X = ad.iloc[:,:-1].values #all the columns for x
y = ad.iloc[:,-1].values #only the last column for y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model1 = GaussianNB()
model1.fit(X_train, y_train)

predictions1 = model1.predict(X_test)
accuracy1 = accuracy_score(y_test, predictions1)

print("accuracy: ", accuracy1)

selection = SelectKBest(chi2, k=7)
X_new = selection.fit_transform(X, y)

print(X_new.shape) #now we have only 7 columns instead of 1559

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=0)

model2 = GaussianNB()
model2.fit(X_train, y_train)

predictions2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, predictions2)

print("accuracy: ", accuracy2)