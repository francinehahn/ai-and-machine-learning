import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

base = pd.read_csv('files/insurance.csv', keep_default_na=False, na_values=[''])
base = base.drop(columns=['Unnamed: 0'])

#accident is de dependent variable and it is in column 7
y = base.iloc[:,7].values #getting all lines from column 7
X = base.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values #getting all lines from all columns but 7

labelEncoder = LabelEncoder() #transforming the categorical data in numerical data
for i in range(X.shape[1]): #it will go through all the columns
    if X[:,i].dtype == "object": #ig the data is categorical
        X[:,i] = labelEncoder.fit_transform(X[:,i])

print("y: ", y)
print("X: ", X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = GaussianNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("predictions: ", predictions)

#METRICS:
accuracy = accuracy_score(y_test, predictions)
print("accuracy: ", accuracy)

precision = precision_score(y_test, predictions, average=None)
print("precision: ", precision)

recall = recall_score(y_test, predictions, average="weighted")
print("recall: ", recall)

f1 = f1_score(y_test, predictions, average="weighted")
print("f1 score: ", f1)

report = classification_report(y_test, predictions)
print("report: ", report)

confusion = ConfusionMatrix(model, classes=["None", "Severe", "Mild", "Moderate"])
confusion.fit(X_train, y_train)
confusion.score(X_test, y_test)
confusion.poof()

