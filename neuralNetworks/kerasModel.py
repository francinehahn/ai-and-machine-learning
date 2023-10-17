import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout

df = pd.read_csv("files/Churn_treino.csv", sep=";")
X = df.drop("Exited", axis=1)
y = df["Exited"]

stardard_scaler = StandardScaler()
numerical = X.select_dtypes(include=["int64", "float64"]).columns
X[numerical] = stardard_scaler.fit_transform(X[numerical])

label_encoder = LabelEncoder()
categorical = X.select_dtypes(include="object").columns

for col in categorical:
    X[col] = label_encoder.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = Sequential()
model.add(Dense(units=64, activation="relu", input_dim=X_train.shape[1]))
model.add(Dropout(0.4))
model.add(Dense(units=32, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, batch_size=32)

predictions = model.predict(X_test)
y_pred = (predictions > 0.5).astype("int32")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

f1 = f1_score(y_test, y_pred)
print("F1 score: ", f1)

recall = recall_score(y_test, y_pred)
print("Recall score: ", recall)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ", cm)
