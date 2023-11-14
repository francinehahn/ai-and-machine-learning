import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC


credit = pd.read_csv("files/credit_simple.csv", sep=";")
count = credit.groupby(["CLASSE"]).size()
print(count) #now we have 700 good payers and 300 bad payers -> we need to balance this by increasing the number of bad payers to 700

y = credit["CLASSE"].values
X = credit.iloc[:,:-1].values

label_encoder = LabelEncoder()
for i in range(X.shape[1]):
    if X[:,i].dtype == "object":
        X[:,i] = label_encoder.fit_transform(X[:,i])

sm = SMOTENC(random_state=0, categorical_features=[3,5,6]) #3="outros planos de pgto", 5="estado civil", 6="proposito"
X_res, y_res = sm.fit_resample(X, y)

print(X_res)
print(y_res)

df = pd.DataFrame({"CLASSE": y_res})
print(df.value_counts()) #now, we have 700 good payers and 700 bad payers

