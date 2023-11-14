import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_iris

data = load_iris().data
model = IsolationForest(contamination=0.01) #the higher the contamination, more outliers will be found
model.fit(data)

predictions = model.predict(data)
print(predictions)

print(data[predictions == -1]) #get all the data where predictions = -1