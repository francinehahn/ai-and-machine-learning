#Local Outlier Factor
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)
data_matrix = np.random.rand(100, 2) #matrix with 100 lines and 2 columns

print(data_matrix)

data_matrix[0,0] = 1000 #adding an outlier
data_matrix = data_matrix.reshape(-1,1)

lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
list = lof.fit_predict(data_matrix)

print(list) #only the first will be -1, because it is an outlier

positions = np.where(list == -1)
print(data_matrix[positions])
