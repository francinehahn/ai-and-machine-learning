import numpy as np
import scipy.stats as stats

data = np.array([30, 29, 30, 31, 100, 29, 28, 30, 300, 31, 32, 33])
z_scores = stats.zscore(data)
print(z_scores)

anomaly = data[np.abs(z_scores) > 3]
print(anomaly)