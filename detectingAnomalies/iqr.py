import numpy as np

dataset = [1,2,3,4,5,6,7,8,9,100]

quartil1 = np.percentile(dataset, 25) #25%
quartil3 = np.percentile(dataset, 75) #75%
iqr = quartil3 - quartil1

lower_boundary = quartil1 - (1.5 * iqr)
upper_boundary = quartil3 + (1.5 * iqr)

outliers = []
for value in dataset:
    if value < lower_boundary or value > upper_boundary:
        outliers.append(value)

print(outliers)
