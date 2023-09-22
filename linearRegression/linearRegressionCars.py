import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns

base = pd.read_csv('files/mt_cars.csv')
base.head()

base = base.drop(['Unnamed'], axis=1)

corr = base.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f') #goo way to see the correlation between variables
plt.show()

#analyzing the correlation between the dependent variable mpg (the variable that we want to predict) and the other independent variables
column_pairs = [('mpg', 'cyl'), ('mpg', 'disp'), ('mpg', 'hp'), ('mpg', 'wt'), ('mpg', 'drat'), ('mpg', 'vs')]
n_plots = len(column_pairs)
fig, axies = plt.subplots(nrows=n_plots, ncols=1, figsize=(6,4 * n_plots))

for i, pair in enumerate(column_pairs):
    x_col, y_col = pair
    sns.scatterplot(x=x_col, y=y_col, data=base, ax=axies[i])
    axies[i].set_title(f'{x_col} vs {y_col}')

plt.tight_layout()
plt.show()

#model = sm.ols(formula='mpg ~ wt + disp + hp', data=base)
#model = sm.ols(formula='mpg ~ disp + cyl', data=base)
model = sm.ols(formula='mpg ~ drat + vs', data=base)
model = model.fit()

# The goal is to keep AIC and BIC as low as possible
#AIC = 156.6 and BIC = 162.5
print("model: ", model.summary())

residuals = model.resid
plt.hist(residuals, bins=20)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of residuals")
plt.show()

stats.probplot(residuals, dist='norm', plot=plt)
plt.title("Q-Q Residuals Plot")
plt.show()

stat, pval = stats.shapiro(residuals)
#if p <= 0.05, we reject the null hypothesis. In other words, the data is not normally distributed
print(f"Shapiro-Wilk statistics: {stat: .3f}, p-value: {pval: .3f}")