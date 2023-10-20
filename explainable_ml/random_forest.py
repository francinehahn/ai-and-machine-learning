import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import IPython
import numpy as np
import eli5
from eli5 import show_prediction
import shap
import lime.lime_tabular
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
import webbrowser

credit = pd.read_csv("files/Credit.csv")

forecasters = credit.iloc[:,:-1].values #all rows, all columns but the last one
class_good_or_bad = credit.iloc[:,-1].values #all rows and only the last column

label_encoder = LabelEncoder()
for i in range(forecasters.shape[1]):
    if forecasters[:,i].dtype == "object":
        forecasters[:,i] = label_encoder.fit_transform(forecasters[:,i])

gender = label_encoder.fit_transform(class_good_or_bad)

X_train, X_test, y_train, y_test = train_test_split(forecasters, class_good_or_bad, test_size=0.3)

model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)

#Lime
explain = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=list(credit)[0:20], class_names="class") #class is the name of the column
predict = lambda x: model.predict_proba(x).astype(float)

exp_lime = explain.explain_instance(X_test[0], predict, num_features=5)
exp_lime.save_to_file('lime_explanation.html')
webbrowser.open('lime_explanation.html')

#eli5
exp_eli5 = eli5.explain_weights(model, feature_names = list(credit)[0:20])

with open('eli5_explanation.html', 'w') as f:
    f.write(eli5.format_as_html(exp_eli5))
webbrowser.open('eli5_explanation.html')

#shap
X_train_2 = X_train.astype(float)
explainer = shap.Explainer(model, X_train_2)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=list(credit.columns)[:20], plot_type="bar")
shap.force_plot(explainer.expected_value[1], shap_values[1])
shap.initjs()

#interpret
set_visualize_provider(InlineProvider())
ebm = ExplainableBoostingClassifier(feature_names=list(credit)[0:20])
ebm.fit(X_train, y_train)
global_explanation = ebm.explain_global()
show(global_explanation)

