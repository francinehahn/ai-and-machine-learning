import h2o
from h2o.automl import H2OAutoML

h2o.init()

data = h2o.import_file("files/insurance.csv")
print(data)

data = data.drop("Unnamed: 0")

train, test = data.split_frame(ratios=[0.7], seed=0)

model_automl = H2OAutoML(max_runtime_secs=180, sort_metric="AUTO")

print(model_automl.train(y="Accident", training_frame=train))

ranking = model_automl.leaderboard
ranking = ranking.as_data_frame()
print(ranking)

predict = model_automl.leader.predict(test)
print(predict)
