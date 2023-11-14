from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from tpot import TPOTClassifier

data = load_iris()
X = data["data"]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#genetis algorithms
tpot = TPOTClassifier(
    generations=10, 
    population_size=100, 
    offspring_size=100, 
    mutation_rate=0.9, 
    crossover_rate=0.1, 
    scoring="accuracy",
    max_time_mins=2,
    random_state=0,
    early_stop=False,
    verbosity=2
)

tpot.fit(X_train, y_train)
print("Best model: ", tpot.fitted_pipeline_)
print(tpot.evaluated_individuals_)

predict = tpot.predict(X_test)
accuracy = accuracy_score(predict, y_test)
print("accuracy: ", accuracy)
