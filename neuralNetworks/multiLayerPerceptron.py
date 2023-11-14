from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"], test_size=0.3, random_state=0)

model = MLPClassifier(
    verbose=True,
    hidden_layer_sizes=(5,4), #this means 2 ocult layers (one with 5 neurons and another with 4 neurons)
    activation="relu",
    batch_size=20,
    learning_rate="adaptive",
    momentum=0.9,
    early_stopping=False,
    max_iter=1000,
    random_state=10
)
model.fit(X_train, y_train)

plt.plot(model.loss_curve_)
plt.xlabel("Iterações")
plt.ylabel("Valor de loss")
plt.title("Loss")
plt.show()

predictions = model.predict(X_test)
print(predictions)

accuracy = accuracy_score(y_test, predictions)
print(accuracy)
