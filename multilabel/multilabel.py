from skmultilearn.adapt import MLARAM
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
import pandas as pd

music = pd.read_csv("files/Musica.csv")

music_class = music.iloc[:, 0:6].values
prevision = music.iloc[:,7:78].values

X_train, X_test, y_train, y_test = train_test_split(prevision, music_class, test_size=0.3, random_state=0)

ann = MLARAM()
ann.fit(X_train, y_train)

predictions = ann.predict(X_test)
print("MLARAM: ", hamming_loss(y_test, predictions))

binary_relevance = BinaryRelevance(classifier=SVC()) #Best result
binary_relevance.fit(X_train, y_train)
predictions = binary_relevance.predict(X_test)
print("Binary relevance: ", hamming_loss(y_test, predictions))

classifier_chain = ClassifierChain(classifier=SVC())
classifier_chain.fit(X_train, y_train)
predictions = classifier_chain.predict(X_test)
print("Classifier chain: ", hamming_loss(y_test, predictions))

label_powerset = LabelPowerset(classifier=SVC())
label_powerset.fit(X_train, y_train)
predictions = label_powerset.predict(X_test)
print("Label powerset: ", hamming_loss(y_test, predictions))

