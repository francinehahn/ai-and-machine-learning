import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

spam = pd.read_csv("files/spam.csv")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(spam["Category"])

messages = spam["Message"].values
X_train, X_test, y_train, y_test = train_test_split(messages, y, test_size=0.3)

token = Tokenizer(num_words=1000)
token.fit_on_texts(X_train)
X_train = token.texts_to_sequences(X_train)
X_test = token.texts_to_sequences(X_test)

#making sure all the matrix are the same size
X_train = pad_sequences(X_train, padding="post", maxlen=500)
X_test = pad_sequences(X_test, padding="post", maxlen=500)

#token.word_index will show the vocabulary
number_of_words = len(token.word_index)

model = Sequential()
model.add(Embedding(input_dim=number_of_words, output_dim=50, input_length=500)) #input_dim is the number of neurons in the start layer
model.add(Flatten())
model.add(Dense(units=10, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train, epochs=20, batch_size=10, verbose=True, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

new_prediciton = model.predict(X_test)
print("Prediction: ", new_prediciton)

prev = (new_prediciton > 0.5)
cm = confusion_matrix(y_test, prev)
print(cm)
