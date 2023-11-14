import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error

(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

X_test_noisy = X_test + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_test_noisy = np.clip(X_test_noisy, 0.0, 1.0)

inputs = Input(shape=(784,))
encoder = Dense(32, activation='relu')(inputs)
decoder = Dense(784, activation='sigmoid')(encoder)
autoencoder = Model(inputs, decoder)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=100, batch_size=256, shuffle=True)

test_normal_decoded = autoencoder.predict(X_test)
test_anomalies_decoded = autoencoder.predict(X_test_noisy)

mean_squared_error_normal = mean_squared_error(X_test, test_normal_decoded)
mean_squared_error_anomalies = mean_squared_error(X_test_noisy, test_anomalies_decoded)

print(mean_squared_error_normal) #0.009
print(mean_squared_error_anomalies) #0.16

