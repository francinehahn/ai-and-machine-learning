#Long Short-Term Memory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataset_train = pd.read_csv('files/Salestrain.csv')
plt.plot(dataset_train, color='blue', label='Vendas')
plt.title('Vendas')
plt.xlabel('Tempo')
plt.ylabel('Vendas')
plt.legend()
plt.show()

sc = MinMaxScaler(feature_range=(0, 1))
trainning_set_scaled = sc.fit_transform(dataset_train)

X_train = []
y_train = []

for i in range(90, len(trainning_set_scaled)):
    data = trainning_set_scaled[i-90:i, 0]
    X_train.append(data)
    y_train.append(trainning_set_scaled[i,0])
    
X_train = np.array(X_train.reshape(-1,90,1))
y_train = np.array(y_train)

model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=300, batch_size=1)

dataset_test = pd.read_csv('files/Salestest.csv')

train_values = dataset_train['data'].values
test_values = dataset_test['data'].values
total_values = np.concatenate((train_values, test_values), axis=0)
time_index = range(len(total_values))

plt.plot(time_index[:len(train_values)], color='blue', label='Vendas - Treinamento')
plt.plot(time_index[len(test_values):], color='red', label='Vendas - Teste')
plt.title('Vendas')
plt.xlabel('Tempo')
plt.ylabel('Vendas')
plt.legend()
plt.show()

dataset_test_anomalies = dataset_test.copy()
dataset_test_anomalies.loc[:9, 'data'] = 90
dataset_test_anomalies.loc[10:34, 'data'] = np.random.uniform(100,200,size=(25,))
dataset_test_anomalies.loc[35:, 'data'] = 90

plt.plot(dataset_test, color='blue', label='Vendas')
plt.plot(dataset_test_anomalies, color='red', label='Vendas com anomalias')
plt.title('Vendas')
plt.xlabel('Tempo')
plt.ylabel('Vendas')
plt.legend()
plt.show()

dataset_total = pd.concat((dataset_train['data'], dataset_test['data']), axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test-90):]
inputs = pd.DataFrame(inputs, columns=['data'])
inputs = sc.transform(inputs)

dataset_total_anomalies = pd.concat((dataset_train['data'], dataset_test_anomalies['data']), axis=0)
inputs_anomalies = dataset_total_anomalies[len(dataset_total_anomalies)-len(dataset_test_anomalies-90):]
inputs_anomalies = pd.DataFrame(inputs_anomalies, columns=['data'])
inputs_anomalies = sc.transform(inputs_anomalies)

X_test = []
X_test_anomalies = []
for i in range(90, len(inputs)):
    X_test.append(inputs[i-90:i,0])
    X_test_anomalies.append(inputs_anomalies[i-90:i,0])
    
X_test, X_test_anomalies = np.array(X_test), np.array(X_test_anomalies)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test_anomalies = np.reshape(X_test_anomalies, (X_test_anomalies.shape[0], X_test_anomalies.shape[1], 1))

prediced_sales = model.predict(X_test)
prediced_sales = sc.inverse_transform(prediced_sales)
prediced_sales_anomalies = model.predict(X_test_anomalies)
prediced_sales_anomalies = sc.inverse_transform(prediced_sales_anomalies)

mean_squared_error_test = mean_squared_error(test_values, prediced_sales)
mean_squared_error_anomalies = mean_squared_error(test_values, prediced_sales_anomalies)

print(f'MSE normal data: ', mean_squared_error_test)
print(f'MSE data with anomalies: ', mean_squared_error_anomalies)

plt.plot(test_values, color='blue', label='Valores reais')
plt.plot(prediced_sales_anomalies, colo='red', label='Previsões com anomalias')
plt.plot(prediced_sales, color='green', label='Previsões')
plt.title('Previsões com anomalias, sem anomalias e valores reais')
plt.xlabel('Tempo')
plt.ylabel('Vendas')
plt.legend()
plt.show()


