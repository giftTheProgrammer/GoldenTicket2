import pickle
import joblib
import os.path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mse
from tensorflow.python.keras.layers.core import Flatten

df = pd.read_csv("DrawDataFile07.csv", header=0)

print(df.head())
print("******************************************************************\n")
print(df.tail())
print("******************************************************************\n")
print(df.shape)

df.info()
print("******************************************************************\n")

print(df.describe())
print("******************************************************************************\n")
df.drop(['Created_at'], axis=1, inplace=True)
print(df.head())
print("******************************************************************************\n")

scaler = StandardScaler().fit(df.values)
transformed_dataset = scaler.transform(df.values)
transformed_df = pd.DataFrame(data=transformed_dataset,index=df.index)

print(transformed_df.head())
print("******************************************************************************\n")

# All our games
number_of_rows = df.values.shape[0]
print(number_of_rows)
print("******************************************************************************\n")

# Amount of games we need to take into consideration for a prediction
window_length = 10
print(window_length)
print("******************************************************************************\n")

# Balls counts
number_of_features = df.values.shape[1]
print(number_of_features)
print("******************************************************************************\n")

X = np.empty([number_of_rows - window_length, window_length, number_of_features],
            dtype=float)

y = np.empty([number_of_rows - window_length, number_of_features], dtype=float)

for i in range(0, number_of_rows - window_length):
    X[i] = transformed_df.iloc[i: i+window_length, 0: number_of_features]
    y[i] = transformed_df.iloc[i+window_length: i+window_length+1, 0: number_of_features]

print(X.shape)
print("******************************\n")
print(y.shape)
print("******************************\n")
print(X[0])
print("******************************************************************************\n")
print(y[0])
print("******************************************************************************\n")
print(X[1])
print("******************************************************************************\n")
print(y[1])
print("******************************************************************************\n")

model = Sequential()
model.add(Bidirectional(LSTM(240, input_shape=(window_length, number_of_features),
                             return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, input_shape=(window_length, number_of_features),
                             return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, input_shape=(window_length, number_of_features),
                             return_sequences=True)))
model.add(Bidirectional(LSTM(240, input_shape=(window_length, number_of_features),
                             return_sequences=True)))
model.add(Dropout(0.2))
#model.add(Flatten())
model.add(Dense(42))
model.add(Dense(number_of_features))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])

model.fit(x=X, y=y, batch_size=10, epochs=700, verbose=2)

print("******************************************************************************\n")

to_predict = df.tail(11)
print(to_predict)
print("******************************************************************************\n")

to_predict.drop([to_predict.index[-1]], axis=0, inplace=True)
print(to_predict)
print("******************************************************************************\n")

prediction = df.tail(1)
print(prediction)
print("******************************************************************************\n")

to_predict = np.array(to_predict)
print(to_predict)
print("******************************************************************************\n")

scaled_to_predict = scaler.transform(to_predict)
print(scaled_to_predict)
print("******************************************************************************\n")

y_pred = model.predict(np.array([to_predict]))
print("The predicted numbers without rounding up or down:", scaler.inverse_transform(y_pred).astype(int)[0])

# save the model to disk
model.save('C:/Users/GiftMoletsane/PycharmProjects/GoldenTicket2/trained_model4')

#filename = "finalized_model.keras"

#if os.path.isfile("C:/Users/GiftMoletsane/PycharmProjects/GoldenTicket2/finalized_model.h5") is False:
 #   model.save("C:/Users/GiftMoletsane/PycharmProjects/GoldenTicket2/finalized_model.h5")
#pickle.dump(model, open(filename, 'wb'))
# joblib.dump(model, filename)

#with open(filename, 'wb') as file:
#    pickle.dump(model, file)