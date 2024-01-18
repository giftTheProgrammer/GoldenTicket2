import pickle

import keras.models
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential

df = pd.read_csv("DrawDataFile27.csv", header=0)

#my_chosen_sample = df.loc[10:17]
my_chosen_sample = df.copy()
print(my_chosen_sample)
print("******************************************************************************\n")

df1 = df.copy()

my_chosen_sample.drop(['Created_at'], axis=1, inplace=True)

tester_draw = my_chosen_sample.tail(1)
#my_chosen_sample.drop(17, inplace=True)
print(my_chosen_sample)
print("******************************************************************************\n")

to_predict = np.array(my_chosen_sample)
print(to_predict)
print("******************************************************************************\n")

scaler = StandardScaler().fit(my_chosen_sample.values)

scaled_to_predict = scaler.transform(to_predict)
print(scaled_to_predict)
print("******************************************************************************\n")

#loaded_model = pickle.load(open('finalized_model.sav', 'rb')) - We are loading the trained and saved model
my_tf_saved_model = keras.models.load_model('C:/Users/GiftMoletsane/PycharmProjects/GoldenTicket2/trained_model4')

my_tf_saved_model.summary()
print("******************************************************************************\n")

y_pred = my_tf_saved_model.predict(np.array([to_predict]))
# y_pred = loaded_model.predict(np.array([to_predict]))
print("The predicted numbers without rounding up or down:", scaler.inverse_transform(y_pred).astype(int)[0])
print("The predicted numbers rounded up:", scaler.inverse_transform(y_pred).astype(int)[0]+1)
print("The predicted numbers rounded down:", scaler.inverse_transform(y_pred).astype(int)[0]-1)