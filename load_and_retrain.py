import pickle

import keras.models
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential

df = pd.read_csv("DrawDataFile23.csv", header=0)

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