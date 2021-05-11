import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import datetime
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


# Loading the data
rain_data = pd.DataFrame(pd.read_csv('data/rain_data_aus.csv'))
wind_table_01 = pd.DataFrame(pd.read_csv('data/wind_table_01.csv'))
wind_table_02 = pd.DataFrame(pd.read_csv('data/wind_table_02.csv'))
wind_table_03 = pd.DataFrame(pd.read_csv('data/wind_table_03.csv'))
wind_table_04 = pd.DataFrame(pd.read_csv('data/wind_table_04.csv'))
wind_table_05 = pd.DataFrame(pd.read_csv('data/wind_table_05.csv'))
wind_table_06 = pd.DataFrame(pd.read_csv('data/wind_table_06.csv'))
wind_table_07 = pd.DataFrame(pd.read_csv('data/wind_table_07.csv'))
wind_table_08 = pd.DataFrame(pd.read_csv('data/wind_table_08.csv'))
# wind_table_07.equals(wind_table_08) returns true, so wind_table_08 is discarded to avoid redundancy

# Droping columns that are leaking data to the model and that have over 30% of NaN data.
rain_data.drop(columns=['modelo_vigente','amountOfRain','sunshine','evaporation','cloud3pm','cloud9am'],inplace=True)
pd.get_dummies(rain_data)

# Renaming all the remaining 7 table's columns 
wind_table_02.columns = wind_table_01.columns
wind_table_03.columns = wind_table_01.columns
wind_table_04.columns = wind_table_01.columns
wind_table_05.columns = wind_table_01.columns
wind_table_06.columns = wind_table_01.columns
wind_table_07.columns = wind_table_01.columns

# Concatenate all wind tables tables into one.
wind_table = pd.concat([wind_table_01,wind_table_02,wind_table_03,wind_table_04,wind_table_05,wind_table_06,wind_table_07])

# Converting the column date to the actual datetime format.
wind_table['date'] = pd.to_datetime(wind_table['date'])
rain_data['date'] = pd.to_datetime(rain_data['date'])
wind_table['date'] = wind_table['date'].apply(lambda x: datetime.datetime.timestamp(x))
rain_data['date'] = rain_data['date'].apply(lambda x: datetime.datetime.timestamp(x))

# Merging the wind and rain tables and dropping the ones that still have NaN
rain_wind = pd.merge(rain_data,wind_table,how='inner',left_on=['date','location'],right_on=['date','location'])
rain_wind = pd.get_dummies(rain_wind,drop_first=True)
rain_wind.dropna(inplace=True)

# Splitting test and training data
X = rain_wind.loc[:,rain_wind.columns!='raintomorrow_Yes']
y = rain_wind['raintomorrow_Yes']


# Slicing the training/testing data and correcting the Y oversampling.
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
oversample = SMOTE()
X_train,y_train = oversample.fit_resample(X_train,y_train)

# Preprocessing the data so it will better fit the model.
list_columns_to_use = ['date', 'mintemp', 'maxtemp', 'rainfall', 'humidity9am', 'humidity3pm', 'pressure9am', 'pressure3pm', 'temp9am', 'temp3pm', 'temp', 'humidity', 'precipitation3pm', 'precipitation9am', 'wind_gustspeed', 'wind_speed9am', 'wind_speed3pm',]
scaler = MinMaxScaler()
X_train.loc[:,list_columns_to_use] = scaler.fit_transform(X_train[list_columns_to_use])
X_test.loc[:,list_columns_to_use] = scaler.transform(X_test[list_columns_to_use])

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],True)

'''
pca = PCA(n_components=100)
pca.fit(X_train.T)

df = pd.DataFrame(pca.components_.T)
'''

model = Sequential([
    Dense(units=120,input_shape=(100,),activation='relu'),
    Dense(units=32,activation='relu'),
    Dense(units=1,activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy',  metrics=['accuracy'])
model.fit(x=X_train, y=y_train, batch_size=200, epochs=50,validation_split=0.15, verbose=2, use_multiprocessing=True)