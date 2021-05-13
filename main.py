import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from imblearn.over_sampling import SMOTE
import datetime

list_names = ['rain_data_aus.csv','wind_table_01.csv','wind_table_02.csv','wind_table_03.csv','wind_table_04.csv','wind_table_05.csv','wind_table_06.csv','wind_table_07.csv','wind_table_08.csv']

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

def model():
    model = Sequential([
        Dense(units=120,input_shape=(111,),activation='relu'),
        Dense(units=16,activation='relu'),
        Dense(units=2,activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.000005), loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
    model.fit(x=X_train, y=y_train, batch_size=100, epochs=2000,validation_split=0.15, verbose=2, use_multiprocessing=True)

'''
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


df = pd.DataFrame(pca.components_.T)

raining = pd.merge(rain_data,wind_table,how='inner',left_on=['date','location'],right_on=['date','location'])

loca = {}
slice_X_train = {}
slice_y_train = {}
slice_X_test = {}
slice_y_test = {}

for x in raining['location'].unique():
    loca[x] = raining[raining['location']==x]
    loca[x] = raining.drop(columns=['location'])
    loca[x] = pd.get_dummies(loca[x],drop_first=True)
    loca[x].dropna(inplace=True)
    Z = loca[x].loc[:,loca[x].columns!='raintomorrow_Yes']
    y = loca[x]['raintomorrow_Yes']

    slice_X_train[x], slice_X_test[x], slice_y_train[x], slice_y_test[x] = train_test_split(Z,y)

models_dict = {}
scaler_dict = {}
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],True)

for x in raining['location'].unique():
    try:
        scaler_dict[x] = MinMaxScaler()
        slice_X_train[x].loc[:,list_columns_to_use] = scaler_dict[x].fit_transform(slice_X_train[x][list_columns_to_use])
        slice_X_test[x].loc[:,list_columns_to_use] = scaler_dict[x].transform(slice_X_test[x][list_columns_to_use])
        
        oversample = SMOTE()
        slice_X_train[x],slice_y_train[x] = oversample.fit_resample(slice_X_train[x],slice_y_train[x])

        model = Sequential([
            Dense(units=8,input_shape=(63,),activation='relu'),
            Dense(units=2,activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.000005), loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
        models_dict[x] = model.fit(x=slice_X_train[x], y=slice_y_train[x], batch_size=20, epochs=200,validation_split=0.15, verbose=2, use_multiprocessing=True)
    except:
        print('Error on ',x)
'''



'''
model = Sequential([
    Dense(units=120,input_shape=(111,),activation='relu'),
    Dense(units=16,activation='relu'),
    Dense(units=2,activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.000005), loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
model.fit(x=X_train, y=y_train, batch_size=100, epochs=2000,validation_split=0.15, verbose=2, use_multiprocessing=True)
'''