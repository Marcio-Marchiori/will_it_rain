import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

# Droping columns that are leaking data to the model.
rain_data.drop(columns=['modelo_vigente','amountOfRain'],inplace=True)
pd.get_dummies(rain_data,drop_first=True)

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

# Merging the wind and rain tables
rain_wind = pd.merge(rain_data,wind_table,how='inner',left_on=['date','location'],right_on=['date','location'])
rain_wind = pd.get_dummies(rain_wind,drop_first=True)


# Removing the data the doesn't make sense, you can't have 9/8 of the sky obscured
rain_wind.drop(rain_wind[rain_wind['cloud9am']>8].index,inplace=True)
rain_wind.drop(rain_wind[rain_wind['cloud3pm']>8].index,inplace=True)

# Splitting test and training data
X = rain_wind.loc[:,rain_wind.columns!='raintomorrow_Yes']
y = rain_wind['raintomorrow_Yes']

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

