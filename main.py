import pandas as pd
import numpy as np


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