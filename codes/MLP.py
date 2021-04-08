# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:22:29 2021

@author: 61090034
"""
#%%
import pandas as pd
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#%% Load dataset 
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
sample = pd.read_csv('../data/sampleSubmission.csv')
weather_data = pd.read_csv('../data/weather.csv')

labels = train_data['WnvPresent'].values 

#%%
def drop_unnecessary_column(data: pd.DataFrame, colList: list):
    return data.drop(colList, axis = 1)

def label_encode_col(data: pd.DataFrame, colName: str):
    le = LabelEncoder()
    data[colName] = le.fit_transform(data[colName])
    return data
    
def one_encode_col(data: pd.DataFrame, colName: str):
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_data = pd.DataFrame(one_hot_encoder.fit_transform(data[[colName]]).toarray())

    data = data.join(encoded_data)
    return data

def date_to_second(data: pd.DataFrame):
    data['Second'] = 0
    data['Date'] = pd.Series(pd.to_datetime(data['Date']))
    for i in range(len(data)):
        data['Second'].iloc[i] = data['Date'].iloc[i].timestamp()
    return data

def rename_encoded_columns(data: pd.DataFrame, name: str, amount: int):
    for i in range(amount):
        data = data.rename(columns={i: name + str(i)})
    return data

#%% 
class WeatherData():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.drop_col(['SnowFall', 'Water1', 'Depth', 'CodeSum',\
                 'Sunset', 'Sunrise'])
        self.cast_column(['Tmax', 'Tmin'], int)
        self.filling_Tavg()
        self.cast_column(['Tavg'], int)
        self.filling_others()
        self.drop_col(['Tmax', 'Tmin'])
        self.merge_station()
        self.cast_column(['Date'], 'datetime64')
    
    def cast_column(self, colName: list, typ: type):
        for col in colName:
            self.data[col] = self.data[col].astype(typ)
        
    def filling_Tavg(self):
        for i in range(len(self.data)):
            temp_data = self.data.iloc[i]
            if (temp_data['Tavg'] == 'M'):
                self.data['Tavg'].iloc[i] = math.floor((self.data['Tmax'].iloc[i] +\
                    self.data['Tmin'].iloc[i] ) / 2)
                    
    def filling_others(self):
        self.data = self.data.replace('M', -1)
        self.data = self.data.replace('T', -1)
        self.data = self.data.replace(' T', -1)
        self.data = self.data.replace('  T', -1)
         
    def drop_col(self, colName: list):
        self.data = drop_unnecessary_column(self.data, colName)
        
    def merge_station(self):
        weather_station1 = self.data[self.data['Station']==1]
        weather_station2 = self.data[self.data['Station']==2]
        weather_station1 = weather_station1.drop('Station', axis=1)
        weather_station2 = weather_station2.drop('Station', axis=1)
        self.data = weather_station1.merge(weather_station2, on='Date')
            
    def get_weather_data(self):
        return self.data
    
#%%
class MainData():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data = label_encode_col(self.data, 'Species')
        self.data = one_encode_col(self.data, 'Species')
        self.data = date_to_second(self.data)
        
    def get_data(self):
        return self.data
#%%
trainClass = MainData(train_data)
testClass = MainData(test_data)

train = trainClass.get_data()
test = testClass.get_data()

#%%   Special column from Unidentified Species -> set its value to 0
train[7] = 0

#%%
train = rename_encoded_columns(train, "SpeciesEncoded", 8)
test = rename_encoded_columns(test, "SpeciesEncoded", 8)

#%%
weather = WeatherData(weather_data).get_weather_data()
#%%
train = train.merge(weather, on = 'Date')
test = test.merge(weather, on = 'Date')


#%%
train = drop_unnecessary_column(train, ['Date', 'Address', 'Species', 'Block',\
                                'Street', 'Trap', 'WnvPresent', 'NumMosquitos',
                                'AddressNumberAndStreet', 'AddressAccuracy'])
    
test = drop_unnecessary_column(test, ['Id', 'Date', 'Address', 'Species',\
                                       'Block', 'Street', 'Trap',\
                                    'AddressNumberAndStreet', 'AddressAccuracy'])

#%%    
min_max_scaler = MinMaxScaler().fit(train)
min_max_scaler2 = MinMaxScaler().fit(test)

X = min_max_scaler.transform(train)

test_predicted_data = min_max_scaler2.transform(test)

#%%
X_train, X_test, y_train, y_test = train_test_split(X, labels, stratify=labels,\
                                                   test_size=0.3 ,random_state = 1)
    
clf = MLPClassifier(random_state=1, solver='adam',\
                    activation='relu')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
#%%
predictions = clf.predict_proba(test_predicted_data)[:,1]
sample['WnvPresent'] = predictions
sample.to_csv('../WnvPrediction.csv', index=False)
