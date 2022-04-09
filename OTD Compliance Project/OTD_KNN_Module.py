#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics

# the custom scaler class 
class CustomScaler(BaseEstimator,TransformerMixin): 
    
    def __init__(self,columns):
        
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# create the special class that we are going to use from here on to predict new data
class OTD_Compliance_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model1','rb') as model_file, open('scaler1', 'rb') as scaler_file:
                self.KNN = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file, encoding= 'unicode_escape'):
            
            # import the data
            df = pd.read_csv(data_file,delimiter=',', encoding= 'unicode_escape')
           
            # drop the Necessary columns
            df = df.drop(['Transport Name', 'Customer Name', 'Shipment Number', 'Ship To Address', 'Truck Plate', 
                          'Product Name', 'Logon Number', 'Driver Name', 'Region', 'First Weighing date', 
                          'Arrival Date And Time','Ring', 'Dispatch Date', 'Truck Plate.1', 'OTD (IN %)'], axis = 1)
           
            # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
            df['Delivery Time (In Hours)'] = 'NaN'
            
            # encode all categorical variables using map function
            df['Night Driving'] = df['Night Driving'].map({'YES':1, 'NO':0})
            df['Customer Type'] = df['Customer Type'].map({'Key':1, 'Trade':0})
            df['Safety Violation'] = df['Safety Violation'].map({'NO VIOLATION':0, 'VIOLATED':1})
            
            # create a separate dataframe, containing dummy values for ALL 'Shipping Point'
            Shipping_Point_column = pd.get_dummies(df['Shipping Point'])
            
            # to avoid multicollinearity, drop the 'Shipping Point' column from df
            df = df.drop(['Shipping Point'], axis = 1)
            
            # concatenate df and the Shipping point dummies
            df = pd.concat([df, Shipping_Point_column], axis = 1)
            
            # reorder and rename newly created columns for consistency of data frame
            df = df.loc[:, ['ABA DEPOT', 'ENUGU DEPOT', 'MFA', 'Owerri Depot', 'PORT HARCOURT 2 DEPOT','PORT HARCOUT DEPOT', 
                      'UYO DEPOT', 'Dispatch Date1', 'Night Driving', 'Customer Type', 'Quantity', 'Safety Violation', 
                      'Delivery Time (In Hours)']]
            
            # re-order the columns in df
            df.rename(columns={'ABA DEPOT' : 'Aba Depot', 'ENUGU DEPOT': 'Enugu Depot', 'MFA' : 'MFA', 
                               'Owerri Depot' : 'Owerri Depot','PORT HARCOURT 2 DEPOT' : 'Port Harcourt II Depot',
                               'PORT HARCOUT DEPOT' : 'Port Harcourt Depot','UYO DEPOT' : 'Uyo Depot', 
                               'Dispatch Date1' : 'Dispatch Date', 'Night Driving' : 'Night Driving', 
                               'Customer Type' : 'Customer Type', 'Quantity' : 'Quantity', 
                               'Safety Violation' : 'Safety Violation', 
                               'Delivery Time (In Hours)' : 'Delivery Time'}, inplace=True)
            
            # convert the 'Date' column into datetime
            df['Dispatch Date'] = pd.to_datetime(df['Dispatch Date'])

            # create a list with month values retrieved from the 'Date' column
            list_months = []
            for i in range(df.shape[0]):
                list_months.append(df['Dispatch Date'][i].month)

            # insert the values in a new column in df, called 'Month Value'
            df['Month Value'] = list_months

            # create a new feature called 'Day of the Week'
            df['Day of the week'] = df['Dispatch Date'].apply(lambda x: x.weekday())


            # drop the 'Date' column from df
            df = df.drop(['Dispatch Date'], axis = 1)

            # re-order the columns in df
            column_names_upd = ['Aba Depot', 'Enugu Depot', 'MFA', 'Owerri Depot', 'Port Harcourt II Depot', 
                                'Port Harcourt Depot', 'Uyo Depot', 'Month Value', 'Day of the week','Night Driving', 
                                'Customer Type', 'Quantity', 'Safety Violation', 'Delivery Time']
            df = df[column_names_upd]

            # replace the NaN values
            df = df.fillna(value=0)

            # drop the original absenteeism time
            df = df.drop(['Delivery Time'],axis=1)
    
                      
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)
    
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.KNN.predict_proba(self.data)[:,1]
                return pred
        
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.KNN.predict(self.data)
                return pred_outputs
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.KNN.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.KNN.predict(self.data)
                return self.preprocessed_data


# In[ ]:




