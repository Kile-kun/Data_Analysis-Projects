#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Necessary libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

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
class Purchased_Bike_model():
    
    def __init__(self, model_file, scaler_file):
        with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
    # take a data file (*.csv) and preprocess it in the same way as done in the preprocess algorithm            
    def load_and_clean_data(self, data_file):
        
            # import the data
            df = pd.read_csv(data_file,delimiter=',')
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            # drop the 'ID' column
            df = df.drop(['ID'], axis = 1)
            # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
            df['Purchased Bike'] = 'NaN'
            # encode all categorical variables using map function
            df['Marital Status'] = df['Marital Status'].map({'M':1, 'S':0})
            df['Gender'] = df['Gender'].map({'M':1, 'F':0})
            df['Education'] = df['Education'].map({'Partial High School':0,'High School':1,'Partial College':2, 'Bachelors':3, 'Graduate Degree':4})
            df['Home Owner'] = df['Home Owner'].map({'Yes':1, 'No':0})
            df['Commute Distance'] = df['Commute Distance'].map({'0-1 Miles':0, '1-2 Miles':1, '2-5 Miles':2, '5-10 Miles':3, '10+ Miles':4})
            df['Region'] = df['Region'].map({'Europe':0, 'Pacific':1, 'North America':2})
            
            # create a separate dataframe, containing dummy values for ALL occupation
            Occupation_column = pd.get_dummies(df['Occupation'])
            
            # drop the real 'occupation' column to make way for it's dummies
            df = df.drop(['Occupation'], axis = 1)
            
            # concatenate 'df' and 'occupation'dummies
            df = pd.concat([df, Occupation_column], axis = 1)
            column_names = ['Marital Status', 'Gender', 'Income', 'Children', 'Education', 'Home Owner', 'Cars', 
                            'Commute Distance', 'Region', 'Age', 'Purchased Bike', 'Clerical', 'Management', 'Manual', 
                            'Professional', 'Skilled Manual']
            df.columns = column_names
            
            # re-arrange the columns to follow the initial order
            column_names_reordered = ['Marital Status', 'Gender', 'Income', 'Children', 'Education', 'Clerical', 
                                      'Management', 'Manual', 'Professional', 'Skilled Manual','Home Owner', 'Cars',
                                      'Commute Distance', 'Region', 'Age', 'Purchased Bike']
            df = df[column_names_reordered]
            
            # replace the NaN values
            df = df.fillna(value=0)
            
            # drop the 'Purchase Bike' column
            df = df.drop(['Purchased Bike'],axis=1)
            
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)
            
    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
    
    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
            
    # predict the outputs and the probabilities and add columns with these values at the end of the new data
    def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data

