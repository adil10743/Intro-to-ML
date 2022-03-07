#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:09:45 2020

@author: adilanees
"""
#I am using a linear regression model on the data and
#measuring its performance with mean absolute error.
#The result of the mean absolute error is 97695.03…
#As a pre-processing step I removed rows with house
#price in the top 2% of the data as I believed the large
#right hand tail to be having significant negative impact
#on the predictions that were outputted.
#The 1995-2015 file would not open in python on my laptop as the file was too large.
#Instead I have opted to use the 'Price Paid Data: 2014' which contained a more manageable amount of data
#2014 was chosen over other years as the closest year to 2015 will be the most similar data
#Clearly supervised learning as we are using labelled training examples
#I use a linear regression model on the 2014 data
#No continuous flow of data and data is not too large so batch learning should be fine.
#Mean Absolute Error performance meausure as RMSE is too sensitive to outliers and the data has a large right tail
import numpy as np
import pandas as pd


def reader(x):
    #To read in csv files and add column headers to the data
    #Returns two dataframes Data and Data_labels which represent the
    #independent and dependent variables
    
    File = pd.read_csv(x, header=None)
    
    File = File.drop(columns=[0,2,3,5,7,8,9,10,12,13,14,15], axis=1)
    #Deletes all the unnecessary columns
    
    File = File.rename(columns={1: "Price",4:"Type",6:"Duration",11:"Locality"})
    #Adds in the 4 column headers

    Data = File[File['Price'] < 985000]
    #Removes all rows with house price above £985,000
    #Originally done without this line but after displaying a histogram of the prices
    #I found a very large right tail. Data_labels.quantile(0.98) gave 985000. Therefore
    #I have removed the top 2% as they will have a large negative effect on the result
    #This leaves the data with still a large right tail but greatly reduced
    
    Data_labels = Data[["Price"]]
    Data = Data.drop(columns=["Price"],axis=1)
    #Independent variables go into Data
    #Dependent variable, Price, goes into Data_labels
    
    return Data, Data_labels


#The data needs to be prepared before it can be used for Machine Learning
#The imported modules will be used in preparing the data
    
from sklearn.preprocessing import OrdinalEncoder 
#To tranform Locality and London columns
#Lease Duration has two possible outputs L and F so can be turned to binary
#Whether the property is in London or not can also be put into binary

from sklearn.preprocessing import OneHotEncoder
#Property type has 5 categories so is best to one-hot encode it and leave in matrix form

from sklearn.compose import ColumnTransformer
#To transform all the data at once and leave it in array form

def Prepared(Data):
    Data = Data.replace({"LONDON":1.})
    #replaces every London with float 1
    Data = Data.replace({"Locality":r'^[A-Z]'},{"Locality":0.},regex=True)
    #replaces every other string in locality column with float 0
    
    full_pipeline = ColumnTransformer([("TypeCat",OneHotEncoder(),["Type"]),("DurationCat",OrdinalEncoder(),["Duration"]),("LocatCat",OrdinalEncoder(),["Locality"])])
    #Will transform all data
    Data_prepared = full_pipeline.fit_transform(Data)
    
    return Data_prepared
    
    
      
Train, Train_labels = reader("pp-2014.csv")
#This is the 2014 training data
Train_prepared = Prepared(Train)

#A view of the histogram can be seen
Hist = Train_labels.hist(bins=range(0,1000000,25000))


from sklearn.linear_model import LinearRegression
#import for the Machine Learning algorithm
lin_reg = LinearRegression()
lin_reg.fit(Train_prepared, Train_labels)
#I now have a working Linear Regression Model

Test, Test_labels = reader("pp-2015.csv")
#This is the 2015 test data
Test_prepared = Prepared(Test)



#My performance measure of mean absolute error
from sklearn.metrics import mean_absolute_error

def mae(Data_prepared,Data_labels):
    Data_predictions = lin_reg.predict(Data_prepared)
    lin_mae = mean_absolute_error(Data_labels,Data_predictions)
    return lin_mae


Prediction = lin_reg.predict(Test_prepared)

Performance15 = mae(Test_prepared, Test_labels)
#Gave result ~£97,700


