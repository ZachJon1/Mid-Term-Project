#!/usr/bin/env python
# coding: utf-8

#import pandas as pd and numpy as np


import pandas as pd
import numpy as np


#import matlab and seaborn plotting libraries, activate inline plotting using %inline

import seaborn as sns
from matplotlib import pyplot as plt

#Source of the data on kaggle, more information can be obtained from this link: https://www.kaggle.com/sootersaalu/amazon-top-50-bestselling-books-2009-2019

#Use pandas to read 'csv' file of the data and store it to a variable called 'df'

df = pd.read_csv('data_kaggle.csv')

##Exploratory Data Analysis

#inspect headers and rows of the data


#inspect the data types for each column  

df.info()


# Before we can use our data, its important to have column names of similar form, remove spaces in headers and check data for spaces and special characters that may hinder accessing the data or hinder analysis
#change all headers to lower case and replace spaces with '_'

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

#identify data of type object and change the entries to lower case, replace spaces with '_'

df.dtypes == 'object'
df

#Check the data for null entries

df.describe()

df.isnull().sum()

# The data has 4 columns of numeric values and 3 columns of categorical type. 
# Data has no missing values
#install and use missingno library to visualize missing data

import missingno

missingno.heatmap(df, figsize=(5,5), fontsize=12 )

missingno.bar(df, color='g', figsize=(5,5), fontsize=12 )


# Conclusion: There are no missing values in our dataframe
#Check for duplicates in categorical columns

data_type_cat_col = list(df.select_dtypes(exclude=('int64', 'float64')).columns)

data_type_cat_col

print(f"Columns with categorical entries: {', '.join(data_type_cat_col)}.")

#loop through data_type_cat_col to check for duplicates in each column

for col in data_type_cat_col:
    if df[col].duplicated().any() == True:
        print(f'{col} column contains duplicates.')
    else:
        print(f'{col} column has no duplicates.')

# categorical columns have duplicates, check for words in different cases or extra spaces 

#loop through each categorical column for typos

for col in data_type_cat_col:
    print(f'Current count of {col} entries: {len(set(df[col]))} <-----> Crosschecking {col} entries: {len(set(df[col].str.title().str.strip()))}')

# There are entries with typos in the name column

#correct the typing errors in the Name column

df.name = df.name.str.title().str.strip()

#Cross check if correction has been applied to the name column using previous code

for col in data_type_cat_col:
    print(f'Current count of {col} entries: {len(set(df[col]))} <-----> Crosschecking {col} entries: {len(set(df[col].str.title().str.strip()))}')

# All entries have been corrected

#Check authors for spelling mistakes and repetitions

#Install fuzzywuzzy library to string match names of repeated authors

import fuzzywuzzy

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

process.extract('George R.R. Martin', df.author, limit=5)

process.extract('J.K. Rowling', df.author, limit=10)

# Author names R.R. Martin and J.K. Rowling are repeated in the column author
#Replace similar names with the correct name

df = df.replace('George R. R. Martin', 'George R.R. Martin')
df = df.replace('J. K. Rowling', 'J.K. Rowling')

#Cross check entries for any repetitions

for col in data_type_cat_col:
    print(f'Current count of {col} entries: {len(set(df[col]))} <-----> Crosschecking {col} entries: {len(set(df[col].str.title().str.strip()))}')

df.author.sort_values().unique()

#check the genre

df.genre.unique()

#check the years column

df.year.sort_values().unique()

#finally check entire dataframe

# Some rows are repeated

#delete year column 

del df['year']

df = df.drop_duplicates(keep = 'first')

process.extract('Wonder', df.name, limit=10)

process.extract('The Girl On The', df.name, limit=10)

# Some of the books are repeated, this could be due to the same book being sold at at different price

#Count repeated books in our data

from collections import Counter

repeat_books = Counter(df.name.tolist())

repeat_books.most_common(10)

#Remove duplicates, keep last entry of duplicates

df.drop_duplicates(subset='name', keep='last')

df=df.drop_duplicates(subset='name', keep='last')

#Top 10 authors

top_10_authors = df.groupby('author')[['user_rating']].mean().sort_values('user_rating', ascending = False)

top_10_authors=top_10_authors.head(10).reset_index()

#Number of books written by authors

number_of_books_written = df.groupby('author')[['name']].count().sort_values('name', ascending=False).head(10).reset_index()

number_of_books_written

#Number of books in each genre


number_of_books_by_genre = df.groupby('genre')[['name']].count().sort_values('name', ascending=False).head(10).reset_index()


# Create Models to Predict User Ratings: These models will give us insights into user rating behaviour

# #Lets predict the user ratings, therefore our target is the user_rating

#create dataframes for train, full_train, validating and finally testing our models


#import train_test_split 
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=28)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=28)

y_train = df_train.user_rating
y_test = df_test.user_rating
y_val = df_val.user_rating
y_full_train = df_full_train.user_rating

#delete target values from our dataframes using in prediction & final test
del df_train['user_rating']
del df_test['user_rating']
del df_val['user_rating']
del df_full_train['user_rating']

#reset indexes of our dataframes

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

#import DictVectorizer, mean_squared_error, roc_auc_score

from sklearn.feature_extraction import DictVectorizer 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

# Therefore the best model is with gradient boost regressor.
# 
# Create a function to train the model and another one to made predictions using the model

from sklearn.ensemble import GradientBoostingRegressor

def trainModel(df_full_train, y_full_train):
    dicts=df_full_train.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(dicts)
    
    model = GradientBoostingRegressor(random_state=28, max_depth=2,n_estimators=700, learning_rate=0.1)
    model.fit(X_full_train, y_full_train)
    
    return dv, model


dv, model = trainModel(df_full_train, y_full_train)


def predict(df, dv, model):
    dicts= df.to_dict(orient = 'records')
           
    X = dv.transform(dicts)
    y_pred = model.predict(X)
            
    return y_pred

y_pred = (df_test, dv, model)


#Use pickle to save the file


import pickle


#Save model and Vectorizer to file model_file.bin


out_file = 'model_file.bin'


with open(out_file, 'wb') as f_model:
    pickle.dump((dv, model), f_model)