# Mid-Term-Project
--------------------------------------------Problem Statement---------------------------------------------------------

Amazon has data on the 50 best sellers from 2009 to 2019. The company would like to have more insights about their customers' behavior through data visualizations and a model to predict user rating of a book. Amazon hopes to build a simple model that will help inform about customer behaviour through the rating of these best sellers.

Data: The data for this project was obtained from kaggle 

Link to the data: https://www.kaggle.com/sootersaalu/amazon-top-50-bestselling-books-2009-2019

Before working on this data I looked at the work of others who used this data some of work that may have affected my analysis are the following 

__________Links_______________

https://www.kaggle.com/aryan27/amazon-top-50-bestselling-books-2009-2019

https://www.kaggle.com/raj5kumar5/amazon-book-rating-prediction

https://www.kaggle.com/ivannatarov/amazon-s-books-eda-plotly-hypothesis-test

https://www.kaggle.com/shreyasajal/amazon-bestselling-books-plotly-visualizations


The libraries that were used in the reading, cleaning, visualization, modeling and deploymet of the model 

___________________________________________________Libraries____________________________________________

pandas
numpy 
gunicorn
Visualization:

seaborn
from matplotlib import pyplot as plt

Cleaning data:

import missingno
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from collections import Counter

Splitting Data, Create Vectorizers :

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer 

Metrics:

from sklearn.metrics import mean_squared_error
*from sklearn.metrics import roc_auc_score  ---------> *Left out of the final set of metrics used in evaluation of different models


The models that were used in training the data sets

_____________________________Models________

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

To tune our parameters, I used GridSearchCV, Link: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

Fine tuning with GridSearchCV was comparatively easy than the initial fine tuning with manual entries for each parameter.

GridSearchCV allows for a mulittasked search for the best parameters for each model. Hence different combinations of the Hyperparameters of the models are used till the best results are obtained.

Also it cross validates the model by using different sets of the training data in searching for the optimum parameters.

The best model that was chosen was based on the mean_squared_error which is a measure of the residual between the model predicted value and the actual data points.

In order to export code pickle was used to store the model

import pickle

from flask import Flask
from flask import request
from flask import jsonify
import requests

The final model model was deployed to the local host with flask and gunicorn 

A docker file & docker image were also created and used to create containers of the model.

The model was also deployed to Heroku.

Key Files 

Project_predict.py

Project_train.py

Project_test.py

requirements.txt

wsgi.py

Procfile

Dockerfile

Mid-Term Project III - Coding mostly done in Jupyter notebook before exporting as script. It contains EDA & training the models as well.

pipfile.lock

pipfile

model_file.bin




