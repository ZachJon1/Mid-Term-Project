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

_______Libraries__________

pandas
numpy 

Plots:

seaborn
from matplotlib import pyplot as plt

Cleaning data:

import missingno
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from collections import Counter

Splitting Data, Create Vectorizers :

from sklearn.model_selection import train_test_split
