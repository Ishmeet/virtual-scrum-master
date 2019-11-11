# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:59:53 2019

@author: Ishmeet
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import pickle

os.getcwd()

#Setting directory to a current working directory
os.chdir('D:\Hack')

dat1=pd.read_excel('9C3CFD60.xls',header=0, encoding="ISO-8859-1", error_bad_lines=False)
   
dat1.head()
dat1.describe()
dat1.dtypes

dat1['Assignee']

# Train test split
X = dat1[['Summary','Description']]
y = dat1['Assignee']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train['Description'])
X_train_counts.shape
count_vect.vocabulary_.get(u'configuration')
count_vect.transform(["configuration"]).toarray()
X_train_counts.toarray()
X_train_counts[1].toarray()

# Multinomial NaiveBayes
clf = MultinomialNB().fit(X_train_counts, y_train)

#Predict
X_new_counts = count_vect.transform(X_test['Description'])
predicted = clf.predict(X_new_counts)
len(y_test)
len(predicted)

# Accuracy
print( "Nave Bayes Accuracy", metrics.accuracy_score(predicted, y_test)*100)
np.mean(predicted == y_test)

# Dump the trained decision tree classifier with Pickle
HackPickle = 'HackPickle.pkl'
# Open the file to save as pkl file
HackPickle_model_pkl = open(HackPickle, 'wb')
pickle.dump(clf, HackPickle_model_pkl)
pickle.dump(count_vect, open("vector.pickel", "wb"))
# Close the pickle instances
HackPickle_model_pkl.close()




# Loading the saved decision tree model pickle
Hack_Pickle_model_pkl = open(HackPickle, 'rb')
Hack_Pickle_model = pickle.load(Hack_Pickle_model_pkl)
print ("Loaded Logistic_Regression model :: ", Hack_Pickle_model)
Hack_Pickle_model.predict(X_new_counts)
