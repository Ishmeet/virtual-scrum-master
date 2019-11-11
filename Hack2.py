# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:59:53 2019

@author: Ishmeet
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
import pickle
import scipy.sparse as sp

os.getcwd()

#Setting directory to a current working directory
os.chdir('D:\Hack\git\Hackathon')

dat1=pd.read_excel('ABC1234.xlsx',header=0, encoding="ISO-8859-1", error_bad_lines=False)

#Data Preprocessing   
dat1.head()
dat1.describe()
dat1.dtypes

dat1['Assignee']
dat1['Assignee'].isnull().sum()
len(dat1['Assignee']) - dat1['Assignee'].isnull().sum()
Z = dat1['Assignee'].dropna()
dataframeZ = dat1[pd.notnull(dat1['Assignee'])]

# Train test split
X = dataframeZ[['Summary','Description',
          'Custom field (APAR Abstract)', 
          'Custom field (APAR Solution)',
          'Custom field (APAR Symptom)',
          'Custom field (Root Cause Analysis)']]
y = dataframeZ['Assignee']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorization
Vector1 = CountVectorizer()
Vector2 = CountVectorizer()
Vector3 = CountVectorizer()
Vector4 = CountVectorizer()
Vector5 = CountVectorizer()
Vector6 = CountVectorizer()
X_train_1 = Vector1.fit_transform(X_train['Summary'].values.astype('U'))
X_train_2 = Vector2.fit_transform(X_train['Description'].values.astype('U'))
X_train_3 = Vector3.fit_transform(X_train['Custom field (APAR Abstract)'].values.astype('U'))
X_train_4 = Vector4.fit_transform(X_train['Custom field (APAR Solution)'].values.astype('U'))
X_train_5 = Vector5.fit_transform(X_train['Custom field (APAR Symptom)'].values.astype('U'))
X_train_6 = Vector6.fit_transform(X_train['Custom field (Root Cause Analysis)'].values.astype('U'))
from scipy.sparse import hstack
X_train_vector = hstack([X_train_1,X_train_2,X_train_3,X_train_4,X_train_5,X_train_6])
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_train = TfidfTransformer()
X_train_vector_tfidf = tfidf_train.fit_transform(X_train_vector)


X_test_1 = Vector1.transform(X_test['Summary'].values.astype('U'))
X_test_2 = Vector2.transform(X_test['Description'].values.astype('U'))
X_test_3 = Vector3.transform(X_test['Custom field (APAR Abstract)'].values.astype('U'))
X_test_4 = Vector4.transform(X_test['Custom field (APAR Solution)'].values.astype('U'))
X_test_5 = Vector5.transform(X_test['Custom field (APAR Symptom)'].values.astype('U'))
X_test_6 = Vector6.transform(X_test['Custom field (Root Cause Analysis)'].values.astype('U'))
X_test_vector = hstack([X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6])
X_test_vector_tfidf = tfidf_train.transform(X_test_vector)

#ML Algorithm
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_vector_tfidf, y_train)
predicted = clf.predict(X_test_vector_tfidf)
np.mean(predicted == y_test)*100

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None).fit(X_train_vector_tfidf, y_train)
predicted_sgd = sgd.predict(X_test_vector_tfidf)
np.mean(predicted_sgd == y_test)*100

from sklearn import tree
Dtreeclf = tree.DecisionTreeClassifier().fit(X_train_vector_tfidf, y_train)
predicted_dtree = Dtreeclf.predict(X_test_vector_tfidf)
np.mean(predicted_dtree == y_test)*100

from sklearn.ensemble import RandomForestClassifier
randomclf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(X_train_vector_tfidf, y_train)
predicted_randomf = randomclf.predict(X_test_vector_tfidf)
np.mean(predicted_randomf == y_test)*100

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3).fit(X_train_vector_tfidf, y_train)
predicted_neigh = neigh.predict(X_test_vector_tfidf)
np.mean(predicted_neigh == y_test)*100

#Metrics
from sklearn import metrics
metrics.classification_report(y_test, predicted_dtree)

#Test
aabb_test_1 = Vector1.transform(["UI refresh is not happening if RG is deleted from backend"])
aabb_test_2 = Vector2.transform(["If resource group is deleted from backend then UI automatic refresh is not happening. If user removes the resource group from backend the cluster tree is not updated and Resource group is still visible on GUI."])
#aabb_test_3 = Vector3.transform(["<p>APAR_ABSTRACT=RG wont disappear from GUI after deletion till user perform sync<br>SYMPTOM - Any Symptoms the customer experience<br>#-------------------------------------------------------------|<br>START_SYMPTOM<br>RG wont disappear from GUI after deletion till user perform sync<br>STOP_SYMPTOM<br>SOLUTION - Description of the fix to the problem<br>#-------------------------------------------------------------|<br>START_SOLUTION<br>provided handling for missing case<br>STOP_SOLUTION<br>WORKAROUND - Temporary way around the problem<br>#-------------------------------------------------------------|<br>START_WORKAROUND<br>NA<br>STOP_WORKAROUND<br></p>"])
aabb_test_3 = Vector3.transform([""])
#aabb_test_4 = Vector4.transform(["provided handling for missing case"])
aabb_test_4 = Vector4.transform([""]) 
#aabb_test_5 = Vector5.transform(["RG wont disappear from GUI after deletion till user perform sync"])
aabb_test_5 = Vector5.transform([""]) 
#aabb_test_6 = Vector6.transform(["specific event handling was missing"])
aabb_test_6 = Vector6.transform([""])
aabb_test_vector = hstack([aabb_test_1, aabb_test_2, aabb_test_3, aabb_test_4, aabb_test_5, aabb_test_6])
aabb_test_vector_tfidf = tfidf_train.transform(aabb_test_vector)
Dtreeclf.predict(aabb_test_vector_tfidf)

# Grid Search
from sklearn.model_selection import GridSearchCV
parameters={'min_samples_split' : range(10,50,20),'max_depth': range(1,20,2)}
clf_tree=tree.DecisionTreeClassifier()
clf=GridSearchCV(clf_tree,parameters)
abc = clf.fit(X_train_vector_tfidf, y_train)
predicted_gridsearch = abc.predict(X_test_vector_tfidf)
gridsearch_model = abc.best_estimator_
my_model = gridsearch_model.fit(X_train_vector_tfidf, y_train)
predicted_my_model = my_model.predict(X_test_vector_tfidf)
np.mean(predicted_my_model == y_test)*100

#Label Propagation
from sklearn.semi_supervised import LabelPropagation
label_prop_model = LabelPropagation()
dat1['Custom field (Root Cause)']
labels = np.copy(dat1['Custom field (Root Cause)'])
RC = label_prop_model.fit(X_train_vector_tfidf, labels)

HackPickle = 'HackPickle.pkl'
# Open the file to save as pkl file
HackPickle_model_pkl = open(HackPickle, 'wb')
pickle.dump(clf, HackPickle_model_pkl)

pickle.dump(Vector1, open("Vector1.pkl", "wb"))
pickle.dump(Vector2, open("Vector2.pkl", "wb"))
pickle.dump(Vector3, open("Vector3.pkl", "wb"))
pickle.dump(Vector4, open("Vector4.pkl", "wb"))
pickle.dump(Vector5, open("Vector5.pkl", "wb"))
pickle.dump(Vector6, open("Vector6.pkl", "wb"))
pickle.dump(tfidf_train, open("tfidf_train.pkl", "wb"))
pickle.dump(clf, open("MultinomialNB.pkl", "wb"))
pickle.dump(sgd, open("SGDClassifier.pkl", "wb"))
pickle.dump(Dtreeclf, open("DecisionTreeClassifier.pkl", "wb"))
pickle.dump(randomclf, open("RandomForestClassifier.pkl", "wb"))
pickle.dump(neigh, open("KNeighborsClassifier.pkl", "wb"))
pickle.dump(my_model, open("GridSearchCV.pkl", "wb"))
# Close the pickle instances
HackPickle_model_pkl.close()



#Root Cause
dat1['Custom field (Root Cause)']
dat1['Custom field (Root Cause)'].isnull().sum()
len(dat1['Custom field (Root Cause)']) - dat1['Custom field (Root Cause)'].isnull().sum()
R = dat1['Custom field (Root Cause)'].dropna()
dataframeR = dat1[pd.notnull(dat1['Custom field (Root Cause)'])]

# Train test split
X2 = dataframeR[['Summary','Description',
          'Custom field (APAR Abstract)', 
          'Custom field (APAR Solution)',
          'Custom field (APAR Symptom)',
          'Custom field (Root Cause Analysis)']]
y2 = dataframeR['Custom field (Root Cause)']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)

# Vectorization
Vector2_1 = CountVectorizer()
Vector2_2 = CountVectorizer()
Vector2_3 = CountVectorizer()
Vector2_4 = CountVectorizer()
Vector2_5 = CountVectorizer()
Vector2_6 = CountVectorizer()
X_train_2_1 = Vector2_1.fit_transform(X_train2['Summary'].values.astype('U'))
X_train_2_2 = Vector2_2.fit_transform(X_train2['Description'].values.astype('U'))
X_train_2_3 = Vector2_3.fit_transform(X_train2['Custom field (APAR Abstract)'].values.astype('U'))
X_train_2_4 = Vector2_4.fit_transform(X_train2['Custom field (APAR Solution)'].values.astype('U'))
X_train_2_5 = Vector2_5.fit_transform(X_train2['Custom field (APAR Symptom)'].values.astype('U'))
X_train_2_6 = Vector2_6.fit_transform(X_train2['Custom field (Root Cause Analysis)'].values.astype('U'))
from scipy.sparse import hstack
X_train2_vector = hstack([X_train_2_1,X_train_2_2,X_train_2_3,X_train_2_4,X_train_2_5,X_train_2_6])
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_train2 = TfidfTransformer()
X_train_vector_tfidf2 = tfidf_train2.fit_transform(X_train2_vector)


X_test_2_1 = Vector2_1.transform(X_test2['Summary'].values.astype('U'))
X_test_2_2 = Vector2_2.transform(X_test2['Description'].values.astype('U'))
X_test_2_3 = Vector2_3.transform(X_test2['Custom field (APAR Abstract)'].values.astype('U'))
X_test_2_4 = Vector2_4.transform(X_test2['Custom field (APAR Solution)'].values.astype('U'))
X_test_2_5 = Vector2_5.transform(X_test2['Custom field (APAR Symptom)'].values.astype('U'))
X_test_2_6 = Vector2_6.transform(X_test2['Custom field (Root Cause Analysis)'].values.astype('U'))
X_test2_vector = hstack([X_test_2_1, X_test_2_2, X_test_2_3, X_test_2_4, X_test_2_5, X_test_2_6])
X_test_vector_tfidf2 = tfidf_train2.transform(X_test2_vector)

#ML Algorithm
from sklearn.naive_bayes import MultinomialNB
clf2 = MultinomialNB().fit(X_train_vector_tfidf2, y_train2)
predicted2 = clf2.predict(X_test_vector_tfidf2)
np.mean(predicted2 == y_test2)*100

from sklearn.linear_model import SGDClassifier
sgd2 = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None).fit(X_train_vector_tfidf2, y_train2)
predicted_sgd2 = sgd2.predict(X_test_vector_tfidf2)
np.mean(predicted_sgd2 == y_test2)*100

from sklearn import tree
Dtreeclf2 = tree.DecisionTreeClassifier().fit(X_train_vector_tfidf2, y_train2)
predicted_dtree2 = Dtreeclf2.predict(X_test_vector_tfidf2)
np.mean(predicted_dtree2 == y_test2)*100

from sklearn.ensemble import RandomForestClassifier
randomclf2 = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(X_train_vector_tfidf2, y_train2)
predicted_randomf2 = randomclf2.predict(X_test_vector_tfidf2)
np.mean(predicted_randomf2 == y_test2)*100

from sklearn.neighbors import KNeighborsClassifier
neigh2 = KNeighborsClassifier(n_neighbors=3).fit(X_train_vector_tfidf2, y_train2)
predicted_neigh2 = neigh2.predict(X_test_vector_tfidf2)
np.mean(predicted_neigh2 == y_test2)*100

pickle.dump(Vector2_1, open("Vector1_2.pkl", "wb"))
pickle.dump(Vector2_2, open("Vector2_2.pkl", "wb"))
pickle.dump(Vector2_3, open("Vector3_2.pkl", "wb"))
pickle.dump(Vector2_4, open("Vector4_2.pkl", "wb"))
pickle.dump(Vector2_5, open("Vector5_2.pkl", "wb"))
pickle.dump(Vector2_6, open("Vector6_2.pkl", "wb"))
pickle.dump(tfidf_train2, open("tfidf_train2.pkl", "wb"))
pickle.dump(clf2, open("MultinomialNB2.pkl", "wb"))
pickle.dump(sgd2, open("SGDClassifier2.pkl", "wb"))
pickle.dump(Dtreeclf2, open("DecisionTreeClassifier2.pkl", "wb"))
pickle.dump(randomclf2, open("RandomForestClassifier2.pkl", "wb"))
pickle.dump(neigh2, open("KNeighborsClassifier2.pkl", "wb"))





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
