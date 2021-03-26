#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate, KFold

import sys

from large_feature_space_kernels.mismatchingTrie import MismatchingTrie 
from estimators.svm_estimator import SVM
from estimators.ridge_regression_estimator import RidgeRegression
from estimators.utils import ensemble_predictions

import cvxopt
cvxopt.solvers.options['show_progress'] = False


# ### Dataset 1

# In[2]:


print("Reading first dataset")

k = 0
list_of_predictions = []
folder = "Data/"
Xtr = pd.read_csv(folder+"Xtr"+str(k)+".csv", index_col=0)
Xte = pd.read_csv(folder+"Xte"+str(k)+".csv", index_col=0)
Ytr = pd.read_csv(folder+"Ytr"+str(k)+".csv", index_col=0)
Ytr["Bound"] = Ytr["Bound"].apply(lambda x: 1 if x == 1 else -1)


# In[3]:


X = Xtr.values[:,0]
y = Ytr.values[:,0]
xtest = Xte.values[:,0]

models = [SVM(k=7, m=0, alpha=0.1),SVM(k=9, m=1, alpha=1),RidgeRegression(k=10,m=1,alpha=10)]

for model in models:
    print("Fitting a new model for dataset 1")
    model.fit(X, y)
    Yte = model.predict(xtest)
    ytest = pd.DataFrame(Yte,index=Xte.index,columns=["Bound"])
    ytest = ytest.replace(-1,0)
    list_of_predictions.append(ytest.iloc[:,0].values)


# In[ ]:


predictions_set1 = ensemble_predictions(list_of_predictions)
predictions_set1 = pd.DataFrame(predictions_set1,index=Xte.index,columns=["Bound"])
predictions_set1 = predictions_set1.astype(int)


# ### Dataset 2

# In[ ]:


print("Reading second dataset")

k = 1
list_of_predictions = []
folder = "Data/"
Xtr = pd.read_csv(folder+"Xtr"+str(k)+".csv", index_col=0)
Xte = pd.read_csv(folder+"Xte"+str(k)+".csv", index_col=0)
Ytr = pd.read_csv(folder+"Ytr"+str(k)+".csv", index_col=0)
Ytr["Bound"] = Ytr["Bound"].apply(lambda x: 1 if x == 1 else -1)


# In[ ]:


X = Xtr.values[:,0]
y = Ytr.values[:,0]
xtest = Xte.values[:,0]

models = [RidgeRegression(k=6, m=0, alpha=0.1),SVM(k=6, m=0, alpha=0.1),RidgeRegression(k=10,m=1,alpha=10)]

for model in models:
    print("Fitting a new model for dataset 2")
    model.fit(X, y)
    Yte = model.predict(xtest)
    ytest = pd.DataFrame(Yte,index=Xte.index,columns=["Bound"])
    ytest = ytest.replace(-1,0)
    list_of_predictions.append(ytest.iloc[:,0].values)


# In[ ]:


predictions_set2 = ensemble_predictions(list_of_predictions)
predictions_set2 = pd.DataFrame(predictions_set2,index=Xte.index,columns=["Bound"])
predictions_set2 = predictions_set2.astype(int)


# ### Dataset 3

# In[ ]:


print("Reading third dataset")

k = 2
list_of_predictions = []
folder = "Data/"
Xtr = pd.read_csv(folder+"Xtr"+str(k)+".csv", index_col=0)
Xte = pd.read_csv(folder+"Xte"+str(k)+".csv", index_col=0)
Ytr = pd.read_csv(folder+"Ytr"+str(k)+".csv", index_col=0)
Ytr["Bound"] = Ytr["Bound"].apply(lambda x: 1 if x == 1 else -1)


# In[ ]:


X = Xtr.values[:,0]
y = Ytr.values[:,0]
xtest = Xte.values[:,0]

models = [RidgeRegression(k=6, m=0, alpha=0.1),RidgeRegression(k=7, m=0, alpha=0.1),RidgeRegression(k=10,m=1,alpha=10)]

for model in models:
    print("Fitting a new model for dataset 3")
    model.fit(X, y)
    Yte = model.predict(xtest)
    ytest = pd.DataFrame(Yte,index=Xte.index,columns=["Bound"])
    ytest = ytest.replace(-1,0)
    list_of_predictions.append(ytest.iloc[:,0].values)


# In[ ]:


predictions_set3 = ensemble_predictions(list_of_predictions)
predictions_set3 = pd.DataFrame(predictions_set3,index=Xte.index,columns=["Bound"])
predictions_set3 = predictions_set3.astype(int)


# Concatenation

# In[ ]:


final_pred = pd.concat([predictions_set1,predictions_set2,predictions_set3],axis=0)
print(final_pred)
final_pred.to_csv("Yte.csv")

