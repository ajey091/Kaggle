import pandas as pd
# import statsmodels.api as sm
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split


data_train = pd.read_csv("C:\Users\Ajey\Google Drive\Kaggle\train.csv",na_filter=0)
data_train
data_test = pd.read_csv('C:\Users\Ajey\Google Drive\Kaggle\test.csv',na_filter=0)
data_test

data_test["Fare"] = pd.to_numeric(data_test["Fare"], errors='coerce')
data_train["Age"] = pd.to_numeric(data_train["Age"], errors='coerce')
data_test["Age"] = pd.to_numeric(data_test["Age"], errors='coerce')

ktrain = np.zeros(len(data_train["Sex"]))
Age2_train = data_train["Age"]
Age2_train[data_train["Age"].isnull()] = 30
ktrain[data_train['Sex']=='female'] = 2
ktrain[ktrain==0] = 1
data_train["Sexint"] = ktrain
data_train["Age2"]  = Age2_train


ktest = np.zeros(len(data_test["Sex"]))
ktest[data_test['Sex']=='female'] = 2
ktest[ktest==0] = 1
Age2_test = data_test["Age"]
Age2_test[data_test["Age"].isnull()] = 30
data_test["Sexint"] = ktest
data_test["Age2"]  = Age2_test

cols = ["Pclass","Sexint","SibSp","Age2"]
X_train = data_train[cols]
X_train.dropna()
y_train = data_train["Survived"]
X_test = data_test[cols]
X_test.dropna()

clf = SVC()
clf.fit(X_train,y_train)
print(clf)


logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
file1=open('PredictedSurvival.txt','w')
np.savetxt('PredictedSurvival.txt',y_pred,fmt='%d')
