# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
# from sklearn import linear_model
from sklearn.model_selection import train_test_split


# Importing the dataset
# dataset = pd.read_csv('/Users/b119user/Google Drive/Kaggle/Housing prices/all/train.csv')
dataset = pd.read_csv('all/train.csv')

X = dataset.iloc[:, [1,4,12,17,18,19]].values
y = dataset.iloc[:, 80].values
y=np.reshape(y, (-1,1))

# Encoding categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# # Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
print(sc.fit(X))

X_scale = sc.transform(X)
y_scale = y

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_scale, y_scale, test_size = 0.2, random_state = 0)

clf = SVR(kernel='rbf', C=1e2, gamma=0.5)
# clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

plt.plot(y_test,y_pred,'.')
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.show()
mse = np.mean((y_pred - y_test)**2)
print(mse)