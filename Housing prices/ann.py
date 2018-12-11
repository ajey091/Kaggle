# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense


# Importing the dataset
# dataset = pd.read_csv('/Users/b119user/Google Drive/Kaggle/Housing prices/all/train.csv')
dataset = pd.read_csv("all/train.csv")
dataset.iloc[:,3] = dataset.iloc[:,3].fillna(0)
X = dataset.iloc[:, [1,2,3,4,12,17,19,46,27,53,54,56,79,43,44,62]].values
y = dataset.iloc[:, 80].values
y=np.reshape(y, (-1,1))

dataset.iloc[:,1] = dataset.iloc[:,1].fillna(0)

# Encoding categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8])
labelencoder_X_9 = LabelEncoder()
X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])
labelencoder_X_12 = LabelEncoder()
X[:, 12] = labelencoder_X_12.fit_transform(X[:, 12])

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
print(sc.fit(X))
X_train = sc.transform(X)

# creating ann

model = Sequential()
model.add(Dense(12, input_dim=16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(6, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae','accuracy'])

history = model.fit(X_train, y, epochs=500, batch_size=20,  verbose=2,validation_split=0.2)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Applying model to test data

dataset_test = pd.read_csv("all/test.csv")
dataset_test.iloc[:,53] = dataset_test.iloc[:,53].fillna("UNKNOWN")
dataset_test.iloc[:,62] = dataset_test.iloc[:,62].fillna(0)
dataset_test.iloc[:,3] = dataset_test.iloc[:,3].fillna(0)
X_test = dataset_test.iloc[:, [1,2,3,4,12,17,19,46,27,53,54,56,79,43,44,62]].values

# Encoding categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_test_1 = LabelEncoder()
X_test[:, 1] = labelencoder_X_test_1.fit_transform(X_test[:, 1].astype(str))
labelencoder_X_test_4 = LabelEncoder()
X_test[:, 4] = labelencoder_X_test_4.fit_transform(X_test[:, 4])
labelencoder_X_test_8 = LabelEncoder()
X_test[:, 8] = labelencoder_X_test_8.fit_transform(X_test[:, 8])
labelencoder_X_test_9 = LabelEncoder()
X_test[:, 9] = labelencoder_X_test_9.fit_transform(X_test[:, 9])
labelencoder_X_test_12 = LabelEncoder()
X_test[:, 12] = labelencoder_X_test_12.fit_transform(X_test[:, 12])

X_test = X_test.astype(np.float)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc2 = MinMaxScaler()
print(sc2.fit(X_test))
X_testscaled = sc2.transform(X_test)

y_pred=model.predict(X_testscaled)
testdataid = dataset_test.iloc[:,0]
myData = [testdataid,y_pred]




f = open('all/output.csv','w')
f.write("Id,SalePrice\n")
for ii  in range(0,len(y_pred)):

    f.write('%s,%s\n' %(str(myData[0][ii]), str(float(y_pred[ii]))))
f.close()

# y_test = sc.inverse_transform(y_test)
# plt.plot(y_test,y_pred,'.')
# plt.ylabel('Predicted')
# plt.xlabel('Actual')
# plt.show()

