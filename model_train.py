import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('D:\ML PROJECT\heart_disease_prediction\Dataset--Heart-Disease-Prediction-using-ANN.csv')

print(data.head())

print(data.describe().T)

print(data.isnull().any())


X = data.iloc[:,:13].values
y = data["target"].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
import joblib

# Save the fitted StandardScaler to a file
joblib.dump(sc, "scaler.pkl")
X_test = sc.transform(X_test)

classifier = Sequential()

classifier.add(Dense(activation='relu',input_dim=13,units=32,kernel_initializer="uniform"))

classifier.add(Dense(activation='relu',units=32,kernel_initializer="uniform"))

classifier.add(Dense(activation="sigmoid",units=1,kernel_initializer="uniform"))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=8,epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test,y_pred)
print(cm)

accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1])
print(accuracy*100)

classifier.save("heart_ann_model.h5")
