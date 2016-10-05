# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:19:03 2016

@author: CHY
"""

## Survival Analysis using Keras
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2, activity_l2
import theano.tensor as T

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter

from lifelines.datasets import load_rossi
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

rossi_dataset = load_rossi()
E=np.array(rossi_dataset["arrest"])
Y=np.array(rossi_dataset["week"])
X=np.array(rossi_dataset)
X=X.astype('float64')
X=X[:,2:]

X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.25, random_state=0)
X_train,X_val,E_train,E_val=train_test_split(X,E,test_size=0.25, random_state=0)

#Standardize
scaler=preprocessing.StandardScaler().fit(X_train[:,[1,6]])
X_train[:,[1,6]]=scaler.transform(X_train[:,[1,6]])
X_val[:,[1,6]]=scaler.transform(X_val[:,[1,6]])

#Sorting for NNL!
sort_idx = np.argsort(Y_train)[::-1]
X_train=X_train[sort_idx]
Y_train=Y_train[sort_idx]
E_train=E_train[sort_idx]


#Loss Function
def negative_log_likelihood(E):
	def loss(y_true,y_pred):
		hazard_ratio = T.exp(y_pred)
		log_risk = T.log(T.extra_ops.cumsum(hazard_ratio))
		uncensored_likelihood = y_pred.T - log_risk
		censored_likelihood = uncensored_likelihood * E
		neg_likelihood = -T.sum(censored_likelihood)
		return neg_likelihood
	return loss
	

#Keras model
model = Sequential()
model.add(Dense(32, input_shape=(7,), init='glorot_uniform')) # shape= length, dimension
model.add(Activation('relu'))
model.add(Dense(32, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation="linear", init='glorot_uniform', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
#

sgd = SGD(lr=1e-5, decay=0.01, momentum=0.9, nesterov=True)
rmsprop=RMSprop(lr=1e-5, rho=0.9, epsilon=1e-8)
model.compile(loss=negative_log_likelihood(E_train), optimizer=sgd)

print('Training...')
model.fit(X_train, Y_train, batch_size=324, nb_epoch=1000, shuffle=False)  # Shuffle False --> Important!!

hr_pred=model.predict(X_train)
hr_pred=np.exp(hr_pred)
ci=concordance_index(Y_train,-hr_pred,E_train)

hr_pred2=model.predict(X_val)
hr_pred2=np.exp(hr_pred2)
ci2=concordance_index(Y_val,-hr_pred2,E_val)
print 'Concordance Index for training dataset:', ci
print 'Concordance Index for test dataset:', ci2

#Cox Fitting
cf = CoxPHFitter()
cf.fit(rossi_dataset, 'week', event_col='arrest')

cf.print_summary()  # access the results using cf.summary