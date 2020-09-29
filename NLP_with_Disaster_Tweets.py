# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 12:32:16 2020

@author: User
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


dataset= pd.read_csv('train.csv')

dataset['keyword'].fillna('other', inplace=True)

datset1=dataset.iloc[:,[1,3,4]]

X=dataset.iloc[:,1:2]


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()



ignore_words = ['?', '!','#','.','...']




import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 7613):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    if review not in ignore_words:
        corpus.append(lemmatizer.lemmatize(review.lower()))
    
    
    
    
    

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 100000)
X2 = cv.fit_transform(corpus).toarray()



X=pd.DataFrame(X)
X2=pd.DataFrame(X2)
training_set=pd.concat([X, X2.reindex(X.index)], axis=1)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_set, y, test_size = 0.20, random_state = 0)

'''

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)

sc2 = MinMaxScaler(feature_range = (0, 1))
y_train = sc2.fit_transform(y_train)


X_train = np.reshape(X_train, (6090,1,2222))








from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout



regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 4, activation='tanh', input_shape = (None, 2222),return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy')



regressor.fit(X_train, y_train, epochs = 10,batch_size = 50)


X_test=np.reshape(X_test,(1523,2222))
X_test=sc.transform(X_test)
X_test=np.reshape(X_test,(1523,1,2222))
y_pred =  regressor.predict(X_test)


y_pred = sc2.inverse_transform(y_pred)
y_pred = (y_pred > 0.3)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)