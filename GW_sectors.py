from __future__ import print_function
from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import os
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['GOTO_NUM_THREADS'] = '12'
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['openmp'] = 'True'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils

import numpy as np
import pandas as pd
from SampleFileTools1 import SampleFile

obj = SampleFile()
obj.read_hdf("default_constant_10.hdf")
df = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

# Extracting signals and time columns and storing them in a new dataframe

data = df[['h1_signal', 'l1_signal', 'v1_signal', 'event_time', 'injection_snr']].copy()


# Extracting the index of the maximum value of strain for each detector from every sample.

import numpy as np
h1 = data.iloc[:,0]
l1 = data.iloc[:,1]
v1 = data.iloc[:,2]
#snr = data.iloc[:,4]
#timestamp = data.iloc[:,3]

h1_index = []
l1_index = []
v1_index = []
for i in range(10000):
    h1_index.append(np.argmax(np.abs(h1[i])))
    l1_index.append(np.argmax(np.abs(l1[i])))
    v1_index.append(np.argmax(np.abs(v1[i])))
    
    
# Extracting the maximum strain value of each detector for every sample
h1_max = []
l1_max = []
v1_max = []
for i in range(10000):
    h1_max.append(np.abs(h1[i]).max())
    l1_max.append(np.abs(l1[i]).max())
    v1_max.append(np.abs(v1[i]).max())

# Creating ratio of amplitudes 'H1/L1', 'L1/V1', 'H1/V1'

h1_l1_amp = []
l1_v1_amp = []
h1_v1_amp = []

for j in range(10000):
    h1_l1_amp.append(h1_max[j]/l1_max[j])
    l1_v1_amp.append(l1_max[j]/v1_max[j])
    h1_v1_amp.append(h1_max[j]/v1_max[j]) 
    
    
    
# Creating the time array

timestamp = []
event_time = 1234567936
count = 0
for i in range(512):
#    grid.append(np.linspace(event_time - sbe[i], event_time + (2.0 - sbe[i]), int(2048 * 0.4)))
    grid = np.linspace(event_time - 0.20, event_time + 0.05, int(2048 * 0.25))
    
    
# Extracting the angles from the dataframe

angles = df[['ra', 'dec']].copy()
ra = angles.iloc[:,0].values
dec = angles.iloc[:,1].values


#Assigning the labels, based on ra and dec angles
label_1 = []
pi = 3.141593
for i in range(10000):

# First surface, First slice

    if(((ra[i] >= pi) and (ra[i] <= 6.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
                label_1.append('One')
    elif(((ra[i] >= 6.0*pi/5.0) and (ra[i] <= 7.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
                label_1.append('Two')
    elif(((ra[i] >= 7.0*pi/5.0) and (ra[i] <= 8.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
                label_1.append('Three')
    elif(((ra[i] >= 8.0*pi/5.0) and (ra[i] <= 9.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
                label_1.append('Four')
    elif(((ra[i] >= 9.0*pi/5.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
                label_1.append('Five')
    elif(((ra[i] >= 0.0) and (ra[i] <= pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
                label_1.append('Six')
    elif(((ra[i] >= pi/5.0) and (ra[i] <= 2.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
                label_1.append('Seven')
    elif(((ra[i] >= 2.0*pi/5.0) and (ra[i] <= 3.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
                label_1.append('Eight')
    elif(((ra[i] >= 3.0*pi/5.0) and (ra[i] <= 4.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
                label_1.append('Nine')
    elif(((ra[i] >= 4.0*pi/5.0) and (ra[i] <= pi)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
                label_1.append('Ten')
            

# Second surface

    elif(((ra[i] >= 2.0*pi/3.0) and (ra[i] <= pi)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
                label_1.append('Ten')
    elif(((ra[i] >= pi/3.0) and (ra[i] <= 2.0*pi/3.0)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
                label_1.append('Eleven')
    elif(((ra[i] >= 0.0) and (ra[i] <= pi/3.0)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
                label_1.append('Twelve')
    elif(((ra[i] >= 2.0*pi/3.0) and (ra[i] <= pi)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
                label_1.append('Thirteen')
    elif(((ra[i] >= pi/3.0) and (ra[i] <= 2.0*pi/3.0)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
                label_1.append('Fourteen')
    elif(((ra[i] >= 0.0) and (ra[i] <= pi/3.0)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
                label_1.append('Fifteen')
    elif(((ra[i] >= 2.0*pi/3.0) and (ra[i] <= pi)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
                label_1.append('Sixteen')
    elif(((ra[i] >= pi/3.0) and (ra[i] <= 2.0*pi/3.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
                label_1.append('Seventeen')
    elif(((ra[i] >= 0.0) and (ra[i] <= pi/3.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
                label_1.append('Eighteen')        
    
    

# Creating the arrays of phase lags 'H1-L1', 'L1-V1' and 'H1-V1'

h1_l1 = []
l1_v1 = []
h1_v1 = []

for i in range(10000):
    h1_l1.append(grid[h1_index[i]] - grid[[l1_index[i]]])
    l1_v1.append(grid[l1_index[i]] - grid[[v1_index[i]]])
    h1_v1.append(grid[h1_index[i]] - grid[[v1_index[i]]])

h1l1 = np.hstack(h1_l1)
l1v1 = np.hstack(l1_v1)
h1v1 = np.hstack(h1_v1)



# Creating the pandas dataframe with the input features and the labels for the neural network

df_1 = {"H1/L1": h1_l1_amp, "L1/V1": l1_v1_amp, "H1/V1": h1_v1_amp, "H1-L1 Phase lag": h1l1, "L1-V1 Phase lag": l1v1, "H1-V1 Phase lag": h1v1, "Sector": label_1}

dataframe = pd.DataFrame(df_1)



# Define column name of the label vector
LABEL = 'SectorEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
dataframe[LABEL] = le.fit_transform(dataframe['Sector'].values.ravel())

# Splitting into input features and labels

X = dataframe.iloc[:, 0:6].values
y = dataframe.iloc[:, 7].values



# Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Set input & output dimensions
#num_time_periods = X_train.shape[1]
num_classes = le.classes_.size
print(list(le.classes_))


# Before continuing, we need to convert all feature data (x_train) and label data (y_train) into a datatype accepted by Keras.
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#One last step we need to do is to conduct one-hot-encoding of our labels. Please only execute this line once!
y_train_hot = np_utils.to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train_hot.shape)


# Making the ANN

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))

# Adding the second hidden layer
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = num_classes, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set

classifier.fit(X_train, y_train_hot, batch_size = 10, epochs = 100)


y_test = np_utils.to_categorical(y_test, num_classes)

score = classifier.evaluate(X_test, y_test, verbose=1)


print('\nAccuracy on test data: %0.2f' % score[1])
print('\nLoss on test data: %0.2f' % score[0])



























