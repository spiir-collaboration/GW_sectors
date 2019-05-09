from __future__ import print_function
from matplotlib import pyplot as plt
#%matplotlib inline
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
obj.read_hdf("default_noise.hdf")
df = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

# Extracting signals and time columns and storing them in a new dataframe

data = df[['h1_strain', 'l1_strain', 'v1_strain', 'event_time', 'injection_snr']].copy()

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
for i in range(30000):
    h1_index.append(np.argmax(np.abs(h1[i])))
    l1_index.append(np.argmax(np.abs(l1[i])))
    v1_index.append(np.argmax(np.abs(v1[i])))

# Extracting the maximum strain value of each detector for every sample
#h1_max = []
#l1_max = []
#v1_max = []
#for i in range(12000):
#    h1_max.append((np.abs(h1[i])).max())
#    l1_max.append((np.abs(l1[i])).max())
#    v1_max.append((np.abs(v1[i])).max())

# Creating ratio of amplitudes 'H1/L1', 'L1/V1', 'H1/V1'

#h1_l1_amp = []
#l1_v1_amp = []
#h1_v1_amp = []

#for j in range(12000):
#    h1_l1_amp.append(h1_max[j]/l1_max[j])
#    l1_v1_amp.append(l1_max[j]/v1_max[j])
#    h1_v1_amp.append(h1_max[j]/v1_max[j]) 

# Creating array of ratios of second largest amplitudes for each detector per sample

#h1_max2 = []
#l1_max2 = []
#v1_max2 = []

#for i in range(12000):
#    h1_des = np.abs(h1[i])
#    l1_des = np.abs(l1[i])
#    v1_des = np.abs(v1[i])
#    h1_des.sort()
#    l1_des.sort()
#    v1_des.sort()
#    h1_max2.append(h1_des[510])
#    l1_max2.append(l1_des[510])
#    v1_max2.append(v1_des[510])
    
#h1_l1_amp2 = []
#l1_v1_amp2 = []
#h1_v1_amp2 = []

#for j in range(12000):
#    h1_l1_amp2.append(h1_max2[j]/l1_max2[j])
#    l1_v1_amp2.append(l1_max2[j]/v1_max2[j])
#    h1_v1_amp2.append(h1_max2[j]/v1_max2[j])

# Creating array of ratios of sum of ten highest ampltudes for each sample

h1_sum = 0.0
l1_sum = 0.0
v1_sum = 0.0
h1_l1_sum = []
l1_v1_sum = []
h1_v1_sum = []

for i in range(30000):
    h1_sum = 0.0
    l1_sum = 0.0
    v1_sum = 0.0
    h1_des = np.abs(h1[i])
    l1_des = np.abs(l1[i])
    v1_des = np.abs(v1[i])
    h1_des.sort()
    l1_des.sort()
    v1_des.sort()
    for j in range(10):
        h1_sum = h1_sum + h1_des[511 - j]
        l1_sum = l1_sum + l1_des[511 - j]
        v1_sum = v1_sum + v1_des[511 - j]
        
    h1_l1_sum.append(h1_sum/l1_sum)
    l1_v1_sum.append(l1_sum/v1_sum)
    h1_v1_sum.append(h1_sum/v1_sum)
    
        
        

    
    
    
#----------------------------------------- FFT----------------------------------------------------------------------------

# Extracting the maximum amplitude from each sample after taking the FFTs
    
#h1_amp = []
#l1_amp = []
#v1_amp = []

#for i in range(12000):
#    h1_amp.append(((np.abs(np.fft.fft(h1[i])))**2.0).max())
#    l1_amp.append(((np.abs(np.fft.fft(l1[i])))**2.0).max())
#    v1_amp.append(((np.abs(np.fft.fft(v1[i])))**2.0).max())

# Creating ratios of maximum FFT amplitudes for each sample

#h1_l1_amp = []
#l1_v1_amp = []
#h1_v1_amp = []

#for i in range(12000):
#    h1_l1_amp.append(h1_amp[i]/l1_amp[i])
#    l1_v1_amp.append(l1_amp[i]/v1_amp[i])
#    h1_v1_amp.append(h1_amp[i]/v1_amp[i])

# Extracting the index of the maximum amplitude for each sample after taking FFTs    
    
h1_index_f = []
l1_index_f = []
v1_index_f = []

for i in range(30000):
    h1_index_f.append(np.argmax(np.abs(np.fft.fft(h1[i]))))
    l1_index_f.append(np.argmax(np.abs(np.fft.fft(l1[i]))))
    v1_index_f.append(np.argmax(np.abs(np.fft.fft(v1[i]))))
    
# Extracting the phases corresponding to the maximum amplitudes for each sample after taking the FFTs

h1_phase = []
l1_phase = []
v1_phase = []
for i in range(30000):
    a = np.angle(np.fft.fft(h1[i]))
    b = np.angle(np.fft.fft(l1[i]))
    c = np.angle(np.fft.fft(v1[i]))
    h1_phase.append(a[h1_index_f[i]])
    l1_phase.append(b[l1_index_f[i]])
    v1_phase.append(c[v1_index_f[i]])
    
# Extracting the frequencies corresponding to the maximum FFT amplitudes for each sample

h1l1_freq = []
l1v1_freq = []
h1v1_freq = []
time_step = 1/2048
for i in range(30000):
    d = np.fft.fftfreq(h1[i].size, time_step)
    e = np.fft.fftfreq(l1[i].size, time_step)
    f = np.fft.fftfreq(v1[i].size, time_step)
    h1l1_freq.append(np.abs(d[h1_index_f[i]] - e[l1_index_f[i]]))
    l1v1_freq.append(np.abs(e[l1_index_f[i]] - f[v1_index_f[i]]))
    h1v1_freq.append(np.abs(d[h1_index_f[i]] - f[v1_index_f[i]]))



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
for i in range(30000):

# First surface

    if(((ra[i] >= pi) and (ra[i] <= 4.0*pi/3.0)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
                label_1.append('One')
    elif(((ra[i] >= 4.0*pi/3.0) and (ra[i] <= 5.0*pi/3.0)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
                label_1.append('Two')
    elif(((ra[i] >= 5.0*pi/3.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
                label_1.append('Three')
    elif(((ra[i] >= pi) and (ra[i] <= 4.0*pi/3.0)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
                label_1.append('Four')
    elif(((ra[i] >= 4.0*pi/3.0) and (ra[i] <= 5.0*pi/3.0)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
                label_1.append('Five')
    elif(((ra[i] >= 5.0*pi/3.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
                label_1.append('Six')
    elif(((ra[i] >= pi) and (ra[i] <= 4.0*pi/3.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
                label_1.append('Seven')
    elif(((ra[i] >= 4.0*pi/3.0) and (ra[i] <= 5.0*pi/3.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
                label_1.append('Eight')
    elif(((ra[i] >= 5.0*pi/3.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
                label_1.append('Nine')
            

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
    
#label_1 = []
#pi = 3.141593
#for i in range(12000):

# First slice

#    if(((ra[i] >= pi) and (ra[i] <= 6.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
#                label_1.append('One')
#    elif(((ra[i] >= 6.0*pi/5.0) and (ra[i] <= 7.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Two')
#    elif(((ra[i] >= 7.0*pi/5.0) and (ra[i] <= 8.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Three')
#    elif(((ra[i] >= 8.0*pi/5.0) and (ra[i] <= 9.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Four')
#    elif(((ra[i] >= 9.0*pi/5.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Five')
#    elif(((ra[i] >= 0.0) and (ra[i] <= pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Six')
#    elif(((ra[i] >= pi/5.0) and (ra[i] <= 2.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Seven')
#    elif(((ra[i] >= 2.0*pi/5.0) and (ra[i] <= 3.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Eight')
#    elif(((ra[i] >= 3.0*pi/5.0) and (ra[i] <= 4.0*pi/5.0)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Nine')
#    elif(((ra[i] >= 4.0*pi/5.0) and (ra[i] <= pi)) and ((dec[i] >= 3.0*pi/10.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Ten')
                
# Second slice

#    elif(((ra[i] >= pi) and (ra[i] <= 6.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
#                label_1.append('Eleven')
#    elif(((ra[i] >= 6.0*pi/5.0) and (ra[i] <= 7.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
#                label_1.append('Twelve')
#    elif(((ra[i] >= 7.0*pi/5.0) and (ra[i] <= 8.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
#                label_1.append('Thirteen')
#    elif(((ra[i] >= 8.0*pi/5.0) and (ra[i] <= 9.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
#                label_1.append('Fourteen')
#    elif(((ra[i] >= 9.0*pi/5.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
#                label_1.append('Fifteen')
#    elif(((ra[i] >= 0.0) and (ra[i] <= pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
#                label_1.append('Sixteen')
#    elif(((ra[i] >= pi/5.0) and (ra[i] <= 2.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
#               label_1.append('Seventeen')
#    elif(((ra[i] >= 2.0*pi/5.0) and (ra[i] <= 3.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
#                label_1.append('Eighteen')
#    elif(((ra[i] >= 3.0*pi/5.0) and (ra[i] <= 4.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
#                label_1.append('Nineteen')
#    elif(((ra[i] >= 4.0*pi/5.0) and (ra[i] <= pi)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
#                label_1.append('Twenty')
                
# Third slice

#    elif(((ra[i] >= pi) and (ra[i] <= 6.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
#                label_1.append('Twenty-one')
#    elif(((ra[i] >= 6.0*pi/5.0) and (ra[i] <= 7.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
#                label_1.append('Twenty-two')
#    elif(((ra[i] >= 7.0*pi/5.0) and (ra[i] <= 8.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
#                label_1.append('Twenty-three')
#    elif(((ra[i] >= 8.0*pi/5.0) and (ra[i] <= 9.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
#                label_1.append('Twenty-four')
#    elif(((ra[i] >= 9.0*pi/5.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
#                label_1.append('Twenty-five')
#    elif(((ra[i] >= 0.0) and (ra[i] <= pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
#                label_1.append('Twenty-six')
#    elif(((ra[i] >= pi/5.0) and (ra[i] <= 2.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
#                label_1.append('Twenty-seven')
#    elif(((ra[i] >= 2.0*pi/5.0) and (ra[i] <= 3.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
#                label_1.append('Twenty-eight')
#    elif(((ra[i] >= 3.0*pi/5.0) and (ra[i] <= 4.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
#                label_1.append('Twenty-nine')
#    elif(((ra[i] >= 4.0*pi/5.0) and (ra[i] <= pi)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
#                label_1.append('Thirty')
                
# Fourth slice

#    elif(((ra[i] >= pi) and (ra[i] <= 6.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
#                label_1.append('Thirty-one')
#    elif(((ra[i] >= 6.0*pi/5.0) and (ra[i] <= 7.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
#                label_1.append('Thirty-two')
#    elif(((ra[i] >= 7.0*pi/5.0) and (ra[i] <= 8.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
#                label_1.append('Thirty-three')
#    elif(((ra[i] >= 8.0*pi/5.0) and (ra[i] <= 9.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
#                label_1.append('Thirty-four')
#    elif(((ra[i] >= 9.0*pi/5.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
#                label_1.append('Thirty-five')
#    elif(((ra[i] >= 0.0) and (ra[i] <= pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
#                label_1.append('Thirty-six')
#    elif(((ra[i] >= pi/5.0) and (ra[i] <= 2.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
#                label_1.append('Thirty-seven')
#    elif(((ra[i] >= 2.0*pi/5.0) and (ra[i] <= 3.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
#                label_1.append('Thirty-eight')
#    elif(((ra[i] >= 3.0*pi/5.0) and (ra[i] <= 4.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
#                label_1.append('Thirty-nine')
#    elif(((ra[i] >= 4.0*pi/5.0) and (ra[i] <= pi)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
#                label_1.append('Forty')
                
# Fifth slice

#    elif(((ra[i] >= pi) and (ra[i] <= 6.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
#                label_1.append('Forty-one')
#    elif(((ra[i] >= 6.0*pi/5.0) and (ra[i] <= 7.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
#                label_1.append('Forty-two')
#    elif(((ra[i] >= 7.0*pi/5.0) and (ra[i] <= 8.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
#                label_1.append('Forty-three')
#    elif(((ra[i] >= 8.0*pi/5.0) and (ra[i] <= 9.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
#                label_1.append('Forty-four')
#    elif(((ra[i] >= 9.0*pi/5.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
#                label_1.append('Forty-five')
#    elif(((ra[i] >= 0.0) and (ra[i] <= pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
#                label_1.append('Forty-six')
#    elif(((ra[i] >= pi/5.0) and (ra[i] <= 2.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
#                label_1.append('Forty-seven')
#    elif(((ra[i] >= 2.0*pi/5.0) and (ra[i] <= 3.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
#                label_1.append('Forty-eight')
#    elif(((ra[i] >= 3.0*pi/5.0) and (ra[i] <= 4.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
#                label_1.append('Forty-nine')
#    elif(((ra[i] >= 4.0*pi/5.0) and (ra[i] <= pi)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
#                label_1.append('Fifty')
            
            
            
# Creating the arrays of time lags 'H1-L1', 'L1-V1' and 'H1-V1'

h1_l1_time = []
l1_v1_time = []
h1_v1_time = []

for i in range(30000):
    h1_l1_time.append(grid[h1_index[i]] - grid[[l1_index[i]]])
    l1_v1_time.append(grid[l1_index[i]] - grid[[v1_index[i]]])
    h1_v1_time.append(grid[h1_index[i]] - grid[[v1_index[i]]])

h1l1_time = np.hstack(h1_l1_time)
l1v1_time = np.hstack(l1_v1_time)
h1v1_time = np.hstack(h1_v1_time)

#----------------------------------------------------FFT------------------------------------------------------------------

# Creating the arrays of the phase lags 'H1-L1', 'L1-V1', 'H1-V1'

h1l1_phase = []
l1v1_phase = []
h1v1_phase = []

for i in range(30000):
    h1l1_phase.append(h1_phase[i] - l1_phase[i])
    l1v1_phase.append(l1_phase[i] - v1_phase[i])
    h1v1_phase.append(h1_phase[i] - v1_phase[i])

    
# Creating the pandas dataframe with the input features and the labels for the neural network

df_1 = {"H1/L1 Sum": h1_l1_sum, "L1/V1 Sum": l1_v1_sum, "H1/V1 Sum": h1_v1_sum, "H1-L1 Phase lag": h1l1_phase, "L1-V1 Phase lag": l1v1_phase, "H1-V1 Phase lag": h1v1_phase, "H1-L1 Time lag": h1l1_time, "L1-V1 Time lag": l1v1_time, "H1-V1 Time lag": h1v1_time, "H1-L1 Frequency": h1l1_freq, "L1-V1 Frequency": l1v1_freq, "H1-V1 Frequency": h1v1_freq, "Sector": label_1}

dataframe = pd.DataFrame(df_1)

# Define column name of the label vector
LABEL = 'SectorEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
dataframe[LABEL] = le.fit_transform(dataframe['Sector'].values.ravel())

# Splitting into input features and labels

X = dataframe.iloc[:, 0:12].values
y = dataframe.iloc[:, 13].values


# Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

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
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
#classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(p = 0.1))

# Adding the third hidden layer
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(p = 0.1))

# Adding the fourth hidden layer
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(p = 0.1))

# Adding the fifth hidden layer
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = num_classes, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set

history = classifier.fit(X_train, y_train_hot, batch_size = 10, epochs = 300)

training_acc = np.asarray(history.history['acc'])
training_loss = np.asarray(history.history['loss'])
print('\nAccuracy on training data: %0.2f' % training_acc[299])
print('\nLoss on training data: %0.2f' % training_loss[299])


y_test = np_utils.to_categorical(y_test, num_classes)

score = classifier.evaluate(X_test, y_test, verbose=1)

print('\nAccuracy on test data: %0.2f' % score[1])
print('\nLoss on test data: %0.2f' % score[0])

#from sklearn.metrics import accuracy_score

#preds = classifier.predict_classes(X_test, verbose=0)
#print('Prediction accuracy: ', accuracy_score(y_test, preds, normalize=True))


