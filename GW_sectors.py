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
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['GOTO_NUM_THREADS'] = '24'
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['openmp'] = 'True'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

import numpy as np
import pandas as pd
from SampleFileTools1 import SampleFile

obj1 = SampleFile()
obj1.read_hdf("default_diff.hdf")
df1 = obj1.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

obj2 = SampleFile()
obj2.read_hdf("default_diff_1.hdf")
df2 = obj2.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

df = pd.concat([df1, df2], ignore_index= True)

# Extracting signals and time columns and storing them in a new dataframe

data = df[['h1_strain', 'l1_strain', 'v1_strain', 'event_time', 'injection_snr']].copy()

# Extracting the index of the maximum value of strain for each detector from every sample.

from scipy.signal import hilbert, chirp
import numpy as np
h1 = data.iloc[:,0]
l1 = data.iloc[:,1]
v1 = data.iloc[:,2]
#snr = data.iloc[:,4]
#timestamp = data.iloc[:,3]

#h1_index = []
#l1_index = []
#v1_index = []
#for i in range(100000):
#    h1_index.append(np.argmax(np.abs(hilbert(h1[i]))))
#    l1_index.append(np.argmax(np.abs(hilbert(l1[i]))))
#    v1_index.append(np.argmax(np.abs(hilbert(v1[i]))))

# Extracting the maximum strain value of each detector for every sample
#h1_max = []
#l1_max = []
#v1_max = []
#for i in range(100000):
#    h1_max.append(np.abs(hilbert(h1[i])).max())
#    l1_max.append(np.abs(hilbert(l1[i])).max())
#    v1_max.append(np.abs(hilbert(v1[i])).max())

# Creating ratio of amplitudes 'H1/L1', 'L1/V1', 'H1/V1'

#h1_l1_amp = []
#l1_v1_amp = []
#h1_v1_amp = []

#for j in range(100000):
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

for i in range(200000):
    h1_sum = 0.0
    l1_sum = 0.0
    v1_sum = 0.0
    h1_des = np.abs(hilbert(h1[i]))
    l1_des = np.abs(hilbert(l1[i]))
    v1_des = np.abs(hilbert(v1[i]))
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

#for i in range(100000):
#    h1_amp.append(((np.abs(np.fft.fft(h1[i])))**2.0).max())
#    l1_amp.append(((np.abs(np.fft.fft(l1[i])))**2.0).max())
#    v1_amp.append(((np.abs(np.fft.fft(v1[i])))**2.0).max())

# Creating ratios of maximum FFT amplitudes for each sample

#h1_l1_amp = []
#l1_v1_amp = []
#h1_v1_amp = []

#for i in range(100000):
#    h1_l1_amp.append(h1_amp[i]/l1_amp[i])
#    l1_v1_amp.append(l1_amp[i]/v1_amp[i])
#    h1_v1_amp.append(h1_amp[i]/v1_amp[i])

#from scipy.signal import welch

#def get_psd_values(y_values, T, N, f_s):
#    f_values, psd_values = welch(y_values, fs=f_s)
#    return f_values, psd_values


#t_n = 0.25
#N = 512
#T = t_n / N
#f_s = 1/T


#h1_sum_psd = 0.0
#l1_sum_psd = 0.0
#v1_sum_psd = 0.0

#h1_l1_amp = []
#l1_v1_amp = []
#h1_v1_amp = []

#for i in range(15000):
#    h1_sum_psd = 0.0
#    l1_sum_psd = 0.0
#    v1_sum_psd = 0.0
#    f_values_1, psd_values_1 = get_psd_values(h1[i], T, N, f_s)
#    f_values_2, psd_values_2 = get_psd_values(l1[i], T, N, f_s)
#    f_values_3, psd_values_3 = get_psd_values(v1[i], T, N, f_s)
#    psd_values_1.sort()
#    psd_values_2.sort()
#    psd_values_3.sort()
#    for j in range(5):
#        h1_sum_psd = h1_sum_psd + psd_values_1[128-j]
#        l1_sum_psd = l1_sum_psd + psd_values_2[128-j]
#        v1_sum_psd = v1_sum_psd + psd_values_3[128-j]
        
#    h1_l1_amp.append(h1_sum_psd/l1_sum_psd)
#    l1_v1_amp.append(l1_sum_psd/v1_sum_psd)
#    h1_v1_amp.append(h1_sum_psd/v1_sum_psd)
#    f_values_1, psd_values_1 = get_psd_values(h1[i], T, N, f_s)
#    f_values_2, psd_values_2 = get_psd_values(l1[i], T, N, f_s)
#    f_values_3, psd_values_3 = get_psd_values(v1[i], T, N, f_s)

#    h1_l1_amp.append(psd_values_1.max()/psd_values_2.max())
#    l1_v1_amp.append(psd_values_2.max()/psd_values_3.max())
#    h1_v1_amp.append(psd_values_1.max()/psd_values_3.max())
    
    
 








# Extracting the index of the maximum amplitude for each sample after taking FFTs    
    
#h1_index_f = []
#l1_index_f = []
#v1_index_f = []

#for i in range(100000):
#    h1_index_f.append(np.argmax(np.abs(np.fft.fft(h1[i]))))
#    l1_index_f.append(np.argmax(np.abs(np.fft.fft(l1[i]))))
#    v1_index_f.append(np.argmax(np.abs(np.fft.fft(v1[i]))))
    
# Extracting the phases corresponding to the maximum amplitudes for each sample after taking the FFTs

#h1_phase = []
#l1_phase = []
#v1_phase = []
#for i in range(100000):
#    a = np.angle(np.fft.fft(h1[i]))
#    b = np.angle(np.fft.fft(l1[i]))
#    c = np.angle(np.fft.fft(v1[i]))
#    h1_phase.append(a[h1_index_f[i]])
#    l1_phase.append(b[l1_index_f[i]])
#    v1_phase.append(c[v1_index_f[i]])
    
# Extracting the frequencies corresponding to the maximum FFT amplitudes for each sample

#h1l1_freq = []
#l1v1_freq = []
#h1v1_freq = []
#time_step = 1/2048
#for i in range(100000):
#    d = np.fft.fftfreq(h1[i].size, time_step)
#    e = np.fft.fftfreq(l1[i].size, time_step)
#    f = np.fft.fftfreq(v1[i].size, time_step)
#    h1l1_freq.append(np.abs(d[h1_index_f[i]] - e[l1_index_f[i]]))
#    l1v1_freq.append(np.abs(e[l1_index_f[i]] - f[v1_index_f[i]]))
#    h1v1_freq.append(np.abs(d[h1_index_f[i]] - f[v1_index_f[i]]))



# Creating the time array

timestamp = []
event_time = 1234567936
count = 0
for i in range(512):
#    grid.append(np.linspace(event_time - sbe[i], event_time + (2.0 - sbe[i]), int(2048 * 0.4)))
    grid = np.linspace(event_time - 0.20, event_time + 0.05, int(2048 * 0.25))
    
    
# Extracting the angles from the dataframe

angles = df[['ra', 'dec']].copy()
#ra = angles.iloc[:,0].values
#dec = angles.iloc[:,1].values

ra = 2.0*np.pi*angles['ra']
dec = np.arcsin(1.0 - 2.0*angles['dec'])



#Assigning the labels, based on ra and dec angles
#label_1 = []
#pi = 3.141593
#for i in range(200000):

# First surface

#    if(((ra[i] >= pi) and (ra[i] <= 4.0*pi/3.0)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
#                label_1.append('One')
#    elif(((ra[i] >= 4.0*pi/3.0) and (ra[i] <= 5.0*pi/3.0)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Two')
#    elif(((ra[i] >= 5.0*pi/3.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Three')
#    elif(((ra[i] >= pi) and (ra[i] <= 4.0*pi/3.0)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
#                label_1.append('Four')
#    elif(((ra[i] >= 4.0*pi/3.0) and (ra[i] <= 5.0*pi/3.0)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
#                label_1.append('Five')
#    elif(((ra[i] >= 5.0*pi/3.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
#                label_1.append('Six')
#    elif(((ra[i] >= pi) and (ra[i] <= 4.0*pi/3.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
#                label_1.append('Seven')
#    elif(((ra[i] >= 4.0*pi/3.0) and (ra[i] <= 5.0*pi/3.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
#                label_1.append('Eight')
#    elif(((ra[i] >= 5.0*pi/3.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
#                label_1.append('Nine')
            

# Second surface

#    elif(((ra[i] >= 2.0*pi/3.0) and (ra[i] <= pi)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Ten')
#    elif(((ra[i] >= pi/3.0) and (ra[i] <= 2.0*pi/3.0)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Eleven')
#    elif(((ra[i] >= 0.0) and (ra[i] <= pi/3.0)) and ((dec[i] >= pi/6.0) and (dec[i] <= pi/2.0))):
#                label_1.append('Twelve')
#    elif(((ra[i] >= 2.0*pi/3.0) and (ra[i] <= pi)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
#                label_1.append('Thirteen')
#    elif(((ra[i] >= pi/3.0) and (ra[i] <= 2.0*pi/3.0)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
#                label_1.append('Fourteen')
#    elif(((ra[i] >= 0.0) and (ra[i] <= pi/3.0)) and ((dec[i] >= -pi/6.0) and (dec[i] <= pi/6.0))):
#                label_1.append('Fifteen')
#    elif(((ra[i] >= 2.0*pi/3.0) and (ra[i] <= pi)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
#                label_1.append('Sixteen')
#    elif(((ra[i] >= pi/3.0) and (ra[i] <= 2.0*pi/3.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
#                label_1.append('Seventeen')
#    elif(((ra[i] >= 0.0) and (ra[i] <= pi/3.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -pi/6.0))):
#                label_1.append('Eighteen')        
    
label_1 = []
pi = 3.141593
for i in range(200000):

# First slice

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
                
# Second slice

    elif(((ra[i] >= pi) and (ra[i] <= 6.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
                label_1.append('Eleven')
    elif(((ra[i] >= 6.0*pi/5.0) and (ra[i] <= 7.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
                label_1.append('Twelve')
    elif(((ra[i] >= 7.0*pi/5.0) and (ra[i] <= 8.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
                label_1.append('Thirteen')
    elif(((ra[i] >= 8.0*pi/5.0) and (ra[i] <= 9.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
                label_1.append('Fourteen')
    elif(((ra[i] >= 9.0*pi/5.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
               label_1.append('Fifteen')
    elif(((ra[i] >= 0.0) and (ra[i] <= pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
                label_1.append('Sixteen')
    elif(((ra[i] >= pi/5.0) and (ra[i] <= 2.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
               label_1.append('Seventeen')
    elif(((ra[i] >= 2.0*pi/5.0) and (ra[i] <= 3.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
                label_1.append('Eighteen')
    elif(((ra[i] >= 3.0*pi/5.0) and (ra[i] <= 4.0*pi/5.0)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
                label_1.append('Nineteen')
    elif(((ra[i] >= 4.0*pi/5.0) and (ra[i] <= pi)) and ((dec[i] >= pi/10.0) and (dec[i] <= 3.0*pi/10.0))):
                label_1.append('Twenty')
                
# Third slice

    elif(((ra[i] >= pi) and (ra[i] <= 6.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
                label_1.append('Twenty-one')
    elif(((ra[i] >= 6.0*pi/5.0) and (ra[i] <= 7.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
                label_1.append('Twenty-two')
    elif(((ra[i] >= 7.0*pi/5.0) and (ra[i] <= 8.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
                label_1.append('Twenty-three')
    elif(((ra[i] >= 8.0*pi/5.0) and (ra[i] <= 9.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
                label_1.append('Twenty-four')
    elif(((ra[i] >= 9.0*pi/5.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
                label_1.append('Twenty-five')
    elif(((ra[i] >= 0.0) and (ra[i] <= pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
                label_1.append('Twenty-six')
    elif(((ra[i] >= pi/5.0) and (ra[i] <= 2.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
                label_1.append('Twenty-seven')
    elif(((ra[i] >= 2.0*pi/5.0) and (ra[i] <= 3.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
                label_1.append('Twenty-eight')
    elif(((ra[i] >= 3.0*pi/5.0) and (ra[i] <= 4.0*pi/5.0)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
                label_1.append('Twenty-nine')
    elif(((ra[i] >= 4.0*pi/5.0) and (ra[i] <= pi)) and ((dec[i] >= -pi/10.0) and (dec[i] <= pi/10.0))):
                label_1.append('Thirty')
                
# Fourth slice

    elif(((ra[i] >= pi) and (ra[i] <= 6.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
                label_1.append('Thirty-one')
    elif(((ra[i] >= 6.0*pi/5.0) and (ra[i] <= 7.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
                label_1.append('Thirty-two')
    elif(((ra[i] >= 7.0*pi/5.0) and (ra[i] <= 8.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
                label_1.append('Thirty-three')
    elif(((ra[i] >= 8.0*pi/5.0) and (ra[i] <= 9.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
                label_1.append('Thirty-four')
    elif(((ra[i] >= 9.0*pi/5.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
                label_1.append('Thirty-five')
    elif(((ra[i] >= 0.0) and (ra[i] <= pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
                label_1.append('Thirty-six')
    elif(((ra[i] >= pi/5.0) and (ra[i] <= 2.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
                label_1.append('Thirty-seven')
    elif(((ra[i] >= 2.0*pi/5.0) and (ra[i] <= 3.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
                label_1.append('Thirty-eight')
    elif(((ra[i] >= 3.0*pi/5.0) and (ra[i] <= 4.0*pi/5.0)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
                label_1.append('Thirty-nine')
    elif(((ra[i] >= 4.0*pi/5.0) and (ra[i] <= pi)) and ((dec[i] >= -3.0*pi/10.0) and (dec[i] <= -pi/10.0))):
                label_1.append('Forty')
                
# Fifth slice

    elif(((ra[i] >= pi) and (ra[i] <= 6.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
                label_1.append('Forty-one')
    elif(((ra[i] >= 6.0*pi/5.0) and (ra[i] <= 7.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
                label_1.append('Forty-two')
    elif(((ra[i] >= 7.0*pi/5.0) and (ra[i] <= 8.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
                label_1.append('Forty-three')
    elif(((ra[i] >= 8.0*pi/5.0) and (ra[i] <= 9.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
                label_1.append('Forty-four')
    elif(((ra[i] >= 9.0*pi/5.0) and (ra[i] <= 2.0*pi)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
                label_1.append('Forty-five')
    elif(((ra[i] >= 0.0) and (ra[i] <= pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
                label_1.append('Forty-six')
    elif(((ra[i] >= pi/5.0) and (ra[i] <= 2.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
                label_1.append('Forty-seven')
    elif(((ra[i] >= 2.0*pi/5.0) and (ra[i] <= 3.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
                label_1.append('Forty-eight')
    elif(((ra[i] >= 3.0*pi/5.0) and (ra[i] <= 4.0*pi/5.0)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
                label_1.append('Forty-nine')
    elif(((ra[i] >= 4.0*pi/5.0) and (ra[i] <= pi)) and ((dec[i] >= -pi/2.0) and (dec[i] <= -3.0*pi/10.0))):
                label_1.append('Fifty')
            
            
            
# Creating the arrays of time lags 'H1-L1', 'L1-V1' and 'H1-V1'

#h1_l1_time = []
#l1_v1_time = []
#h1_v1_time = []

#for i in range(100000):
#    h1_l1_time.append(grid[h1_index[i]] - grid[[l1_index[i]]])
#    l1_v1_time.append(grid[l1_index[i]] - grid[[v1_index[i]]])
#    h1_v1_time.append(grid[h1_index[i]] - grid[[v1_index[i]]])

#h1l1_time = np.hstack(h1_l1_time)
#l1v1_time = np.hstack(l1_v1_time)
#h1v1_time = np.hstack(h1_v1_time)

from scipy.signal import correlate

h1l1_time = []
l1v1_time = []
h1v1_time = []

h1l1_time_h = []
l1v1_time_h = []
h1v1_time_h = []

N = 512

time = np.arange(1-N,N)

for i in range(200000):
    h1l1_time.append(time[correlate(h1[i],l1[i]).argmax()])
    l1v1_time.append(time[correlate(l1[i],v1[i]).argmax()])
    h1v1_time.append(time[correlate(h1[i],v1[i]).argmax()])

for i in range(200000):
    h1_h = hilbert(h1[i])
    l1_h = hilbert(l1[i])
    v1_h = hilbert(v1[i])
    h1l1_time_h.append(time[correlate(np.abs(h1_h),np.abs(l1_h)).argmax()])
    l1v1_time_h.append(time[correlate(np.abs(l1_h),np.abs(v1_h)).argmax()])
    h1v1_time_h.append(time[correlate(np.abs(h1_h),np.abs(v1_h)).argmax()])
    
    
h1l1_sum = []
l1v1_sum = []
h1v1_sum = []

h1l1_sum_h = []
l1v1_sum_h = []
h1v1_sum_h = []


for i in range(200000):
    h1l1_sum.append(correlate(h1[i],l1[i]).max())
    l1v1_sum.append(correlate(l1[i],v1[i]).max())
    h1v1_sum.append(correlate(h1[i],v1[i]).max())


for i in range(200000):
    h1_h = hilbert(h1[i])
    l1_h = hilbert(l1[i])
    v1_h = hilbert(v1[i])
    h1l1_sum_h.append(correlate(np.abs(h1_h),np.abs(l1_h)).max())
    l1v1_sum_h.append(correlate(np.abs(l1_h),np.abs(v1_h)).max())
    h1v1_sum_h.append(correlate(np.abs(h1_h),np.abs(v1_h)).max())
    
    
    
#from scipy.signal import hilbert, chirp

h1l1_phase = []
l1v1_phase = []
h1v1_phase = []

h1l1_freq = []
l1v1_freq = []
h1v1_freq = []

h1_10 = []
l1_10 = []
v1_10 = []

fs = 2048.0

for i in range(200000):
    h1_10 = []
    l1_10 = []
    v1_10 = []
    h1_des = hilbert(h1[i])
    l1_des = hilbert(l1[i])
    v1_des = hilbert(v1[i])
    h1_des.sort()
    l1_des.sort()
    v1_des.sort()
    for j in range(10):
        h1_10.append(h1_des[511-j])
        l1_10.append(l1_des[511-j])
        v1_10.append(v1_des[511-j])


    h1l1_phase.append(np.average((np.unwrap(np.angle(h1_10))) - (np.unwrap(np.angle(l1_10)))))
    l1v1_phase.append(np.average((np.unwrap(np.angle(l1_10))) - (np.unwrap(np.angle(v1_10)))))
    h1v1_phase.append(np.average((np.unwrap(np.angle(h1_10))) - (np.unwrap(np.angle(v1_10)))))
    
#    h1l1_freq.append(np.average(np.diff((np.unwrap(np.angle(h1_10))) - (np.unwrap(np.angle(l1_10))))/(2.0*np.pi)*fs))
#    l1v1_freq.append(np.average(np.diff((np.unwrap(np.angle(l1_10))) - (np.unwrap(np.angle(v1_10))))/(2.0*np.pi)*fs))
#    h1v1_freq.append(np.average(np.diff((np.unwrap(np.angle(h1_10))) - (np.unwrap(np.angle(v1_10))))/(2.0*np.pi)*fs))
    
    


#for i in range(100000):
#    h1l1_phase.append(np.average(np.unwrap(np.angle(hilbert(h1[i])))) - np.unwrap(np.angle(hilbert(l1[i]))))
#    l1v1_phase.append(np.average(np.unwrap(np.angle(hilbert(l1[i])))) - np.unwrap(np.angle(hilbert(v1[i]))))
#    h1v1_phase.append(np.average(np.unwrap(np.angle(hilbert(h1[i])))) - np.unwrap(np.angle(hilbert(v1[i]))))
    
    
h1l1_angle = []
l1v1_angle = []
h1v1_angle = []
    
for i in range(200000):
    h1_h = hilbert(h1[i])
    l1_h = hilbert(l1[i])
    v1_h = hilbert(v1[i])
    c_h1l1 = np.inner(h1_h, np.conj(l1_h))/ np.sqrt(np.inner(h1_h, np.conj(h1_h)) * np.inner(l1_h, np.conj(l1_h)))
    c_l1v1 = np.inner(l1_h, np.conj(v1_h))/ np.sqrt(np.inner(l1_h, np.conj(l1_h)) * np.inner(v1_h, np.conj(v1_h)))
    c_h1v1 = np.inner(h1_h, np.conj(v1_h))/ np.sqrt(np.inner(h1_h, np.conj(h1_h)) * np.inner(v1_h, np.conj(v1_h)))
    h1l1_angle.append(np.angle(c_h1l1))
    l1v1_angle.append(np.angle(c_l1v1))
    h1v1_angle.append(np.angle(c_h1v1))
        

        
#----------------------------------------------------FFT------------------------------------------------------------------

# Creating the arrays of the phase lags 'H1-L1', 'L1-V1', 'H1-V1'

#h1l1_phase = []
#l1v1_phase = []
#h1v1_phase = []

#for i in range(100000):
#    h1l1_phase.append(h1_phase[i] - l1_phase[i])
#    l1v1_phase.append(l1_phase[i] - v1_phase[i])
#    h1v1_phase.append(h1_phase[i] - v1_phase[i])

    
# Creating the pandas dataframe with the input features and the labels for the neural network

df_1 = {"H1/L1": h1_l1_sum, "L1/V1": l1_v1_sum, "H1/V1": h1_v1_sum, "H1-L1 Time lag": h1l1_time, "L1-V1 Time lag": l1v1_time, "H1-V1 Time lag": h1v1_time, "H1-L1 Phase lag": h1l1_phase, "L1-V1 Phase lag": l1v1_phase, "H1-V1 Phase lag": h1v1_phase, "H1/L1 Corr": h1l1_sum, "L1/V1 Corr": l1v1_sum, "H1/V1 Corr": h1v1_sum, "H1-L1 Angle": h1l1_angle, "L1-V1 Angle": l1v1_angle, "H1-V1 Angle": h1v1_angle, "H1/L1 Corr Hilbert": h1l1_sum_h, "L1/V1 Corr Hilbert": l1v1_sum_h, "H1/V1 Corr Hilbert": h1v1_sum_h, "H1-L1 Time Hilbert": h1l1_time_h, "L1-V1 Time Hilbert": l1v1_time_h, "H1-V1 Time Hilbert": h1v1_time_h, "Sector": label_1}

dataframe = pd.DataFrame(df_1)

# Define column name of the label vector
LABEL = 'SectorEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
dataframe[LABEL] = le.fit_transform(dataframe['Sector'].values.ravel())

# Splitting into input features and labels

X = dataframe.iloc[:, 0:21].values
y = dataframe.iloc[:, 22].values


# Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

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
classifier.add(Dense(units = 200, kernel_initializer = 'uniform', input_dim = 21))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(p = 0.2))

# Adding the second hidden layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(p = 0.2))

# Adding the third layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(p = 0.2))

# Adding the fourth layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(p = 0.2))

# Adding the fifth layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(p = 0.2))

# Adding the sixth layer
#classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
#classifier.add(BatchNormalization())
#classifier.add(Activation('relu'))
#classifier.add(Dropout(p = 0.2))

# Adding the seventh layer
#classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
#classifier.add(BatchNormalization())
#classifier.add(Activation('relu'))
#classifier.add(Dropout(p = 0.2))

# Adding the eighth layer
#classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
#classifier.add(BatchNormalization())
#classifier.add(Activation('relu'))
#classifier.add(Dropout(p = 0.2))

# Adding the ninth hidden layer
#classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
#classifier.add(BatchNormalization())
#classifier.add(Activation('relu'))
#classifier.add(Dropout(p = 0.2))

# Adding the tenth hidden layer
#classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
#classifier.add(BatchNormalization())
#classifier.add(Activation('relu'))
#classifier.add(Dropout(p = 0.2))

# Adding the eleventh hidden layer
#classifier.add(Dense(units = 500, kernel_initializer = 'uniform'))
#classifier.add(BatchNormalization())
#classifier.add(Activation('relu'))
#classifier.add(Dropout(p = 0.1))

# Adding the twelfth hidden layer
#classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
#classifier.add(BatchNormalization())
#classifier.add(Activation('relu'))
#classifier.add(Dropout(p = 0.35))

# Adding the output layer
classifier.add(Dense(units = num_classes, kernel_initializer = 'uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('softmax'))


#callbacks_list = [
#    keras.callbacks.ModelCheckpoint(
#        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
#        monitor='val_loss', save_best_only=True),
#    keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=35)
#]

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set

#history = classifier.fit(X_train, y_train_hot, batch_size = 2000, epochs = 300, callbacks = callbacks_list, validation_split = 0.2, verbose=1)

history = classifier.fit(X_train, y_train_hot, batch_size = 2000, epochs = 5, verbose=1)



training_acc = np.asarray(history.history['acc'])
training_loss = np.asarray(history.history['loss'])
print('\nAccuracy on training data: %0.2f' % training_acc[-1])
print('\nLoss on training data: %0.2f' % training_loss[-1])


y_test_hot = np_utils.to_categorical(y_test, num_classes)

#y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
#print('\nNumber of correct predictions:' % len(y_pred))


score = classifier.evaluate(X_test, y_test_hot, verbose=1)

print('\nAccuracy on test data: %0.2f' % score[1])
print('\nLoss on test data: %0.2f' % score[0])


y_pred_test = classifier.predict(X_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test_hot, axis=1)

np.savetxt('Scores.out', np.transpose([max_y_pred_test,max_y_test]))

from sklearn.metrics import accuracy_score

print('Prediction accuracy:', accuracy_score(max_y_test, max_y_pred_test))

 







    
    
    
    
    













    
    
    
    
    






























