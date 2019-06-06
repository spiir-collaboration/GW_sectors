from __future__ import print_function
from matplotlib import pyplot as plt
#%matplotlib notebook
import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import numpy as np
import pandas as pd
from SampleFileTools1 import SampleFile

obj = SampleFile()
obj.read_hdf("default_freq_500.hdf")
df = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

#obj = SampleFile()
#obj.read_hdf("default_noise_2.hdf")
#df = obj2.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

#df = pd.concat([df1, df2])

# Extracting signals and time columns and storing them in a new dataframe

data = df[['h1_strain', 'l1_strain', 'v1_strain', 'h1_signal', 'l1_signal', 'v1_signal', 'mass1', 'mass2', 'ra','dec','injection_snr']].copy()

seconds_before_event = 0.20
seconds_after_event = 0.05
target_sampling_rate = 2048
sample_length = 0.25

# Create a grid on which the sample can be plotted so that the
    # `event_time` is at position 0
grid = np.linspace(0 - seconds_before_event, 0 + seconds_after_event, int(target_sampling_rate * sample_length))

import numpy as np
h1 = data.iloc[:,0]
l1 = data.iloc[:,1]
v1 = data.iloc[:,2]
h1_signal = data.iloc[:,3]
l1_signal = data.iloc[:,4]
v1_signal = data.iloc[:,5]

# Create subplots for H1, L1 and V1
fig, axes1 = plt.subplots(nrows=3)
axes2 = [ax.twinx() for ax in axes1]
# Plot the strains for H1 and L1
for i, (det_name, det_string) in enumerate([('H1', h1), ('L1', l1), ('V1', v1)]): 
        axes1[i].plot(grid, det_string[500], color='C0')
#        axes1[i].set_xlim(-before, after)
        axes1[i].set_xlim(-seconds_before_event, seconds_after_event)
        axes1[i].tick_params('x', labelsize = 7)
        axes1[i].set_ylim(-300, 300)
        axes1[i].tick_params('y', colors='C0', labelsize=8)
        axes1[i].set_ylabel('Amplitude of\n'
                            'Whitened Strain ({})'.format(det_name), color='C0', fontsize=6)
        
maximum = max(np.max(h1_signal[500]), np.max(l1_signal[500]), np.max(v1_signal[500]))
 
for i, (det_name, det_string1) in enumerate([('H1', h1_signal), ('L1', l1_signal), ('V1', v1_signal)]): 
        axes2[i].plot(grid, det_string1[500]/maximum, color='C1')
#        axes1[i].set_xlim(-before, after)
        axes2[i].set_xlim(-seconds_before_event, seconds_after_event)
#        axes2[i].tick_params('x', labelsize = 7)
        axes2[i].set_ylim(-1.2, 1.2)
        axes2[i].tick_params('y', colors='C1', labelsize=8)
        axes2[i].set_ylabel('Rescaled Amplitude of\n'
                             'Simulated Detector\n'
                            'Signal ({})'.format(det_name), color='C1', fontsize=6)

axes1[0].axvline(x=0, color='black', ls='--', lw=1)
axes1[1].axvline(x=0, color='black', ls='--', lw=1)
axes1[2].axvline(x=0, color='black', ls='--', lw=1)

# Set x-labels
#    axes1[2].set_xticklabels([])
axes1[2].set_xlabel('Time from `event_time` (in seconds)')

plt.savefig('Plot_sample.png')



