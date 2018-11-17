# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:55:58 2018

@author: Edward
"""

from bokeh.sampledata.us_states import data as states
xs = states['AK']['lons']
xs = np.array(xs)
xs[xs>0] = -xs[xs>0]
states['AK']['lons'] = list(xs)
import json
file_dir = 'D:/Edward/Documents/Assignments/Scripts/Python/Plots/resource/US_states.json'
with open(file_dir,'w') as outfile:
    json.dump(states, outfile)
    
# %%
import json
file_dir = 'D:/Edward/Documents/Assignments/Scripts/Python/Plots/resource/US_states.json'
with open(file_dir, 'r') as data_file:
    states = json.load(data_file)
    
xs, ys = states['AK']['lons'],states['AK']['lats']

#  transoformation
X = np.array([xs, ys, list(np.ones(len(xs)))])
A = spm_matrix_2d(P=[-77, 5, 0, 0.25, 0.35])
Y = A @ X
states['AK']['lons'] = list(Y[0,:])
states['AK']['lats'] = list(Y[1,:])

xs, ys = states['HI']['lons'],states['HI']['lats']
X = np.array([xs, ys, list(np.ones(len(xs)))])
A = spm_matrix_2d(P=[-38, 17, 0, 0.4, 0.4])
Y = A @ X
states['HI']['lons'] = list(Y[0,:])
states['HI']['lats'] = list(Y[1,:])

#  plot transformation
fig, ax = plt.subplots(1,1)
fig.set_size_inches(7, 4.5)
for s in states.keys():
    xs, ys = states[s]['lons'],states[s]['lats']
    ax.plot(xs, ys)

    
# %% save a copy
file_dir = 'D:/Edward/Documents/Assignments/Scripts/Python/Plots/resource/US_states_scaled.json'
with open(file_dir,'w') as outfile:
    json.dump(states, outfile)


# %%
from bokeh.sampledata.us_counties import data as counties
