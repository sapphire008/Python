# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:55:48 2018

Convert geojson to simple json dict

@author: Edward
"""


import json
# %% plot map
def plot_state_map(states, P_AK=[-77, 5, -0.22, 0.25, 0.35], P_HI=[-38, 17, 0, 0.4, 0.4], exclude_states=['','PR', 'VI', 'DC', 'GU', 'MP', 'AS']):
    xs, ys = states['AK']['lons'],states['AK']['lats']
    #  transoformation
    X = np.array([xs, ys, list(np.ones(len(xs)))])
    A = spm_matrix_2d(P=P_AK)
    Y = A @ X
    states['AK']['lons'] = list(Y[0,:])
    states['AK']['lats'] = list(Y[1,:])
    
    xs, ys = states['HI']['lons'],states['HI']['lats']
    X = np.array([xs, ys, list(np.ones(len(xs)))])
    A = spm_matrix_2d(P=P_HI)
    Y = A @ X
    states['HI']['lons'] = list(Y[0,:])
    states['HI']['lats'] = list(Y[1,:])
    
    #  plot transformation
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(7, 4.5)
    for s in states.keys():
        if s in exclude_states: continue
        xs, ys = states[s]['lons'],states[s]['lats']
        ax.plot(xs, ys)
    return states

# %% states
#fs = '5m'
#file_dir = 'D:/Edward/Documents/Assignments/Scripts/Python/Plots/resource/cb_2017_us_state_%s.geojson'%fs
#with open(file_dir, 'r') as data_file:
#    state_map = json.load(data_file)
#    
#state_map = state_map['features']
#states = {}
#for sm in state_map:
#    name = sm['properties']['NAME']
#    key = sm['properties']['STUSPS']
#    state_id = sm['properties']['STATEFP']
#    coords = sm['geometry']['coordinates']
#    lons_lats = np.empty((0, 2))
#    # Gather the lons and lats
#    for n, c in enumerate(coords):
#        current_coords = np.array(c).squeeze()
#        if key == 'AK' and any(current_coords[:, 0]>0):
#            current_coords[:, 0] = -current_coords[:, 0]
#        lons_lats = np.concatenate((lons_lats, current_coords), axis=0)
#        if n < len(coords)-1:
#            lons_lats = np.concatenate((lons_lats, np.array([[np.nan, np.nan]])), axis=0)
#            
#    lons = list(lons_lats[:, 0])
#    lats = list(lons_lats[:, 1])
#    states[key] = {'name':name, 'id':state_id, 'lons':lons, 'lats':lats}
#
##plot_state_map(states, P_AK)
#if fs == '500k':
#    states_shifted = plot_state_map(states, P_AK=[-70, 9, 0, 0.30, 0.30], P_HI=[-4, 12, 0,0.6, 0.6]) # for 500k resolution only
#else:
#    states_shifted = plot_state_map(states, P_AK=[-70, 9, 0, 0.30, 0.30], P_HI=[51, 7, 0,])
## write file
#file_dir = 'D:/Edward/Documents/Assignments/Scripts/Python/Plots/resource/CBUS_states_scaled_%s.json'%fs
#with open(file_dir,'w') as outfile:
#    json.dump(states_shifted, outfile)

# %% 
def plot_county_map(counties, P_AK=[-77, 5, -0.22, 0.25, 0.35], P_HI=[-38, 17, 0, 0.4, 0.4], exclude_states=['', 'PR', 'VI', 'DC', 'GU', 'MP', 'AS']):
    for key in counties.keys():
        if key[0] == 2 or key[0] == 15: # Alaska or Hawaii
            xs = counties[key]['lons']
            ys = counties[key]['lats']
            X = np.array([xs, ys, list(np.ones(len(xs)))])
            A = spm_matrix_2d(P=P_HI if key[0]==15 else P_AK)
            Y = A @ X
            counties[key]['lons'] = list(Y[0,:])
            counties[key]['lats'] = list(Y[1,:])
        else:
            pass
    
    #  plot transformation
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(7, 4.5)
    for c in counties.keys():
        #print(counties[c]['state_key'])
        if counties[c]['state_key'] in exclude_states: continue
        xs, ys = counties[c]['lons'],counties[c]['lats']
        ax.plot(xs, ys)
    return counties

# %% Counties
fs = '20m'
state_dir = 'D:/Edward/Documents/Assignments/Scripts/Python/Plots/resource/CBUS_states_scaled_20m.json'
county_dir = 'D:/Edward/Documents/Assignments/Scripts/Python/Plots/resource/cb_2017_us_county_%s.geojson'%fs
with open(state_dir, 'r') as data_file:
    states = json.load(data_file)

with open(county_dir, 'r') as data_file:
    county_map = json.load(data_file)

counties = {}
counties_ids = []
county_map = county_map['features']
for cm in county_map:
    name = cm['properties']['NAME']
    county_id = cm['properties']['COUNTYFP']
    state_id = cm['properties']['STATEFP']
    state_name, state_key = '', ''
    for kk in states:
        if states[kk]['id'] == state_id:
            state_name = states[kk]['name']
            state_key = kk
            break
    
    coords = cm['geometry']['coordinates']
    lons_lats = np.empty((0, 2))
    # Gather the lons and lats
    for n, c in enumerate(coords):
        current_coords = np.array(c).squeeze()
        if current_coords.ndim<2:
            current_coords = np.empty((0,2))
            for m, cc in enumerate(c):
                current_coords = np.concatenate((current_coords, cc), axis=0)
                if  m < len(c)-1:
                    current_coords = np.concatenate((current_coords, np.array([[np.nan, np.nan]])), axis=0)
        if state_key == 'AK' and any(current_coords[:, 0]>0):
            current_coords[:, 0] = -current_coords[:, 0]
        lons_lats = np.concatenate((lons_lats, current_coords), axis=0)
        if n < len(coords)-1:
            lons_lats = np.concatenate((lons_lats, np.array([[np.nan, np.nan]])), axis=0)

    lons = list(lons_lats[:, 0])
    lats = list(lons_lats[:, 1])
    counties[(int(state_id), int(county_id))] = {'name':name, 'id': county_id, 'state': state_name, 
                'state_key': state_key, 'lons':lons, 'lats':lats}
if fs == '500k':
    counties_scaled = plot_county_map(counties, P_AK=[-70, 9, 0, 0.30, 0.30], P_HI=[-4, 12, 0,0.6, 0.6]) # for 500k resolution only
else:
    counties_scaled = plot_county_map(counties, P_AK=[-70, 9, 0, 0.30, 0.30], P_HI=[51, 7, 0,])
    
# write file
file_dir = 'D:/Edward/Documents/Assignments/Scripts/Python/Plots/resource/CBUS_counties_scaled_%s.json'%fs
def remap_keys(mapping, func=str): #json can't take tuple keys, only string keys
    return {func(k):v for k,v in mapping.items()}

with open(file_dir,'w') as outfile:
    json.dump(remap_keys(counties_scaled), outfile)
    
# %%
with open(file_dir, 'r') as outfile:
    counties_loaded = remap_keys(json.load(outfile), func=lambda x: tuple(str2num(x)))
    