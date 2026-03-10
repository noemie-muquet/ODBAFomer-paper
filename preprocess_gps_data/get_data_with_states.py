import pandas as pd
import numpy as np
import os 

# =============================================================================
# inputs : csv files from evaluation dataset with one trip per file (data should be regularized)
# output : one csv file for the entire evaluation dataset which is formatted to fit the HMM 
# given parameters : 
    # - cutoff_len : exclude trips that are too short
# =============================================================================

path = os.getcwd()

data_folder = os.path.join(path,'data','eval_data_timeseries_hmm')
raw_data_loc = os.path.join(data_folder, 'raw_data')

def calculate_longest_distance(df):
    coords = df[['lon', 'lat']].to_numpy()
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    max_distance = np.max(dist_matrix)
    return max_distance

def degrees_to_km(degrees, latitude):
    R = 6371.0 # radius of the Earth in kilometers
    lat_rad = np.radians(latitude)
    km_per_degree_lat = R * np.pi / 180
    km_per_degree_lon = R * np.pi / 180 * np.cos(lat_rad)
    distance_km = degrees * km_per_degree_lat if degrees >= 0 else degrees * km_per_degree_lon
    return distance_km

cutoff_len = 10


df_final = pd.DataFrame(columns=["file name"])

for root, directories, files in os.walk(raw_data_loc):
    for file_name in files :
        
        if file_name.endswith('.csv') :
            csv_file_path = os.path.join(root, file_name)
            df_data = pd.read_csv(csv_file_path)
            
            start_lat = df_data['lat'].iloc[0]
            
            R = calculate_longest_distance(df_data)

            # computes max distance to next during trip
            max_dist = round(degrees_to_km(R, start_lat))
            
            # exit loop if trip too short 
            if max_dist < cutoff_len : 
                continue
            
            for i in range(len(df_data)) :
                df_final = pd.concat([df_final, pd.DataFrame({"file name": [file_name], 
                                                          "datetime": [df_data['datetime'].iloc[i]],
                                                          "lon": [df_data['lon'].iloc[i]], 
                                                          "lat" : [df_data['lat'].iloc[i]], 
                                                          "odba_f" : [df_data['odba_f'].iloc[i]]})], 
                                   ignore_index=True)

file_name = 'hmm_input.csv'
df_final.to_csv(os.path.join(data_folder, file_name), index=False)

#%%

# =============================================================================
# RUN HERE R SCRIPT fit_HMM.R
# =============================================================================

#%%

trajectories = pd.read_csv(os.path.join(data_folder, file_name))
sp = pd.read_csv(os.path.join(data_folder, "states_probabilities.csv")) # R script output

output_path = os.path.join(data_folder, 'eval_trips_with_hmm')
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
state = []
n_1 = 0
n_2 = 0

for i in range(len(sp)) : 
    if sp['p1'].iloc[i] > sp['p2'].iloc[i] : 
        state.append(1)
        n_1 += 1
        
    elif sp['p2'].iloc[i] > sp['p1'].iloc[i] : 
        state.append(2)
        n_2 += 1 
        
sp['state'] = state
trajectories['state'] = state

for name, group in trajectories.groupby('file name'):
    output_file = os.path.join(output_path, f"{name}")
    group.to_csv(output_file, index=False)