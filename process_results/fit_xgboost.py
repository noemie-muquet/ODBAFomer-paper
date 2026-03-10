import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import os 
from scipy.stats import wasserstein_distance
import xarray as xr 
from utils.utils import degrees_to_km, calculate_longest_distance, get_step_length, get_turning_angle

# =============================================================================
# inputs : csv files, one trip per file, regularized
# outputs : ODBA infered by XGBoost
# given parameters : 
    # - cutoff_len : minimal max distance of a trip to be processed
    # - cutoff_step_size : maximal step size accepted (to filter out trajectories with anomalies)
# =============================================================================

# fixes minimal distance to colony for a trip to be processed 
cutoff_len = 10

# fixes a maximum value of step size to filter out trajectories anomalies
cutoff_step_size = 20000

# initializes input and outputs directories 
path = os.getcwd()

data_folder = os.path.join(path, 'data','XGBoost_data')
train_folder = os.path.join(data_folder,'train_data')
test_folder = os.path.join(path, 'data', 'eval_data_timeseries_hmm', 'eval_trips_with_hmm')

results_folder = os.path.join(data_folder, 'XGBoost_outputs')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
covariates_folder = os.path.join(path, 'data', 'train_data', 'covariates')

# bathymetry maps paths 
abr_path = os.path.join(covariates_folder, 'bathy_gebco_abr_1080_1080.nc')
fdn_path = os.path.join(covariates_folder, 'bathy_gebco_fdn_1080_1080.nc')
spsp_path = os.path.join(covariates_folder, 'bathy_gebco_spsp_1080_1080.nc')

test_bathy = fdn_path

# =============================================================================
# prepare data for inference 
# =============================================================================

# count files (to check against ODBAFormer dataset size)
N_files = 0

# initatite dastaset 
all_bathy = []
df_all = pd.DataFrame(columns = ['lon', 'lat', 'odba_f', 'step_length', 'turning_angle'])

for file_name in os.listdir(train_folder):
    if file_name.endswith('.csv'):  
        data_path = os.path.join(train_folder, file_name)
        df_raw = pd.read_csv(data_path)
        df = df_raw[["datetime", "lon", "lat", "odba_f"]]
        
        start_lat = df['lat'].iloc[0]
        start_lon = df['lon'].iloc[0]
        
        # if no odba on file, don't process it 
        if sum(df['odba_f']) == 0 :
            continue
        
        # open bathymetry
        if np.floor(start_lat) == -18 and np.floor(start_lon) == -39 : 
            data = xr.open_dataset(abr_path)
            
        elif np.floor(start_lat) == -4 and np.floor(start_lon) == -33 :
            data = xr.open_dataset(fdn_path)

        elif np.floor(start_lat) == 0 and np.floor(start_lon) == -30 :
            data = xr.open_dataset(spsp_path)

        # max distance to nest to compute scaling factor
        R = calculate_longest_distance(df)
        
        # computes max distance to next during trip
        max_dist = round(degrees_to_km(R, start_lat))
        
        # exit loop if trip too short 
        if max_dist < cutoff_len : 
            continue
        
        df['step_length'] = get_step_length(df)
        df['turning_angle'] = get_turning_angle(df)
        
        df = df.dropna()
        
        # get bathy value under each gps fix 
        bathymetry = data['elevation']
        bathy = bathymetry.values 
        max_bathy = np.min(bathy)
        latitudes = data['lat'].values
        longitudes = data['lon'].values
        
        deltax = longitudes[2] - longitudes[1]
        deltay = latitudes[2] - latitudes [1]
        exit = False
        bathy_traj = [] 
        for i in range(len(df)):
            x = df['lon'].iloc[i]
            y = df['lat'].iloc[i]
            x_i = np.where((longitudes >= x - deltax) & (longitudes <= x + deltax))
            y_i = np.where((latitudes >= y - deltay) & (latitudes <= y + deltay))
        
            if len(x_i[0]) == 0 or len(y_i[0]) == 0:
                exit = True
                break
        
            x_i = x_i[0][0]
            y_i = y_i[0][0]
            bathy_i = bathy[x_i, y_i]/max_bathy
            bathy_traj.append(bathy_i)
        
        if exit:
            continue  
        
        all_bathy.extend(bathy_traj)    
        
        df_all = pd.concat([df_all, df])
        
        N_files += 1

df_all['bathy'] = all_bathy 
print(f"Dataset size = {N_files}")

# =============================================================================
# initiate and fit XGBoost 
# =============================================================================

X = df_all[['step_length', 'turning_angle', 'bathy']]
y = df_all['odba_f']

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
    )
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# =============================================================================
# test XGBoost 
# =============================================================================

pred_sum_list = []
real_sum_list = []
pred_sum_1_list = []
real_sum_1_list = []
pred_sum_2_list = []
real_sum_2_list = []
emd_list = []

pred_values = []
real_values = []

for file_name in os.listdir(test_folder):
    if file_name.endswith('.csv'):  
        
        # outlier removal
        if 'BRA_FDN_ME_2019-04-16_SDAC_03_NA_M_GPS_AXY_RT01_UTC.csv_trip_2.csv' in file_name :
            continue
        
        data_path = os.path.join(test_folder, file_name)
        df_test_raw = pd.read_csv(data_path)
        df_test = df_test_raw[["lon", "lat", "odba_f", "state"]]

        real_values_indiv = df_test["odba_f"].to_numpy()
        real_values.extend(real_values_indiv.tolist())

        df_test['step_length'] = get_step_length(df_test)
        df_test['turning_angle'] = get_turning_angle(df_test) 
        
        df_test = df_test.dropna()
        
        start_lat = df_test['lat'].iloc[0]
        start_lon = df_test['lon'].iloc[0]
        
        data = xr.open_dataset(test_bathy)
            
        bathymetry = data['elevation']
        bathy = bathymetry.values 
        max_bathy = np.min(bathy)
        latitudes = data['lat'].values
        longitudes = data['lon'].values
        
        deltax = longitudes[2] - longitudes[1]
        deltay = latitudes[2] - latitudes [1]
        
        bathy_traj = []  
        
        for i in range(len(df_test)):
            x = df_test['lon'].iloc[i]
            y = df_test['lat'].iloc[i]
            x_i = np.where((longitudes >= x - deltax) & (longitudes <= x + deltax))
            y_i = np.where((latitudes >= y - deltay) & (latitudes <= y + deltay))
            x_i = x_i[0][0]
            y_i = y_i[0][0]
            bathy_i = bathy[x_i, y_i]/max_bathy
            bathy_traj.append(bathy_i)
            
       
        df_test['bathy'] = bathy_traj 

        X = df_test[['step_length', 'turning_angle', 'bathy']]
        y = df_test['odba_f']
        
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        plt.figure(figsize = (8,6))
        plt.plot(df_test['odba_f'])
        plt.plot(y_pred)
        plt.show()
        
        pred_values.extend(y_pred.tolist())
        
        df_test['y_pred'] = y_pred
        
        real_sum = df_test['odba_f'].sum()
        pred_sum = df_test['y_pred'].sum()
        
        real_sum_state_1 = df_test[df_test['state'] == 1]['odba_f'].sum()
        pred_sum_state_1 = df_test[df_test['state'] == 1]['y_pred'].sum()
        
        real_sum_state_2 = df_test[df_test['state'] == 2]['odba_f'].sum()
        pred_sum_state_2 = df_test[df_test['state'] == 2]['y_pred'].sum()
    
        emd = wasserstein_distance(df_test["odba_f"], df_test['y_pred'])
        
        pred_sum_list.append(pred_sum/1000)
        real_sum_list.append(real_sum/1000)
        pred_sum_1_list.append(pred_sum_state_1/1000)
        real_sum_1_list.append(real_sum_state_1/1000)
        pred_sum_2_list.append(pred_sum_state_2/1000)
        real_sum_2_list.append(real_sum_state_2/1000)
        emd_list.append(emd)
        
        df_final = df_test[['odba_f', 'y_pred']]
        df_final.to_csv(os.path.join(results_folder, file_name), index=False)

        
metrics = pd.DataFrame({'real sum' : real_sum_list, 
                        'pred sum' : pred_sum_list,
                        'real sum state 1' : real_sum_1_list,
                        'pred sum state 1' : pred_sum_1_list,
                        'real sum state 2' : real_sum_2_list,
                        'pred sum state 2' : pred_sum_2_list,
                        'emd' : emd_list})

metrics.to_csv(os.path.join(data_folder, 'XGBoost_metrics.csv'), index=False)
