import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os
import xarray as xr
from skimage.transform import resize
import pygrib
from utils.utils import generate_custom_colormap, calculate_longest_distance, degrees_to_km, haversine, upscale_and_crop

# =============================================================================
# inputs :  csv files, one trip per file, regularized
# outputs :  ODBAFormer inputs (trajectory, ODBA (target) and environmental covariates)
# given parameters : 
    # - N : final matrix size 
    # - cutoff_len : minimal max distance of a trip to be processed
    # - cutoff_step_size : maximal step size accepted (to filter out trajectories with anomalies)
# =============================================================================

# initializes input and outputs directories 
path = os.getcwd()

data_folder = os.path.join(path, 'data','train_data')
input_folder = os.path.join(data_folder,'raw_csv')
 
results_folder = os.path.join(data_folder, 'ODBAFormer_inputs')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
visu_folder = os.path.join(results_folder, 'plots')
if not os.path.exists(visu_folder):
    os.makedirs(visu_folder)
        
covariates_folder = os.path.join(data_folder, 'covariates')

# bathymetry maps paths 
abr_path = os.path.join(covariates_folder, 'bathy_gebco_abr_1080_1080.nc')
fdn_path = os.path.join(covariates_folder, 'bathy_gebco_fdn_1080_1080.nc')
spsp_path = os.path.join(covariates_folder, 'bathy_gebco_spsp_1080_1080.nc')

# wind .grib files location 
wind_folder =  os.path.join(covariates_folder, 'wind')

# fixes matrix/map size 
N = 100
# fixes minimal distance to colony for a trip to be processed 
cutoff_len = 10
# fixes a maximum value of step size to filter out trajectories anomalies
cutoff_step_size = 20000

# initiate custom colormaps
custom_cmap = generate_custom_colormap('Spectral_r')
custom_cmap_2 = generate_custom_colormap('copper_r')

# get wind fields file list and extract keywords from file names to match wind to trajectory
# wind file name format: LOC_YEAR_MONTH_STARTDATE.grib

file_names = [f for f in os.listdir(wind_folder) if os.path.isfile(os.path.join(wind_folder, f))]

file_lookup = {}
for file_name in file_names:
    parts = file_name.split("_")
    if len(parts) == 4:  
        location = parts[0]
        year = parts[1]
        month = parts[2]
        day = int(parts[3].split(".")[0]) 

        file_lookup[(year, month)] = (file_name, day)

# get trajectory files
data_files = [f for f in os.listdir(input_folder) 
              if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith('.csv')]

# =============================================================================
# loop on all files 
# =============================================================================

for data_file in os.listdir(input_folder):
    if data_file.endswith('.csv'):  
        data_path = os.path.join(input_folder, data_file)
        df_data = pd.read_csv(data_path)
        
### wind fields setup: align wind data to trajectory dates
        
        # get trajectory date and time 
        first_datetime = pd.to_datetime(df_data.loc[0, 'datetime']) 
        last_datetime = pd.to_datetime(df_data.iloc[-1]['datetime'])
        
        year = str(first_datetime.year)
        month = str(first_datetime.month).zfill(2)
        
        # specific exeptions for fieldworks spreading on two different months 
        if year == '2019' and month == '07' : 
            month = '06'
            
        if year == '2023' and month == '07' : 
            month = '06'
                
        start_time = int(first_datetime.hour)
        start_day = int(first_datetime.day)
        end_time = int(last_datetime.hour)
        end_day = int(last_datetime.day)
        
        # match trajectory to wind field using year and month (LOCATION NOT IMPLEMENTED BECAUSE NOT NEEDED FOR THIS SPECIFIC DATASET)
        matched_entry = file_lookup.get((year, month), None)
        if matched_entry:
            matched_file, day = matched_entry 
        else : 
            print(f'unmatched file: {data_file}')
        
        # align time of trajectory with wind data
        if end_day-start_day == 0 : 
            n_hrs = end_time - start_time
            start_hour = start_time + (start_day-int(day))*24   
        else : 
            n_hrs = (25 - start_time) + (end_day-start_day - 1)*24 + end_time
            start_hour = start_time + (start_day-int(day)) *24
            
        # for trips of less than 1 hrs    
        if n_hrs == 0 :
            n_hrs = 1
        
        # for trips > 1 day
        if start_day < day : 
            start_hour = start_time + (start_day-int(day)+30) *24
            
### now that wind is aligned to trajectory:
    
        if matched_entry:
            
            # exit loop is no accelerometry recorded 
            if sum(df_data['odba_f']) == 0 :
                print(f'no odba: {data_file}')
                continue
            
            # gets nest position 
            start_lat = df_data['lat'].iloc[0]
            start_lon = df_data['lon'].iloc[0]
            
            # max distance to nest to compute scaling factor
            R = calculate_longest_distance(df_data)
            
            # computes max distance to next during trip
            max_dist = round(degrees_to_km(R, start_lat))
            
            # exit loop if trip too short 
            if max_dist < cutoff_len : 
                print(f'too short: {data_file}')
                continue
            
            # computes time resolution
            df_data['datetime'] = pd.to_datetime(df_data['datetime'])
            start = df_data['datetime'].iloc[1]
            end = df_data['datetime'].iloc[-1]
            total_time = (end - start).total_seconds()
            total_time_hrs = round((end - start).total_seconds() / 3600)

            # computes step size and total distance of the trip
            step_size = []
            for i in range(len(df_data)-1) : 
                step_size.append(haversine(df_data['lon'].iloc[i], df_data['lat'].iloc[i], df_data['lon'].iloc[i+1], df_data['lat'].iloc[i+1]))
            trip_length = round(np.sum(step_size)/1000)
            
            # exit loop if abnormal step size value
            if max(step_size) > cutoff_step_size  :
                print(f'outlier point detected: {data_file}')
                continue
            
### prepares ODBAFormer input tensors 

            # opens the right bathymetry map given nest position 
            if np.floor(start_lat) == -18 and np.floor(start_lon) == -39 : 
                data = xr.open_dataset(abr_path)
                time_zone = 3
                
            elif np.floor(start_lat) == -4 and np.floor(start_lon) == -33 :
                data = xr.open_dataset(fdn_path)
                time_zone = 2

            elif np.floor(start_lat) == 0 and np.floor(start_lon) == -30 :
                data = xr.open_dataset(spsp_path)
                time_zone = 3
            
            # bathymetry values 
            bathymetry = data['elevation']
            bathy = bathymetry.values 
            max_bathy = np.min(bathy)
            latitudes = data['lat'].values
            longitudes = data['lon'].values
            
            # compute rescaling around trajectory extent 
            x_min = min(df_data['lon'])
            y_min = min(df_data['lat'])
            x_max = max(df_data['lon'])
            y_max = max(df_data['lat'])

            x_m = x_min + x_max
            x_m = x_m/2

            y_m = y_min + y_max
            y_m = y_m/2

            x_1 = x_m - R/1.9
            x_2 = x_m + R/1.9

            y_1 = y_m - R/1.9
            y_2 = y_m + R/1.9            
            
            # get indexes to crop bathymetry map accordingly
            deltax = longitudes[2] - longitudes[1]
            deltay = latitudes[2] - latitudes [1]
            
            x_1_i = np.where((longitudes >= x_1 - deltax) & (longitudes <= x_1 + deltax))
            x_2_i = np.where((longitudes >= x_2 - deltax) & (longitudes <= x_2 + deltax))
            y_1_i = np.where((latitudes >= y_1 - deltay) & (latitudes <= y_1 + deltay))
            y_2_i = np.where((latitudes >= y_2 - deltay) & (latitudes <= y_2 + deltay))
            
            if len(x_1_i[0]) == 0 or len(x_2_i[0]) == 0 or len(y_1_i[0]) == 0 or len(y_2_i[0]) == 0 :
                print(f'point out of bathy bounds: {data_file}')
                continue
            
            x_1_i = x_1_i[0][0]
            x_2_i = x_2_i[0][0]
            y_1_i = y_1_i[0][0]
            y_2_i = y_2_i[0][0]
            
            # crop bathy map and normalize it
            new_bathy = bathy[y_1_i:y_2_i, x_1_i:x_2_i]/max_bathy
            new_bathy = new_bathy.astype(np.float32) 
            
            # resized and interpolates cropped bathy map to fit ODBAFormer input size requirement 
            new_bathy = resize(new_bathy, (N, N), order=3, mode='reflect', anti_aliasing=True) 
            
            # creates a N*N matrix of coordinates on the croped area (gridded space)
            x = np.linspace(x_1, x_2 + (x_2-x_1)/N, N)
            y = np.linspace(y_1, y_2 + (y_2-y_1)/N, N)
            X, Y = np.meshgrid(x, y)
            
            # computes gps points density on the N*N matrix
            points_density = np.zeros_like(X, dtype=int)

            for index, row in df_data.iterrows():
                lat = row['lat']
                lon = row['lon']
                i = int((lat - y_1) // ((y_2-y_1)/N))
                j = int((lon - x_1) // ((x_2-x_1)/N))
                if 0 <= i < len(y) and 0 <= j < len(x):
                    points_density[i, j] += 1

            # computes odba spatial density on the N*N matrix
            odba_density = np.zeros_like(X, dtype=float)
             
            for index, row in df_data.iterrows():
                 lat = row['lat']
                 lon = row['lon']
                 i = int((lat - y_1) // ((y_2-y_1)/N))
                 j = int((lon - x_1) // ((x_2-x_1)/N))
                 if 0 <= i < len(y) and 0 <= j < len(x) :
                     odba_density[i, j] += row['odba_f']/1000

            # open wind data
            file_path = os.path.join(wind_folder, matched_file)
            grbs = pygrib.open(file_path)
            
            total_u_wind = None
            total_v_wind = None
            count = 0
            
            start_i = start_hour + 1 + time_zone
            end_i = start_hour + n_hrs + 1 + time_zone
            
            # sanity check
            if end_i > len(grbs)//2 :
                end_i = len(grbs)//2
                print("wind data not long enough")
            
            # get average wind over trip
            for i in range(start_i, end_i):
                u_wind = grbs.message(2*i - 1)
                v_wind = grbs.message(2*i)
                u_data, lats, lons = u_wind.data()
                v_data, _, _ = v_wind.data()
                if total_u_wind is None:
                    total_u_wind = u_data
                    total_v_wind = v_data
                else:
                    total_u_wind += u_data
                    total_v_wind += v_data    
                count += 1
                
            grbs.close()
            
            average_u_wind = total_u_wind / count
            average_v_wind = total_v_wind / count
            
            # # normalize by max value over domain 
            max_u_wind = max(np.max(average_u_wind), - np.min(average_u_wind))
            max_v_wind = max(np.max(average_v_wind), - np.min(average_v_wind))
            average_u_wind = average_u_wind / max_u_wind
            average_v_wind = average_v_wind / max_v_wind
                   
            M = len(average_u_wind)
            
            first_lon = lons[0,0]
            last_lon = lons[0,-1]
            longitudes  = np.linspace(last_lon, first_lon, M)
            first_lat = lats[0,0]
            last_lat = lats[-1,0]
            latitudes  = np.linspace(last_lat, first_lat, M)
            
            # rescale wind maps
            deltax = longitudes[1] - longitudes[2]
            deltay = latitudes[2] - latitudes [1]
            
            x_1_i = np.where((longitudes >= x_1 - 1.5*deltax) & (longitudes <= x_1 + 1.5*deltax))
            x_2_i = np.where((longitudes >= x_2 - 1.5*deltax) & (longitudes <= x_2 + 1.5*deltax))
            y_1_i = np.where((latitudes >= y_1 - 1.5*deltay) & (latitudes <= y_1 + 1.5*deltay))
            y_2_i = np.where((latitudes >= y_2 - 1.5*deltay) & (latitudes <= y_2 + 1.5*deltay))
            
            if len(x_1_i[0]) == 0 or len(x_2_i[0]) == 0 or len(y_1_i[0]) == 0 or len(y_2_i[0]) == 0 :
                print(f'point out of wind field bounds: {data_file}')
                continue
                
            x_1_i = x_1_i[0][0]
            x_2_i = x_2_i[0][0]
            y_1_i = y_1_i[0][0]
            y_2_i = y_2_i[0][0]
            
            # exit loop if rescaling is not possible
            if x_1_i == x_2_i or y_1_i == y_2_i :
                continue 
            
            crop_size = (N, N)

            # compute upscale factors
            scale_y = crop_size[0] / (y_2_i - y_1_i)
            scale_x = crop_size[1] / (x_1_i - x_2_i)
            
            # apply to all fields
            new_u_wind = upscale_and_crop(average_u_wind, y_1_i, y_2_i, x_2_i, x_1_i, crop_size, scale_y, scale_x)
            new_v_wind = upscale_and_crop(average_v_wind, y_1_i, y_2_i, x_2_i, x_1_i, crop_size, scale_y, scale_x)
         
 ### save .npy file and plots
 
            # order is: points, bathy, u-wind, v-wind, target
            result_tensor = np.stack([points_density, new_bathy, new_u_wind, new_v_wind, odba_density], axis=-1)
            file_id = data_file.replace(".csv", "")
            result_path = os.path.join(results_folder, f'{file_id}_input.npy')
            np.save(result_path, result_tensor)
            
            # plot bathymetry map, real trajectory and diving areas        
            fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (24,10))       
            
            im0 = ax[0,0].imshow(new_bathy, extent=[x_1, x_2, y_1, y_2], cmap='YlGnBu', interpolation='nearest', origin='lower', vmin = -1,  vmax = 1)
            ax[0,0].plot(df_data['lon'], df_data['lat'], 'red')
            fig.colorbar(im0, ax = ax[0,0])
            ax[0,0].set_xlabel('lon')
            ax[0,0].set_ylabel('lat')
            ax[0,0].set_title('Normalized depth and trajectory')
            
            # plot gps point density matrix 
            im2 = ax[0,1].imshow(points_density, origin='lower', cmap=custom_cmap)
            fig.colorbar(im2, ax = ax[0,1])
            ax[0,1].set_title('GPS points density')
            
            # plot odba density matrix     
            im3 = ax[0,2].imshow(odba_density, origin='lower', cmap=custom_cmap_2, vmin= 0)
            fig.colorbar(im3, ax = ax[0,2])
            ax[0,2].set_title('kODBA density')
            
            
            im4 = ax[1,0].imshow(new_u_wind, cmap='gist_rainbow', vmin = -0.7, vmax = 0.7)
            fig.colorbar(im4, ax = ax[1,0])
            ax[1,0].set_title('Average U wind speed over trip')
            
            im5 = ax[1,1].imshow(new_v_wind, cmap='gist_rainbow', vmin = -0.7, vmax = 0.7)
            fig.colorbar(im5, ax = ax[1,1])
            ax[1,1].set_title('Average V wind speed over trip')
            
            # add informations to the plot 
            ax[1,2].axis('off')
            ax[1,2].text(0.5, 0.5, f'Max distance to nest : {max_dist} km \n\nTrip length : {trip_length} km \n\nTotal time : {total_time_hrs} hrs', fontsize=16, ha='center', va='center')
                        
            fig.suptitle(data_file)
            plt.tight_layout()

            # save plot
            plot_file_path = os.path.join(visu_folder, f'{file_id}_input.png')
            plt.savefig(plot_file_path)
            plt.close('all')