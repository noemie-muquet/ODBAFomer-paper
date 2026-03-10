import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 
from skimage.transform import resize
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from matplotlib.patches import Ellipse


# generates colormaps were 0-values are white and non-0 values are of wanted colormap cmap
def generate_custom_colormap(cmap):
    colormap = plt.colormaps.get_cmap(cmap)  
    newcolors = colormap(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])  # set the first color (corresponding to zero) to white
    newcmp = mcolors.ListedColormap(newcolors)
    return newcmp

custom_cmap = generate_custom_colormap('Spectral_r')
custom_cmap_2 = generate_custom_colormap('copper_r')


# computes how far a trajectory goes from nest in °
def calculate_longest_distance(df):
    coords = df[['lon', 'lat']].to_numpy()
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    max_distance = np.max(dist_matrix)
    return max_distance

# compute a distance in km given a distance in ° and a latitude : 
def degrees_to_km(degrees, latitude):
    R = 6371.0 # radius of the Earth in kilometers
    lat_rad = np.radians(latitude)
    km_per_degree_lat = R * np.pi / 180
    km_per_degree_lon = R * np.pi / 180 * np.cos(lat_rad)
    distance_km = degrees * km_per_degree_lat if degrees >= 0 else degrees * km_per_degree_lon
    return distance_km

# direct distance calculator in m
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # radius of the Earth in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# function to upscale entire field and crop
def upscale_and_crop(field, y1, y2, x1, x2, crop_size, scale_y, scale_x):
    upscaled_field = resize(
        field,
        (int(field.shape[0] * scale_y), int(field.shape[1] * scale_x)),
        order=1,
        preserve_range=True,
        anti_aliasing=True)
    new_y1 = int(y1 * scale_y)
    new_y2 = new_y1 + crop_size[0]
    new_x1 = int(x1 * scale_x)
    new_x2 = new_x1 + crop_size[1]
    return upscaled_field[new_y1:new_y2, new_x1:new_x2].astype(np.float32)

# compute step length in m
def get_step_length(df) : 
    R = 6371000  
    lat_rad = np.radians(df['lat'])
    lon_rad = np.radians(df['lon'])
    dlat = lat_rad.diff()
    dlon = lon_rad.diff()
    a = np.sin(dlat / 2)**2 + np.cos(lat_rad) * np.cos(lat_rad.shift()) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    step_length = R * c
    return step_length

# compute truning angle in °
def get_turning_angle(df) :
    lat1 = np.radians(df['lat'].shift())
    lat2 = np.radians(df['lat'])
    lon1 = np.radians(df['lon'].shift())
    lon2 = np.radians(df['lon'])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    turning_angle = np.arctan2(y, x) 
    return turning_angle

# calculate p-value and CI, data1 is ai metrics, data2 is XGBoost results
def compute_stat(data1, data2) : 
    data1 = np.array(data1)
    data2 = np.array(data2)
    mean_diff = np.mean(data1) - np.mean(data2)
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    se_diff = np.sqrt(np.var(data1, ddof=1)/len(data1) + np.var(data2, ddof=1)/len(data2))
    ci_range = stats.t.ppf(0.975, df=min(len(data1), len(data2))-1) * se_diff
    ci_lower = mean_diff - ci_range
    ci_upper = mean_diff + ci_range
    
    return(p_value, ci_lower, ci_upper)

# calculate error metrics on energy landscape proxy
def compute_error(real_map, simu_map):
    real_map = np.array(real_map)
    simu_map = np.array(simu_map)
    
    ssim_index = ssim(real_map, simu_map, data_range=real_map.max() - real_map.min())
    
    mask = real_map > 0

    real_filtered = real_map[mask]
    simu_filtered = simu_map[mask]
    mae = np.mean(np.abs(real_filtered - simu_filtered))
    rmse = np.sqrt(np.mean((real_filtered - simu_filtered)**2))
    bias = np.mean(simu_filtered - real_filtered)
    
    return mae, rmse, bias, ssim_index

def add_ellipse(ax, x, y, n_std=2.0, edgecolor='red', facecolor='red', alpha=0.2, lw=2):
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    angle = np.rad2deg(np.arctan2(*v[:,0][::-1]))
    
    ellipse = Ellipse(xy=(mean_x, mean_y),
                      width=lambda_[0]*2*n_std,
                      height=lambda_[1]*2*n_std,
                      angle=angle,
                      edgecolor=edgecolor,
                      facecolor=facecolor,
                      alpha=alpha,
                      lw=lw)
    ax.add_patch(ellipse)