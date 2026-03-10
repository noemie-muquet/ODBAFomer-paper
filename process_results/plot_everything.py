import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os 
from skimage.transform import resize
import matplotlib as mpl
import seaborn as sns
from utils.utils import calculate_longest_distance, compute_error, compute_stat, add_ellipse

mpl.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = "Calibri"

#%%

# =============================================================================
# Plot results direct ODBA density prediction
# =============================================================================

exp_name = 'bathy'

path = os.getcwd()

# test examples directory
exp_dir = os.path.join(path, 'experiments', exp_name, 'test_examples')

# infered ODBA 
results_folder = os.path.join(exp_dir, 'results')

# target ODBA
targets_folder = os.path.join(exp_dir, 'targets')

trips_dir =  os.path.join(path, 'data', 'eval_data_timeseries_hmm', 'eval_trips_with_hmm')

outlier = 'BRA_FDN_ME_2019-04-16_SDAC_03_NA_M_GPS_AXY_RT01_UTC.csv_trip_2.csv'

N = 100
res = 0.05

x_1_list = []
x_2_list = []
y_1_list = []
y_2_list = []

file_name_list = []

for file_name in os.listdir(trips_dir) : 
    data_path = os.path.join(trips_dir, file_name)
    df_data = pd.read_csv(data_path)
    R = calculate_longest_distance(df_data)
    
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
    
    x_1_list.append(x_1)
    x_2_list.append(x_2)
    y_1_list.append(y_1)
    y_2_list.append(y_2)
    file_id = file_name.replace(".csv", "")
    file_name_list.append(file_id)
   
X_1 = min(x_1_list)
X_2 = max(x_2_list)

Y_1 = min(y_1_list)
Y_2 = max(y_2_list)

x = np.linspace(X_1, X_2 + res, int(np.floor((X_2-X_1)/res)))
y = np.linspace(Y_1, Y_2 + res, int(np.floor((Y_2-Y_1)/res)))
X, Y = np.meshgrid(x, y)

simu_map = np.zeros_like(X, dtype=float)
real_map = np.zeros_like(X, dtype=float)


for file_name in os.listdir(results_folder) : 
    if outlier in file_name :
        continue
    if file_name.endswith('.csv'):  
        file_dir = os.path.join(results_folder, file_name)
        df_pred = pd.read_csv(file_dir)
        pred = df_pred.to_numpy()
        pred = np.delete(pred,0,1)
        pred = np.flipud(pred)
        
        file_id = file_name.replace(".csv", "") 
        
        for i in range(len(file_name_list)) :
            if file_name_list[i] in file_id : 
                
                x_1 = x_1_list[i]
                x_2 = x_2_list[i]
                y_1 = y_1_list[i]
                y_2 = y_2_list[i]
                                    
                new_Nx = np.round((y_2-y_1)/res)
                new_Ny = np.round((y_2-y_1)/res)
                
                new_pred = resize(pred, (new_Nx, new_Ny))
                
                x_1_i = np.where((x >= x_1 - res) & (x <= x_1 + res))
                y_1_i = np.where((y >= y_1 - res) & (y <= y_1 + res))
                
                x_1_i = x_1_i[0][0]
                x_2_i = x_1_i + len(new_pred)
                y_1_i = len(y) - y_1_i[0][0]
                y_2_i = y_1_i - len(new_pred)
                            
                simu_map[y_2_i : y_1_i, x_1_i : x_2_i] += new_pred*1000
            



for file_name in os.listdir(targets_folder) : 
    if outlier in file_name :
        continue
    file_dir = os.path.join(targets_folder, file_name)
    df_real = pd.read_csv(file_dir)
    real = df_real.to_numpy()
    real = np.delete(real,0,1)
    real = np.flipud(real)    
    
    file_id = file_name.replace(".csv", "") 
    
    for i in range(len(file_name_list)) :
        if file_name_list[i] in file_id : 
            x_1 = x_1_list[i]
            x_2 = x_2_list[i]
            y_1 = y_1_list[i]
            y_2 = y_2_list[i]
                      
            new_Nx = np.round((y_2-y_1)/res)
            new_Ny = np.round((y_2-y_1)/res)
            
            new_real = resize(real, (new_Nx, new_Ny))
            
            x_1_i = np.where((x >= x_1 - res) & (x <= x_1 + res))
            y_1_i = np.where((y >= y_1 - res) & (y <= y_1 + res))
            
            x_1_i = x_1_i[0][0]
            x_2_i = x_1_i + len(new_real)
            y_1_i = len(y) - y_1_i[0][0]
            y_2_i = y_1_i - len(new_real)
                        
            real_map[y_2_i: y_1_i, x_1_i : x_2_i] += new_real * 1000
            
            
simu_map = simu_map[4:36, 2:27]
real_map = real_map[4:36, 2:27]  
Y_1 = Y_1 + 5*res
Y_2 = Y_1 + 33*res
X_1 = X_1 + 3*res
X_2 = X_1 + 26*res 
extent = [X_1,X_2,Y_1,Y_2]

cmap = plt.cm.managua
colors = [cmap(i / 23) for i in range(23)]
k = 0

fig, ax = plt.subplots(figsize=(6, 6))  
for root, dirs, files in os.walk(trips_dir):
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(root, file))
            if len(df) > 8640: 
                continue
            start_lat, start_lon = df['lat'].iloc[0], df['lon'].iloc[0]
            ax.plot(df['lon'], df['lat'], color = colors[k])
            k += 1
ax.scatter(start_lon, start_lat, facecolor = 'red', edgecolor = 'darkred', linewidth = .5, s = 40, zorder = 10)
ax.set_xlim([X_1,X_2])
ax.set_ylim([Y_1,Y_2])
plt.axis('equal')
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.subplots_adjust(left=0, right=0.60, top=0.76, bottom=0)
plt.show()

plt.figure(figsize=(6,6))
im1 = plt.imshow(real_map, cmap='RdYlBu_r', vmin=0, vmax=3000, extent = extent)
plt.scatter(start_lon, start_lat, facecolor = 'red', edgecolor = 'darkred', linewidth = .5, s = 40, zorder = 10)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
cbar = plt.colorbar(im1)
cbar.set_label('Total ODBA (g)', fontsize=14)
plt.show()

plt.figure(figsize=(6,6))
im2 = plt.imshow(simu_map, cmap = 'RdYlBu_r', vmin = 0, vmax = 3000, extent = extent)
plt.scatter(start_lon, start_lat, facecolor = 'red', edgecolor = 'darkred', linewidth = .5, s = 40, zorder = 10)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
cbar = plt.colorbar(im2)
cbar.set_label('Total ODBA (g)', fontsize=14)
plt.show()

# print metrics
mae, rmse, bias, ssim_index = compute_error(real_map, simu_map)
print(f'mae: {mae}')
print(f'rmse: {rmse}')
print(f'bias: {bias}')
print(f'ssim: {ssim_index}')

#%%

# =============================================================================
# Plot results against XGBoost
# =============================================================================

exp_name = 'bathy'

path = os.getcwd()

# test examples directory
exp_dir = os.path.join(path, 'experiments', exp_name, 'test_examples')

# infered ODBA values
results_folder = os.path.join(exp_dir, 'values')

# XGBoost data
benchmark_folder = os.path.join(path, 'data', 'XGBoost_data')
benchmark_values_folder = os.path.join(benchmark_folder, 'XGBoost_outputs')

# get metrics from ODBAFormer
ai_metrics = pd.read_csv(os.path.join(exp_dir, 'res.csv'))

ai_metrics = ai_metrics.drop(index=[24])
ai_metrics = ai_metrics.drop(index=[3])

global_bias_ai = list((ai_metrics['pred sum'] - ai_metrics['real sum'])/ai_metrics['real sum'] * 100)
bias_state_1_ai = list((ai_metrics['pred sum state 1'] - ai_metrics['real sum state 1'])/ai_metrics['real sum state 1'] * 100)
bias_state_2_ai = list((ai_metrics['pred sum state 2'] - ai_metrics['real sum state 2'])/ai_metrics['real sum state 2'] * 100)
emd_ai = list(ai_metrics['emd'])

# get metrics from XGBoost
ml_metrics = pd.read_csv(os.path.join(benchmark_folder, 'XGBoost_metrics.csv'))

global_bias_ml = list((ml_metrics['pred sum'] - ml_metrics['real sum'])/ml_metrics['real sum'] * 100)
bias_state_1_ml = list((ml_metrics['pred sum state 1'] - ml_metrics['real sum state 1'])/ml_metrics['real sum state 1'] * 100)
bias_state_2_ml = list((ml_metrics['pred sum state 2'] - ml_metrics['real sum state 2'])/ml_metrics['real sum state 2'] * 100)
emd_ml = list(ml_metrics['emd'])

print(compute_stat(global_bias_ai, global_bias_ml))
print(compute_stat(bias_state_1_ai, bias_state_1_ml))
print(compute_stat(bias_state_2_ai, bias_state_2_ml))
print(compute_stat(emd_ai, emd_ml))

df_all = pd.concat([
    pd.DataFrame({'Error (%)': global_bias_ai, 'Model': 'ODBAFormer', 'Metric': r'$\Delta$'}),
    pd.DataFrame({'Error (%)': global_bias_ml, 'Model': 'XGBoost', 'Metric': r'$\Delta$'}),

    pd.DataFrame({'Error (%)': bias_state_1_ai, 'Model': 'ODBAFormer', 'Metric': r'$\Delta_{traveling}$'}),
    pd.DataFrame({'Error (%)': bias_state_1_ml, 'Model': 'XGBoost', 'Metric': r'$\Delta_{traveling}$'}),

    pd.DataFrame({'Error (%)': bias_state_2_ai, 'Model': 'ODBAFormer', 'Metric': r'$\Delta_{foraging}$'}),
    pd.DataFrame({'Error (%)': bias_state_2_ml, 'Model': 'XGBoost', 'Metric': r'$\Delta_{foraging}$'}),
])

custom_palette = { 'ODBAFormer': 'lightseagreen', 'XGBoost': 'palevioletred' }

plt.figure(figsize=(6,6))

sns.stripplot(
    data=df_all,
    x='Metric',
    y='Error (%)',
    hue='Model',
    dodge=True,
    palette=custom_palette,
    alpha=0.7,
    size=6,
    zorder=1
)

ax = sns.boxplot(
    data=df_all,
    x='Metric',
    y='Error (%)',
    hue='Model',
    dodge=True,
    palette=custom_palette,
    width=0.6,
    linewidth=1.5,
    fliersize=0,  
    zorder=2,

)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[:2],
    labels[:2],
    fontsize=14,
    loc='lower center',
    bbox_to_anchor=(0.41, -0.17),  
    borderaxespad=0,
    frameon=False, 
    ncols = 2
)

for patch in ax.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.5))   
    patch.set_edgecolor('darkslategrey')
    patch.set_linewidth(1)
    
# Style adjustments
for patch in ax.artists:
    patch.set_edgecolor('darkslategrey')

for line in ax.lines:
    line.set_color('darkslategrey')

plt.grid(axis='y', linestyle='--', alpha=0.8, zorder=0, linewidth=1.5)

plt.xlabel('', fontsize=14)
plt.ylabel('Error (%)', fontsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.xticks(fontsize=14)
plt.yticks(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97], pad=2.0)
plt.show()


print(f'EMD & {np.mean(emd_ai):.3f} ± {np.std(emd_ai):.3f} & {np.mean(emd_ml):.3f} ± {np.std(emd_ml):.3f}\\')
print(f'$\Delta$ (\%) & {np.mean(global_bias_ai):.1f} ± {np.std(global_bias_ai):.1f} & {np.mean(global_bias_ml):.1f} ± {np.std(global_bias_ml):.1f}\\')
print(f'$\Delta_t$ (\%) & {np.mean(bias_state_1_ai):.1f} ± {np.std(bias_state_1_ai):.1f} & {np.mean(bias_state_1_ml):.1f} ± {np.std(bias_state_1_ml):.1f}\\')
print(f'$\Delta_f$ (\%)  & {np.mean(bias_state_2_ai):.1f} ± {np.std(bias_state_2_ai):.1f} & {np.mean(bias_state_2_ml):.1f} ± {np.std(bias_state_2_ml):.1f}\\')


values_files = [
    os.path.join(results_folder, f) 
    for f in os.listdir(results_folder)]

df_list = [pd.read_csv(f) for f in values_files]
ai_values_df = pd.concat(df_list, ignore_index=True)
ai_values = ai_values_df["simu"].to_numpy() * 1000

values_files = [
    os.path.join(benchmark_values_folder, f) 
    for f in os.listdir(benchmark_values_folder)]

df_list = [pd.read_csv(f) for f in values_files]
ml_values_df = pd.concat(df_list, ignore_index=True)
real_values = ml_values_df["odba_f"].to_numpy()
ml_values = ml_values_df["y_pred"].to_numpy()


plt.figure(figsize=(6,5))
sns.kdeplot(ai_values, label = 'ODBAFormer', color = 'lightseagreen', zorder = 4,  bw_adjust=2, fill = True, lw  = 1.5)
sns.kdeplot(ml_values, label = 'XGBoost', color = 'palevioletred', zorder = 2, bw_adjust=1, fill = True, lw  = 1.5)
sns.kdeplot(real_values, label = 'Real values', color = 'slategrey', zorder = 3, bw_adjust=1, fill = True, lw  = 1.5)
plt.legend()
plt.xlim([0, 450])

plt.grid(axis='y', linestyle='--', alpha=0.8, zorder=0, linewidth=1.5)

legend = plt.legend(
    fontsize=14,
    loc='lower center',
    bbox_to_anchor=(0.45, -0.3),   
    borderaxespad=0,
    frameon=False, 
    ncols = 3
)


plt.xlabel('ODBA (g)', fontsize = 14)
plt.ylabel('Density', fontsize = 14, labelpad=10)
ax = plt.gca()  
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.xticks(np.arange(0, 450, 100), fontsize=12) 
plt.yticks(np.arange(0, 0.01, 0.002), fontsize=12) 
plt.tight_layout(rect=[0, 0, 1, 0.97], pad=2.0)
plt.show()


x = np.linspace(100, 920, 100)
x1 = np.linspace(40, 570, 100)
x2 = np.linspace(0, 450, 100)

fig, axes = plt.subplots(1,3,figsize=(15,5))

ax1 = axes[0]
sc1 = ax1.scatter(ai_metrics['pred sum'], ai_metrics['real sum'], s=20, label='ODBAFormer', color='lightseagreen')
sc2 = ax1.scatter(ml_metrics['pred sum'], ml_metrics['real sum'], s=20, label='XGBoost', color='palevioletred')
ax1.plot(x, x, color='grey', linewidth=2, alpha=0.6)
add_ellipse(ax1, ai_metrics['pred sum'], ai_metrics['real sum'], edgecolor='lightseagreen', facecolor='lightseagreen', alpha=0.2)
add_ellipse(ax1, ml_metrics['pred sum'], ml_metrics['real sum'], edgecolor='palevioletred', facecolor='palevioletred', alpha=0.2)

ax1.grid(axis='y', linestyle='--', alpha=0.8, zorder=0, linewidth=1.5)
ax1.set_xlabel('Predicted ODBA sum (g)', fontsize=14)
ax1.set_ylabel('Real ODBA sum (g)', fontsize=14)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_position(('outward', 5))
ax1.spines['bottom'].set_position(('outward', 5))
ax1.tick_params(axis='both', labelsize=12)

ax2 = axes[1]
ax2.scatter(ai_metrics['pred sum state 1'], ai_metrics['real sum state 1'], s=20, color='lightseagreen')
ax2.scatter(ml_metrics['pred sum state 1'], ml_metrics['real sum state 1'], s=20, color='palevioletred')
ax2.plot(x1, x1, color='grey', linewidth=2, alpha=0.6)
add_ellipse(ax2, ai_metrics['pred sum state 1'], ai_metrics['real sum state 1'], edgecolor='lightseagreen', facecolor='lightseagreen', alpha=0.2)
add_ellipse(ax2,ml_metrics['pred sum state 1'], ml_metrics['real sum state 1'], edgecolor='palevioletred', facecolor='palevioletred', alpha=0.2)

ax2.grid(axis='y', linestyle='--', alpha=0.8, zorder=0, linewidth=1.5)
ax2.set_xlabel('Predicted ODBA sum (g)', fontsize=14)
ax2.set_ylabel('Real ODBA sum (g)', fontsize=14)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_position(('outward', 5))
ax2.spines['bottom'].set_position(('outward', 5))
ax2.tick_params(axis='both', labelsize=12)

ax3 = axes[2]
ax3.scatter(ai_metrics['pred sum state 2'], ai_metrics['real sum state 2'], s=20, color='lightseagreen')
ax3.scatter(ml_metrics['pred sum state 2'], ml_metrics['real sum state 2'], s=20, color='palevioletred')
ax3.plot(x2, x2, color='grey', linewidth=2, alpha=0.6)
add_ellipse(ax3, ai_metrics['pred sum state 2'], ai_metrics['real sum state 2'], edgecolor='lightseagreen', facecolor='lightseagreen', alpha=0.2)
add_ellipse(ax3, ml_metrics['pred sum state 2'], ml_metrics['real sum state 2'], edgecolor='palevioletred', facecolor='palevioletred', alpha=0.2)

ax3.grid(axis='y', linestyle='--', alpha=0.8, zorder=0, linewidth=1.5)
ax3.set_xlabel('Predicted ODBA sum (g)', fontsize=14)
ax3.set_ylabel('Real ODBA sum (g)', fontsize=14)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_position(('outward', 5))
ax3.spines['bottom'].set_position(('outward', 5))
ax3.tick_params(axis='both', labelsize=12)

plt.tight_layout(rect=[0,0.1,1,0.97], pad=3.0)

fig.legend(handles=[sc1, sc2], labels=['ODBAFormer', 'XGBoost'],
           loc='lower center', bbox_to_anchor=(0.52, 0.08), ncol=2, fontsize=14, frameon = False)

plt.show()

#%%

# =============================================================================
# Plot results for different covariates
# =============================================================================
#%%

path = os.getcwd()

# only GPS 
exp_0_name = 'gps_only'
path_0 = os.path.join(path, 'experiments', exp_0_name, 'test_examples', 'res.csv')
df_pred_0 = pd.read_csv(path_0)
df_pred_0.drop(index=[3], inplace=True)

# GPS + bathymetry 
exp_1_name = 'bathy'
path_1 = os.path.join(path, 'experiments', exp_1_name, 'test_examples', 'res.csv')
df_pred_1 = pd.read_csv(path_1)
df_pred_1.drop(index=[3], inplace=True)
df_pred_1.drop(index=[24], inplace=True)

# GPS + wind 
exp_2_name = 'wind' 
path_2 = os.path.join(path, 'experiments', exp_2_name, 'test_examples', 'res.csv')
df_pred_2 = pd.read_csv(path_2)
df_pred_2.drop(index=[3], inplace=True)

# GPS + wind + bathymetry 
exp_3_name = 'bathy_wind' 
path_3 = os.path.join(path, 'experiments', exp_3_name, 'test_examples', 'res.csv')
df_pred_3 = pd.read_csv(path_3)
df_pred_3.drop(index=[3], inplace=True)

# metrics setup 
N_simus = 4

emd = []
emd_std = []
bias_g = []
bias_g_std = []
bias_1 = []
bias_1_std = []
bias_2 = []
bias_2_std = []

for i in range(N_simus):  
    
    df_name = f"df_pred_{i}"
    df = globals()[df_name]  

    df['global bias'] = (df['pred sum'] - df['real sum'])/df['real sum']*100
    df['bias state 1'] = (df['pred sum state 1'] - df['real sum state 1'])/df['real sum state 1']*100
    df['bias state 2'] = (df['pred sum state 2'] - df['real sum state 2'])/df['real sum state 2']*100
    
    emd.append(df['emd'].mean())
    emd_std.append(df['emd'].std())
    bias_g.append(df['global bias'].mean())
    bias_g_std.append(df['global bias'].std())
    bias_1.append(df['bias state 1'].mean())
    bias_1_std.append(df['bias state 1'].std())
    bias_2.append(df['bias state 2'].mean())
    bias_2_std.append(df['bias state 2'].std())
    
metrics = pd.DataFrame({'emd' : emd,
                        'emd std' : emd_std,
                        'global bias' : bias_g,
                        'global bias std' : bias_g_std,
                        'bias state 1' : bias_1,
                        'bias state 1 std' : bias_1_std,
                        'bias state 2' : bias_2, 
                        'bias state 2 std' : bias_2_std})


plt.figure(figsize=(8,6))

palette = ["yellowgreen", "lightseagreen", "mediumpurple", "coral"]

df_all = pd.concat([
    df_pred_0[["global bias","bias state 1","bias state 2"]].assign(case="GPS"),
    df_pred_1[["global bias","bias state 1","bias state 2"]].assign(case="GPS + bathymetry"),
    df_pred_2[["global bias","bias state 1","bias state 2"]].assign(case="GPS + wind"),
    df_pred_3[["global bias","bias state 1","bias state 2"]].assign(case="GPS + wind\n+ bathymetry")
])

df_melted = df_all.melt(id_vars='case', 
                        value_vars=['global bias','bias state 1','bias state 2'], 
                        var_name='metric', 
                        value_name='value')

metric_labels = {
    'global bias': '$\Delta$',
    'bias state 1': '$\Delta_{traveling}$',
    'bias state 2': '$\Delta_{foraging}$'
}
df_melted['metric_label'] = df_melted['metric'].map(metric_labels)


sns.stripplot(
    data=df_melted,
    x='metric_label',
    y='value',
    hue='case',
    palette=palette,
    dodge=True,        
    alpha=0.7,
    jitter=0.2, 
    size=5,       
    zorder=1
)

ax = sns.boxplot(
    data=df_melted,
    x='metric_label',
    y='value',
    hue='case',
    palette=palette,
    width=0.75,
    showcaps=True,
    showfliers=False,
    color = palette, 
    linewidth = 1.5)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[:4],
    labels[:4],
    fontsize=12,
    loc='lower center',
    bbox_to_anchor=(0.49, -0.2),  
    borderaxespad=0,
    frameon=False, 
    ncols = 4
)

for patch in ax.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.5))   
    patch.set_edgecolor('darkslategrey')
    patch.set_linewidth(1)
    
for patch in ax.artists:
    patch.set_edgecolor('darkslategrey')

for line in ax.lines:
    line.set_color('darkslategrey')

plt.grid(axis='y', linestyle='--', alpha=0.8, zorder=0, linewidth=1.5)
plt.xlabel('')
plt.ylabel('Bias (%)', fontsize=14, labelpad=-1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.xticks(fontsize=12) 
plt.yticks(np.arange(-25,125,25), fontsize=12) 

plt.tight_layout(rect=[0, 0, 1, 0.97], pad=2.0)
plt.show()

#%%

# =============================================================================
# Plot results for one and two-species models
# =============================================================================

# single-species model evaluated on masked booby 
exp_0_name = 'bathy'
path_0 = os.path.join(path, 'experiments', exp_0_name, 'test_examples', 'res.csv')
df_pred_0 = pd.read_csv(path_0)
df_pred_0.drop(index=[3], inplace=True)
df_pred_0.drop(index=[24], inplace=True)

# two-species model evaluated on masked booby 
exp_1_name = '2_species_masked'
path_1 = os.path.join(path, 'experiments', exp_1_name, 'test_examples', 'res.csv')
df_pred_1 = pd.read_csv(path_1)
df_pred_1.drop(index=[3], inplace=True)

# two-species model evaluated on brown booby 
exp_2_name = '2_species_browns'
path_2 = os.path.join(path, 'experiments', exp_2_name, 'test_examples', 'res.csv')
df_pred_2 = pd.read_csv(path_2)

# metrics setup 
N_simus = 3

emd = []
emd_std = []
bias_g = []
bias_g_std = []
bias_1 = []
bias_1_std = []
bias_2 = []
bias_2_std = []


for i in range(N_simus):  
    
    df_name = f"df_pred_{i}"
    df = globals()[df_name]  

    df['global bias'] = (df['pred sum'] - df['real sum'])/df['real sum']*100
    df['bias state 1'] = (df['pred sum state 1'] - df['real sum state 1'])/df['real sum state 1']*100
    df['bias state 2'] = (df['pred sum state 2'] - df['real sum state 2'])/df['real sum state 2']*100
    
    emd.append(df['emd'].mean())
    emd_std.append(df['emd'].std())
    bias_g.append(df['global bias'].mean())
    bias_g_std.append(df['global bias'].std())
    bias_1.append(df['bias state 1'].mean())
    bias_1_std.append(df['bias state 1'].std())
    bias_2.append(df['bias state 2'].mean())
    bias_2_std.append(df['bias state 2'].std())

    
metrics = pd.DataFrame({'emd' : emd,
                        'emd std' : emd_std,
                        'global bias' : bias_g,
                        'global bias std' : bias_g_std,
                        'bias state 1' : bias_1,
                        'bias state 1 std' : bias_1_std,
                        'bias state 2' : bias_2, 
                        'bias state 2 std' : bias_2_std})


plt.figure(figsize=(7,6))

palette = ["lightseagreen", "olive", "peru"]

df_all = pd.concat([
    df_pred_0[["global bias","bias state 1","bias state 2"]].assign(case="SS model, MB"),
    df_pred_1[["global bias","bias state 1","bias state 2"]].assign(case="TS model, MB"),
    df_pred_2[["global bias","bias state 1","bias state 2"]].assign(case="TS model, BB"),
])

df_melted = df_all.melt(id_vars='case', 
                        value_vars=['global bias','bias state 1','bias state 2'], 
                        var_name='metric', 
                        value_name='value')

metric_labels = {
    'global bias': '$\Delta$',
    'bias state 1': '$\Delta_{traveling}$',
    'bias state 2': '$\Delta_{foraging}$'
}
df_melted['metric_label'] = df_melted['metric'].map(metric_labels)


sns.stripplot(
    data=df_melted,
    x='metric_label',
    y='value',
    hue='case',
    palette=palette,
    dodge=True,       
    alpha=0.7,
    jitter=0.2, 
    size=5,      
    zorder=1
)

ax = sns.boxplot(
    data=df_melted,
    x='metric_label',
    y='value',
    hue='case',
    palette=palette,
    width=0.75,
    showcaps=True,
    showfliers=False,
    color = palette, 
    linewidth = 1.5)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[:3],
    labels[:3],
    fontsize=12,
    loc='lower center',
    bbox_to_anchor=(0.45, -0.15),   
    borderaxespad=0,
    frameon=False, 
    ncols = 3
)

for patch in ax.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.5))   
    patch.set_edgecolor('darkslategrey')
    patch.set_linewidth(1)
    
for patch in ax.artists:
    patch.set_edgecolor('darkslategrey')

for line in ax.lines:
    line.set_color('darkslategrey')

plt.grid(axis='y', linestyle='--', alpha=0.8, zorder=0, linewidth=1.5)
plt.xlabel('')
plt.ylabel('Bias (%)', fontsize=14, labelpad=-1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.xticks(fontsize=12) 
plt.yticks(fontsize=12) 

plt.tight_layout(rect=[0, 0, 1, 0.97], pad=2.0)
plt.show()
