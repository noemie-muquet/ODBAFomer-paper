import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os 
import seaborn as sns
from scipy.stats import wasserstein_distance
from utils.utils import calculate_longest_distance

### DISCLAIMER THIS IS VERY MUCH *NOT* OPTIMIZED

# =============================================================================
# inputs : 
        # - models output from testing
        # - true data for evaluation dataset with behavioral states computed by HMM 
# outputs : 
        # - infered ODBA timeseries as csv 
        # - performance metrics summarized in a csv file 
        # - plots of ODBA timeseries vs real raw data and vs real data pre and post processed
# =============================================================================

exp_name = 'test'

path = os.getcwd()

# test examples directory
exp_dir = os.path.join(path, 'experiments', exp_name, 'test_examples')

# infered ODBA 
results_folder = os.path.join(exp_dir, 'results')

# target ODBA
targets_folder = os.path.join(exp_dir, 'targets')

# GPS density maps (inputs)
inputs_folder = os.path.join(exp_dir, 'inputs')

# trajectory timeseries with HMM states 
hmm_folder = os.path.join(path, 'data', 'eval_data_timeseries_hmm', 'eval_trips_with_hmm')


# output ODBA timeseries 
values_folder = os.path.join(exp_dir, 'values')  
os.makedirs(values_folder, exist_ok=True)  

# compute infered ODBA timeseries (this could be optimized) and metrics for each 
res_df = pd.DataFrame(columns=["file name"])

for file_name in os.listdir(hmm_folder):
    if file_name.endswith('.csv'):
        file_id = file_name.replace(".csv", "")
        df_data_path = os.path.join(hmm_folder, file_name)
        df_data = pd.read_csv(df_data_path)
        for pred_file in os.listdir(results_folder):
            if file_id in pred_file:  
                if pred_file.endswith('.csv'):
                    df_pred = pd.read_csv(os.path.join(results_folder, pred_file))
                    
                    pred = df_pred.to_numpy()
                    pred = np.delete(pred,0,1)
                    pred = np.flipud(pred)
                    
                    for input_file in os.listdir(inputs_folder):
                        if file_id in input_file:  
                            if input_file.endswith('.csv'): 
                                df_input = pd.read_csv(os.path.join(inputs_folder, input_file))
                                
                                gps_in = df_input.to_numpy()
                                gps_in = np.delete(gps_in,0,1)
                                gps_in = np.flipud(gps_in)
                                            
                    for target_file in os.listdir(targets_folder):
                        if file_id in target_file:  
                            if target_file.endswith('.csv'): 
                                df_target = pd.read_csv(os.path.join(targets_folder, target_file))
                                
                                target = df_target.to_numpy()
                                target = np.delete(target,0,1)
                                target = np.flipud(target)
                                
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
                            
                            
                                N = len(pred)    
                                M = len(df_data)
                                ms = (x_2 - x_1)/N 
                                simu = []
                                tg = []                            
                                
                                for k in range(M) : 
                                    for i in range(N):
                                        for j in range(N) : 
                                            loclon = x_1 + i*ms 
                                            loclat = y_1 + j*ms
                                            if loclon <= df_data['lon'].iloc[k] <= loclon + ms and loclat <= df_data['lat'].iloc[k] <= loclat + ms :
                                                simu.append(pred[N-j-1,i]/gps_in[N-j-1,i])
                                                tg.append(target[N-j-1,i]/gps_in[N-j-1,i])
                                
                                
                                df_final = pd.DataFrame({"odba": df_data["odba_f"] / 1000,
                                                        "target": tg[0:len(df_data)],
                                                        "simu": simu[0:len(df_data)],
                                                        "state": df_data["state"]
                                                        })
                                
                                mse = np.mean((df_final['target'] - df_final['simu']) ** 2)
                                abs_err = np.mean((np.abs(df_final['target'] - df_final['simu']))/df_final['target'])
                                emd = wasserstein_distance(df_final['target']*1000, df_final['simu']*1000)
                                
                                real_sum = df_final['odba'].sum()
                                pred_sum =  df_final['simu'].sum()
                                
                                real_sum_state_1 = df_final[df_final['state'] == 1]['odba'].sum()
                                pred_sum_state_1 = df_final[df_final['state'] == 1]['simu'].sum()
                                
                                real_sum_state_2 = df_final[df_final['state'] == 2]['odba'].sum()
                                pred_sum_state_2 = df_final[df_final['state'] == 2]['simu'].sum()

                                res_df = pd.concat([res_df, pd.DataFrame({"file name": [file_name], 
                                                                          "mse": [mse], 
                                                                          "abs err" : [abs_err],
                                                                          "real sum": [real_sum], 
                                                                          "pred sum": [pred_sum], 
                                                                          "real sum state 1" : [real_sum_state_1], 
                                                                          "pred sum state 1" : [pred_sum_state_1], 
                                                                          "real sum state 2" : [real_sum_state_2], 
                                                                          "pred sum state 2" : [pred_sum_state_2], 
                                                                          "emd" : [emd]
                                                                          })], 
                                                   ignore_index=True)
                                
                                # save infered ODBA timeseries as csv 
                                values_file_name = f'{file_name}_values.csv'
                                values_file_path = os.path.join(values_folder, values_file_name)
                                df_final[['target', 'simu']].to_csv(values_file_path, index=False)

                                ws = 12
                                df_final['odba_smoothed'] = df_final['odba'].rolling(window=ws).mean()
                                df_final['simu_smoothed'] = df_final['simu'].rolling(window=ws).mean()
                                
                                # plots 
                                plt.figure(figsize=(12, 24)) 
                                plt.subplot(4,1,1)
                                plt.plot(df_final['target'], label = 'taget kODBA')
                                plt.plot(df_final['simu'], label = 'predicted kODBA', color = 'black')
                                plt.xlabel('step')
                                plt.title('Target vs predicted ODBA')
                                plt.legend()
                                
                                plt.subplot(4,1,2)
                                plt.plot(df_final['odba'], label = 'real kODBA')
                                plt.plot(df_final['simu'], label = 'predicted kODBA', linewidth = 2, color = 'black')
                                plt.xlabel('step')
                                plt.title('Real vs predicted ODBA')
                                plt.legend()
                                
                                plt.subplot(4,1,3)
                                plt.plot(df_final['odba_smoothed'], label = 'real kODBA')
                                plt.plot(df_final['simu_smoothed'], label = 'predicted kODBA', linewidth = 2, color = 'black')
                                plt.xlabel('step')
                                plt.title(f'Real vs predicted ODBA, moving average {ws*10//60} minutes')
                                plt.legend()
                                
                                plt.subplot(4,1,4)                               
                                sns.kdeplot(df_final['target'], common_norm=True, label = 'real kODBA')
                                sns.kdeplot(df_final['simu'], common_norm=True, label = 'predicted kODBA', color = 'black')
                                plt.title('kODBA value distribution')
                                plt.xlabel('kODBA values')
                                plt.ylabel('Density')
                                plt.legend()
                                plt.tight_layout()
    
                                plot_file_name = f'{file_id}_timeseries.png'
                                plot_file_path = os.path.join(results_folder, plot_file_name)
                                plt.savefig(plot_file_path)    
                                plt.close()
                                
# save metrics as csv file                                                           
output_path = os.path.join(exp_dir, 'res.csv')
res_df.to_csv(output_path, index=False)


