import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import glob
import seaborn as sns # For better aesthetics
import pandas as pd # For rolling standard deviation

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Configuration names (ensure these match your actual config keys used for experiment naming)
CONFIG_NAME_SAC_MLP = "default_sac_mlp"
CONFIG_NAME_PPO_MLP = "default_ppo_mlp"
CONFIG_NAME_SAC_RNN = "default_sac_rnn"
CONFIG_NAME_PPO_RNN = "default_ppo_rnn" # Assuming this config name was used for a PPO RNN experiment
CONFIG_NAME_DEFAULT_ALIAS = "default_mapping" # Alias for MLP SAC/PPO if used

# Algorithm identifiers used in folder names
ALGO_SAC = "sac"
ALGO_PPO = "ppo"

# TensorBoard scalar tags
REWARD_AVG_100_TAG = "Reward/Average_100"
REWARD_EPISODE_TAG = "Reward/Episode" 

# SAC specific tags for mapping
SAC_ENV_STEPS_TAG_EPISODE_BASED = "Progress/Total_Env_Steps"
SAC_TOTAL_UPDATES_TAG_EPISODE_BASED = "Progress/Total_Updates"

# Colors (using a seaborn palette for distinctness)
palette = sns.color_palette("deep", 4) # Get 4 distinct colors
PPO_MLP_COLOR = palette[0] # Blue
SAC_MLP_COLOR = palette[1] # Orange
PPO_RNN_COLOR = palette[2] # Green
SAC_RNN_COLOR = palette[3] # Red/Purple

# X-axis limits (Set to None to plot full range)
STEPS_PLOT_X_MAX_MILLIONS = None 
TIME_PLOT_X_MAX_HOURS = None    

# Rolling standard deviation window
STD_WINDOW_SIZE = 10 

# --- Helper Functions ---

def find_latest_experiment_folder(base_dir, config_name, algo_name):
    pattern = os.path.join(base_dir, f"{config_name}_{algo_name}_*")
    folders = glob.glob(pattern)
    if not folders: return None
    folders.sort(key=lambda f: int(os.path.basename(f).split('_')[-1]))
    return folders[-1]

def extract_scalar_data(event_file_path, scalar_tags_list):
    data = {tag: {'wall_times': np.array([]), 'steps': np.array([]), 'values': np.array([])} 
            for tag in scalar_tags_list}
    try:
        ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        available_tags = ea.Tags()['scalars']
        for tag in scalar_tags_list:
            if tag in available_tags:
                events = ea.Scalars(tag)
                data[tag]['wall_times'] = np.array([event.wall_time for event in events])
                data[tag]['steps'] = np.array([event.step for event in events])
                data[tag]['values'] = np.array([event.value for event in events])
            else: print(f"Warning: Tag '{tag}' not found in {event_file_path}")
    except Exception as e: print(f"Error processing event file {event_file_path}: {e}")
    return data

def plot_data_with_std_shade(plot_title, xlabel, ylabel, output_filename, 
                             datasets, 
                             x_is_time=False, x_is_million_steps=False,
                             x_max_limit=None):
    sns.set_theme(style="whitegrid") 
    plt.figure(figsize=(12, 7)) # Slightly wider for more lines

    legend_handles_labels = {} 
    for x_data, y_data, std_data, label, color, linestyle in datasets:
        if not (x_data.size > 0 and y_data.size > 0 and len(x_data) == len(y_data)):
            print(f"Warning: No data or mismatched data lengths for main label '{label}'. Skipping.")
            continue

        x_plot = x_data / 1e6 if x_is_million_steps else x_data
        
        x_plot_filtered, y_data_filtered, std_data_filtered = x_plot, y_data, std_data
        if x_max_limit is not None:
            mask = x_plot <= x_max_limit
            if np.any(mask):
                last_true_idx = np.where(mask)[0][-1]
                if last_true_idx + 1 < len(mask): mask[last_true_idx + 1] = True
            
            x_plot_filtered = x_plot[mask]
            y_data_filtered = y_data[mask]
            if std_data is not None and std_data.size > 0 :
                std_data_filtered = std_data[mask[:len(std_data)]] 
            else: std_data_filtered = None
        
        if not (x_plot_filtered.size > 0 and y_data_filtered.size > 0): continue
            
        line, = plt.plot(x_plot_filtered, y_data_filtered, color=color, linestyle=linestyle, 
                         alpha=0.95, linewidth=2.0, label=label)
        if label not in legend_handles_labels: legend_handles_labels[label] = line

        if std_data_filtered is not None and std_data_filtered.size == y_data_filtered.size:
            plt.fill_between(x_plot_filtered, 
                             y_data_filtered - std_data_filtered, 
                             y_data_filtered + std_data_filtered, 
                             color=color, alpha=0.20)
        elif std_data is not None:
            print(f"Warning: Mismatched std_data for '{label}'. No shade plotted.")

    plt.title(plot_title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    if legend_handles_labels:
        legend = plt.legend(legend_handles_labels.values(), legend_handles_labels.keys(), 
                            fontsize=11, loc='lower right') # Consider 'best' or adjusting
        if legend: legend.get_frame().set_alpha(0.95)

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=12)

    left_xlim_val = 0 
    all_min_x = [np.min(d[0] / (1e6 if x_is_million_steps else 1)) for d in datasets if d[0].size > 0]
    if all_min_x and min(all_min_x) < 0: left_xlim_val = min(all_min_x)
    
    if x_max_limit is not None: plt.xlim(left=left_xlim_val, right=x_max_limit)
    else: plt.xlim(left=left_xlim_val)

    all_y_vals_plot = [val for _, y_data, std_data, _, _, _ in datasets 
                       for val in (y_data - (std_data if std_data is not None and std_data.size==y_data.size else 0) if y_data.size > 0 else [])] + \
                      [val for _, y_data, std_data, _, _, _ in datasets 
                       for val in (y_data + (std_data if std_data is not None and std_data.size==y_data.size else 0) if y_data.size > 0 else [])]
    if all_y_vals_plot:
        y_min_plot = np.nanmin(all_y_vals_plot)
        y_max_plot = np.nanmax(all_y_vals_plot)
        if not (np.isnan(y_min_plot) or np.isnan(y_max_plot)):
            y_padding = (y_max_plot - y_min_plot) * 0.05 
            plt.ylim(y_min_plot - y_padding, y_max_plot + y_padding)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, output_filename)
    plt.savefig(save_path, dpi=300) 
    plt.close()
    print(f"Plot saved: {save_path}")

# --- Main Script ---
if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Searching for experiment folders...")

    configs_to_plot = [
        {"name": CONFIG_NAME_PPO_MLP, "algo": ALGO_PPO, "label_suffix": "PPO MLP", "color": PPO_MLP_COLOR, "data": {}},
        {"name": CONFIG_NAME_SAC_MLP, "algo": ALGO_SAC, "label_suffix": "SAC MLP", "color": SAC_MLP_COLOR, "data": {}},
        {"name": CONFIG_NAME_PPO_RNN, "algo": ALGO_PPO, "label_suffix": "PPO RNN", "color": PPO_RNN_COLOR, "data": {}},
        {"name": CONFIG_NAME_SAC_RNN, "algo": ALGO_SAC, "label_suffix": "SAC RNN", "color": SAC_RNN_COLOR, "data": {}},
    ]
    
    # Add alias search for MLP configs
    alias_map = {
        CONFIG_NAME_PPO_MLP: CONFIG_NAME_DEFAULT_ALIAS,
        CONFIG_NAME_SAC_MLP: CONFIG_NAME_DEFAULT_ALIAS
    }

    for cfg_info in configs_to_plot:
        folder = find_latest_experiment_folder(EXPERIMENTS_DIR, cfg_info["name"], cfg_info["algo"])
        primary_name_used = cfg_info["name"]
        if not folder and cfg_info["name"] in alias_map:
            alias_name = alias_map[cfg_info["name"]]
            print(f"Folder for '{cfg_info['name']}_{cfg_info['algo']}_*' not found. Trying alias '{alias_name}_{cfg_info['algo']}_*'.")
            folder = find_latest_experiment_folder(EXPERIMENTS_DIR, alias_name, cfg_info["algo"])
            if folder: primary_name_used = alias_name # Update the name used for display if alias found
        
        cfg_info["display_name"] = primary_name_used # Store the name part for legend

        if folder:
            print(f"\nProcessing folder: {os.path.basename(folder)} for {cfg_info['label_suffix']}")
            event_files = glob.glob(os.path.join(folder, "tensorboard", "events.out.tfevents.*"))
            if event_files:
                tags_to_extract = [REWARD_AVG_100_TAG, REWARD_EPISODE_TAG]
                if cfg_info["algo"] == ALGO_SAC: # SAC needs mapping tags
                    tags_to_extract.extend([SAC_ENV_STEPS_TAG_EPISODE_BASED, SAC_TOTAL_UPDATES_TAG_EPISODE_BASED])
                cfg_info["data"] = extract_scalar_data(event_files[0], tags_to_extract)
            else:
                print(f"Error: No event file found in folder: {os.path.join(folder, 'tensorboard')}")
        else:
            print(f"Warning: Could not find experiment folder for {cfg_info['label_suffix']}. It will be skipped in plots.")

    # --- Plot 1: Reward vs Environment Steps (Millions) ---
    print("\nPreparing data for Plot 1: Reward vs Environment Steps (Millions)...")
    datasets_plot1 = []
    for cfg_info in configs_to_plot:
        if not cfg_info["data"]: continue # Skip if no data extracted

        avg_y = cfg_info["data"].get(REWARD_AVG_100_TAG, {}).get('values', np.array([]))
        avg_x_orig = cfg_info["data"].get(REWARD_AVG_100_TAG, {}).get('steps', np.array([]))
        ep_y = cfg_info["data"].get(REWARD_EPISODE_TAG, {}).get('values', np.array([]))
        ep_x_orig = cfg_info["data"].get(REWARD_EPISODE_TAG, {}).get('steps', np.array([]))

        avg_x_env = avg_x_orig
        ep_x_env = ep_x_orig
        
        # Interpolation for SAC variants
        if cfg_info["algo"] == ALGO_SAC:
            map_updates = cfg_info["data"].get(SAC_TOTAL_UPDATES_TAG_EPISODE_BASED, {}).get('values', np.array([]))
            map_env_steps = cfg_info["data"].get(SAC_ENV_STEPS_TAG_EPISODE_BASED, {}).get('values', np.array([]))

            if map_updates.size >= 2 and map_env_steps.size >= 2:
                sort_idx_map = np.argsort(map_updates)
                map_updates_sorted = map_updates[sort_idx_map]
                map_env_steps_sorted = map_env_steps[sort_idx_map]
                min_map_upd, max_map_upd = map_updates_sorted[0], map_updates_sorted[-1]

                if avg_x_orig.size > 0:
                    valid_mask_avg = (avg_x_orig >= min_map_upd) & (avg_x_orig <= max_map_upd)
                    avg_x_orig_filt = avg_x_orig[valid_mask_avg]
                    avg_y_filt = avg_y[valid_mask_avg]
                    if avg_x_orig_filt.size > 0:
                        avg_x_env = np.interp(avg_x_orig_filt, map_updates_sorted, map_env_steps_sorted)
                        avg_y = avg_y_filt # Y values correspond to the filtered original X
                
                if ep_x_orig.size > 0:
                    valid_mask_ep = (ep_x_orig >= min_map_upd) & (ep_x_orig <= max_map_upd)
                    ep_x_orig_filt = ep_x_orig[valid_mask_ep]
                    ep_y_filt = ep_y[valid_mask_ep]
                    if ep_x_orig_filt.size > 0:
                        ep_x_env = np.interp(ep_x_orig_filt, map_updates_sorted, map_env_steps_sorted)
                        ep_y = ep_y_filt # Y values correspond
            else: # Fallback if mapping data insufficient
                print(f"Warning: Insufficient mapping data for {cfg_info['label_suffix']}. Plotting vs updates.")
                avg_x_env = avg_x_orig
                ep_x_env = ep_x_orig
        
        # Calculate rolling std on episode data (using env_steps x-axis)
        std_y_final = None
        avg_x_final = np.array([])
        avg_y_final = np.array([])

        if ep_y.size > STD_WINDOW_SIZE and ep_x_env.size == ep_y.size :
            sort_idx_ep = np.argsort(ep_x_env)
            ep_x_env_sorted = ep_x_env[sort_idx_ep]
            ep_y_sorted = ep_y[sort_idx_ep]
            
            rolling_std = pd.Series(ep_y_sorted).rolling(window=STD_WINDOW_SIZE, min_periods=1).std().to_numpy()
            
            # Interpolate std to the avg_x_env axis
            if avg_x_env.size > 0 and ep_x_env_sorted.size > 0:
                sort_idx_avg = np.argsort(avg_x_env) # Ensure avg_x_env is sorted for interpolation target
                avg_x_final = avg_x_env[sort_idx_avg]
                avg_y_final = avg_y[sort_idx_avg]

                std_y_final = np.interp(avg_x_final, ep_x_env_sorted, rolling_std, left=np.nan, right=np.nan)
                std_y_final = pd.Series(std_y_final).fillna(method='bfill').fillna(method='ffill').to_numpy()
        elif avg_y.size > 0 : # If no std can be calculated, use avg data directly
             avg_x_final = avg_x_env
             avg_y_final = avg_y
        
        if avg_x_final.size > 0:
            datasets_plot1.append((avg_x_final, avg_y_final, std_y_final,
                                   cfg_info["label_suffix"], cfg_info["color"], '-'))

    if datasets_plot1:
        plot_data_with_std_shade(plot_title="RL Algorithm Comparison: Reward vs Environment Steps",
                                 xlabel="Environment Steps (Millions)",
                                 ylabel="Average Return (Avg 100 Episodes)",
                                 output_filename="reward_vs_steps_all_shaded_full.png",
                                 datasets=datasets_plot1,
                                 x_is_million_steps=True,
                                 x_max_limit=STEPS_PLOT_X_MAX_MILLIONS)
    else:
        print("No data available to generate Plot 1 (Reward vs Steps).")

    # --- Plot 2: Reward vs Running Time ---
    print("\nPreparing data for Plot 2: Reward vs Running Time...")
    datasets_plot2 = []
    for cfg_info in configs_to_plot:
        if not cfg_info["data"]: continue

        avg_y_time = cfg_info["data"].get(REWARD_AVG_100_TAG, {}).get('values', np.array([]))
        avg_wall_raw = cfg_info["data"].get(REWARD_AVG_100_TAG, {}).get('wall_times', np.array([]))
        ep_y_time = cfg_info["data"].get(REWARD_EPISODE_TAG, {}).get('values', np.array([]))
        ep_wall_raw = cfg_info["data"].get(REWARD_EPISODE_TAG, {}).get('wall_times', np.array([]))
        
        std_y_final_time = None
        avg_x_final_time_hours = np.array([])
        avg_y_final_time = np.array([])

        if avg_wall_raw.size > 0:
            t0_avg = avg_wall_raw[0]
            avg_x_final_time_hours = (avg_wall_raw - t0_avg) / 3600.0
            avg_y_final_time = avg_y_time # Assuming avg_y_time is already aligned with avg_wall_raw
        
        if ep_y_time.size > STD_WINDOW_SIZE and ep_wall_raw.size > 0 and avg_x_final_time_hours.size > 0:
            t0_ep = ep_wall_raw[0]
            ep_x_time_hours = (ep_wall_raw - t0_ep) / 3600.0

            sort_idx_ep_t = np.argsort(ep_x_time_hours)
            ep_x_time_sorted = ep_x_time_hours[sort_idx_ep_t]
            ep_y_time_sorted = ep_y_time[sort_idx_ep_t]
            rolling_std_time = pd.Series(ep_y_time_sorted).rolling(window=STD_WINDOW_SIZE, min_periods=1).std().to_numpy()
            
            # Ensure avg_x_final_time_hours and avg_y_final_time are sorted by time for interpolation
            sort_idx_avg_t = np.argsort(avg_x_final_time_hours)
            avg_x_final_time_sorted = avg_x_final_time_hours[sort_idx_avg_t]
            avg_y_final_time_sorted = avg_y_final_time[sort_idx_avg_t]

            std_y_final_time = np.interp(avg_x_final_time_sorted, ep_x_time_sorted, rolling_std_time, left=np.nan, right=np.nan)
            std_y_final_time = pd.Series(std_y_final_time).fillna(method='bfill').fillna(method='ffill').to_numpy()
            
            # Use sorted time and y for plotting
            avg_x_final_time_hours = avg_x_final_time_sorted
            avg_y_final_time = avg_y_final_time_sorted
            
        elif avg_x_final_time_hours.size > 0 : # If no std, use the avg data
            pass # avg_x_final_time_hours and avg_y_final_time are already set

        if avg_x_final_time_hours.size > 0:
            datasets_plot2.append((avg_x_final_time_hours, avg_y_final_time, std_y_final_time,
                                   cfg_info["label_suffix"], cfg_info["color"], '-'))

    if datasets_plot2:
        plot_data_with_std_shade(plot_title="RL Algorithm Comparison: Reward vs Running Time",
                                 xlabel="Running Time (hours)",
                                 ylabel="Average Return (Avg 100 Episodes)",
                                 output_filename="reward_vs_time_all_shaded_full.png",
                                 datasets=datasets_plot2,
                                 x_is_time=True,
                                 x_max_limit=TIME_PLOT_X_MAX_HOURS)
    else:
        print("No data available to generate Plot 2 (Reward vs Time).")

    print("\nScript finished.")