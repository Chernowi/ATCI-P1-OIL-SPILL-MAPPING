import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import glob
import seaborn as sns
import pandas as pd

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

ALL_CONFIG_NAMES = [
    "default_sac_mlp", "default_ppo_mlp", "default_sac_rnn", "default_ppo_rnn",
    "sac_mlp_actor_lr_low", "sac_mlp_actor_lr_high",
    "sac_mlp_critic_lr_low", "sac_mlp_critic_lr_high",
    "sac_mlp_gamma_low", "sac_mlp_gamma_high",
    "sac_mlp_tau_low", "sac_mlp_tau_high",
    "sac_mlp_hidden_dims_small", "sac_mlp_hidden_dims_large",
    "sac_mlp_per",
    "ppo_mlp_actor_lr_low", "ppo_mlp_actor_lr_high",
    "ppo_mlp_gae_lambda_low", "ppo_mlp_gae_lambda_high",
    "ppo_mlp_policy_clip_low", "ppo_mlp_policy_clip_high",
    "ppo_mlp_entropy_coef_low", "ppo_mlp_entropy_coef_high",
    "ppo_mlp_hidden_dim_small", "ppo_mlp_hidden_dim_large",
    "sac_rnn_rnn_hidden_size_small", "sac_rnn_rnn_hidden_size_big",
    "default_mapping"
]
ALGO_SAC = "sac"
ALGO_PPO = "ppo"
REWARD_AVG_100_TAG = "Reward/Average_100"
REWARD_EPISODE_TAG = "Reward/Episode"
METRIC_AVG_EP_TAG = "Performance/Metric_AvgEp" # Still used if not overridden
METRIC_END_EP_TAG = "Performance/Metric_EndEp" # Source for rolling avg metric
SAC_ENV_STEPS_TAG_EPISODE_BASED = "Progress/Total_Env_Steps"
SAC_TOTAL_UPDATES_TAG_EPISODE_BASED = "Progress/Total_Updates"

palette = sns.color_palette("deep", 10)
CONSISTENT_LINESTYLE = '-'
STEPS_PLOT_X_MAX_MILLIONS = None
TIME_PLOT_X_MAX_HOURS = None
STD_WINDOW_SIZE = 100 # For the shaded standard deviation region
METRIC_ROLLING_AVG_WINDOW = 100 # For the main metric line's rolling average

# --- Helper Functions (infer_algo_from_config_name, find_latest_experiment_folder, extract_scalar_data) ---
# Assume these are present and correct as per your previous version. I'll include them for completeness.

def infer_algo_from_config_name(config_name):
    if config_name.startswith("sac_") or "sac" in config_name.lower():
        return ALGO_SAC
    if config_name.startswith("ppo_") or "ppo" in config_name.lower():
        return ALGO_PPO
    if config_name == "default_mapping": return ALGO_SAC # default_mapping is an alias for SAC MLP
    return None

def find_latest_experiment_folder(base_dir, config_name_pattern, algo_name_pattern):
    pattern = os.path.join(base_dir, f"{config_name_pattern}_{algo_name_pattern}_*")
    folders = glob.glob(pattern)

    if not folders: # Fallback for cases like "default_mapping_sac_timestamp" vs "sac_mlp_sac_timestamp"
        pattern_alt = os.path.join(base_dir, f"{config_name_pattern}_*") # default_mapping_*
        folders_alt = glob.glob(pattern_alt)
        valid_alt_folders = []
        for f_alt in folders_alt:
            basename = os.path.basename(f_alt)
            parts = basename.split('_')
            # Check if algo_name_pattern is part of the folder name, typically before timestamp
            # e.g., default_mapping_sac_12345 or my_config_ppo_67890
            if len(parts) > 1 and (parts[-2] == algo_name_pattern or algo_name_pattern in basename) :
                 valid_alt_folders.append(f_alt)
        folders = valid_alt_folders if valid_alt_folders else folders_alt


    if not folders: return None

    # Filter for folders that end with a timestamp (digits)
    # and match the expected config_name and algo_name structure more closely
    filtered_folders = []
    for f in folders:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) > 0 and parts[-1].isdigit(): # Ends with timestamp
            # Strict check: config_name_pattern then algo_name_pattern then timestamp
            expected_prefix_strict = f"{config_name_pattern}_{algo_name_pattern}_"
            # More relaxed: config_name itself contains algo, e.g. "sac_mlp" with algo "sac"
            expected_prefix_embedded = f"{config_name_pattern}_" # e.g. sac_mlp_...

            if basename.startswith(expected_prefix_strict):
                 filtered_folders.append(f)
            elif algo_name_pattern in config_name_pattern and basename.startswith(expected_prefix_embedded): # e.g. config "sac_mlp", algo "sac"
                 filtered_folders.append(f)
            elif config_name_pattern == "default_mapping" and basename.startswith(f"default_mapping_{algo_name_pattern}_"):
                 filtered_folders.append(f)
            # Add more specific heuristics if needed for other naming conventions

    if not filtered_folders: # If strict filtering yielded nothing, broaden slightly
        potential_folders = [f for f in folders if os.path.basename(f).split('_')[-1].isdigit()]
        filtered_folders = potential_folders if potential_folders else folders # Fallback to original list if still nothing

    if not filtered_folders: return None

    try: # Sort by the timestamp part
        filtered_folders.sort(key=lambda f: int(os.path.basename(f).split('_')[-1]))
        return filtered_folders[-1]
    except ValueError: # If timestamp is not purely numeric for some reason
        return filtered_folders[-1] if filtered_folders else None


def extract_scalar_data(event_file_path, scalar_tags_list):
    data = {tag: {'wall_times': np.array([]), 'steps': np.array([]), 'values': np.array([])} for tag in scalar_tags_list}
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
    except Exception as e:
        print(f"Error processing event file {event_file_path}: {e}")
    return data

def get_plot_data(exp_data, y_source_tag, x_axis_type="env_steps", std_source_y_tag=None, y_rolling_avg_window=None):
    # --- Primary Y-data (for the line plot) ---
    y_values_main_raw = exp_data.get(y_source_tag, {}).get('values', np.array([]))
    x_orig_steps_main = exp_data.get(y_source_tag, {}).get('steps', np.array([]))
    wall_times_raw_main = exp_data.get(y_source_tag, {}).get('wall_times', np.array([]))

    if y_values_main_raw.size == 0: # No data for the primary y-source
        return np.array([]), np.array([]), None

    if y_rolling_avg_window and y_values_main_raw.size >= 1:
        y_plot_processed = pd.Series(y_values_main_raw).rolling(window=y_rolling_avg_window, min_periods=1).mean().to_numpy()
    else:
        y_plot_processed = y_values_main_raw

    # --- X-axis for the main plot line ---
    x_plot_main_final = np.array([])
    y_plot_main_final = np.array([]) # This will hold y_plot_processed after filtering for SAC

    if x_axis_type == "env_steps":
        if exp_data.get("_algo") == ALGO_SAC:
            map_updates = exp_data.get(SAC_TOTAL_UPDATES_TAG_EPISODE_BASED, {}).get('values', np.array([]))
            map_env_steps = exp_data.get(SAC_ENV_STEPS_TAG_EPISODE_BASED, {}).get('values', np.array([]))
            if map_updates.size >= 2 and map_env_steps.size >= 2 and x_orig_steps_main.size > 0:
                sort_idx_map = np.argsort(map_updates)
                map_updates_sorted, map_env_steps_sorted = map_updates[sort_idx_map], map_env_steps[sort_idx_map]
                min_map_upd, max_map_upd = map_updates_sorted[0], map_updates_sorted[-1]
                
                valid_mask = (x_orig_steps_main >= min_map_upd) & (x_orig_steps_main <= max_map_upd)
                x_orig_main_filt = x_orig_steps_main[valid_mask]
                y_plot_processed_filt = y_plot_processed[valid_mask]

                if x_orig_main_filt.size > 0:
                    x_plot_main_final = np.interp(x_orig_main_filt, map_updates_sorted, map_env_steps_sorted)
                    y_plot_main_final = y_plot_processed_filt
                # else: x_plot_main_final, y_plot_main_final remain empty
            elif x_orig_steps_main.size > 0 : # Not enough mapping data, or x_orig_steps_main empty
                x_plot_main_final = x_orig_steps_main 
                y_plot_main_final = y_plot_processed
            # else: x_plot_main_final, y_plot_main_final remain empty
        else: # PPO
            x_plot_main_final = x_orig_steps_main
            y_plot_main_final = y_plot_processed
    elif x_axis_type == "time":
        if wall_times_raw_main.size > 0:
            x_plot_main_final = (wall_times_raw_main - wall_times_raw_main[0]) / 3600.0
            y_plot_main_final = y_plot_processed
        # else: x_plot_main_final, y_plot_main_final remain empty

    if x_plot_main_final.size == 0: # If main plot line can't be formed
        return np.array([]), np.array([]), None

    # --- Standard Deviation Data (for the shaded region) ---
    std_source_y_tag_resolved = std_source_y_tag if std_source_y_tag else REWARD_EPISODE_TAG
    y_values_std_raw = exp_data.get(std_source_y_tag_resolved, {}).get('values', np.array([]))
    x_orig_steps_std = exp_data.get(std_source_y_tag_resolved, {}).get('steps', np.array([]))
    wall_times_raw_std = exp_data.get(std_source_y_tag_resolved, {}).get('wall_times', np.array([]))

    std_plot_final_interp = None
    if y_values_std_raw.size > STD_WINDOW_SIZE:
        x_for_std_calc_raw = np.array([])
        y_for_std_calc_raw_effective = np.array([])

        if x_axis_type == "env_steps":
            if exp_data.get("_algo") == ALGO_SAC:
                map_updates_std = exp_data.get(SAC_TOTAL_UPDATES_TAG_EPISODE_BASED, {}).get('values', np.array([]))
                map_env_steps_std = exp_data.get(SAC_ENV_STEPS_TAG_EPISODE_BASED, {}).get('values', np.array([]))
                if map_updates_std.size >= 2 and map_env_steps_std.size >= 2 and x_orig_steps_std.size > 0:
                    sort_idx_map_std = np.argsort(map_updates_std)
                    map_upd_s_std, map_env_s_std = map_updates_std[sort_idx_map_std], map_env_steps_std[sort_idx_map_std]
                    min_m_u_std, max_m_u_std = map_upd_s_std[0], map_upd_s_std[-1]

                    valid_mask_std = (x_orig_steps_std >= min_m_u_std) & (x_orig_steps_std <= max_m_u_std)
                    x_orig_std_filt = x_orig_steps_std[valid_mask_std]
                    y_val_std_raw_filt = y_values_std_raw[valid_mask_std]
                    
                    if x_orig_std_filt.size > 0:
                        x_for_std_calc_raw = np.interp(x_orig_std_filt, map_upd_s_std, map_env_s_std)
                        y_for_std_calc_raw_effective = y_val_std_raw_filt
                elif x_orig_steps_std.size > 0:
                    x_for_std_calc_raw = x_orig_steps_std
                    y_for_std_calc_raw_effective = y_values_std_raw
            else: # PPO
                x_for_std_calc_raw = x_orig_steps_std
                y_for_std_calc_raw_effective = y_values_std_raw
        elif x_axis_type == "time":
            if wall_times_raw_std.size > 0:
                x_for_std_calc_raw = (wall_times_raw_std - wall_times_raw_std[0]) / 3600.0
                y_for_std_calc_raw_effective = y_values_std_raw
        
        if x_for_std_calc_raw.size > STD_WINDOW_SIZE and y_for_std_calc_raw_effective.size == x_for_std_calc_raw.size:
            sort_idx_std_calc = np.argsort(x_for_std_calc_raw)
            x_std_calc_sorted = x_for_std_calc_raw[sort_idx_std_calc]
            y_std_calc_sorted = y_for_std_calc_raw_effective[sort_idx_std_calc]
            
            rolling_std_values = pd.Series(y_std_calc_sorted).rolling(window=STD_WINDOW_SIZE, min_periods=1).std().to_numpy()

            if x_plot_main_final.size > 0 and x_std_calc_sorted.size > 0:
                # Sort main plot data before interpolation
                sort_idx_main_final = np.argsort(x_plot_main_final)
                x_plot_main_final_sorted = x_plot_main_final[sort_idx_main_final]
                y_plot_main_final_sorted = y_plot_main_final[sort_idx_main_final]

                std_plot_final_interp = np.interp(x_plot_main_final_sorted, x_std_calc_sorted, rolling_std_values, left=np.nan, right=np.nan)
                std_plot_final_interp = pd.Series(std_plot_final_interp).fillna(method='bfill').fillna(method='ffill').to_numpy()
                
                x_plot_main_final = x_plot_main_final_sorted
                y_plot_main_final = y_plot_main_final_sorted

    # Final sort if not sorted during std dev processing (e.g. if std dev wasn't plotted)
    if x_plot_main_final.size > 0 and (std_plot_final_interp is None or std_plot_final_interp.size != x_plot_main_final.size):
        current_x_sorting = np.argsort(x_plot_main_final)
        # Check if already sorted to avoid unnecessary re-indexing
        if not np.array_equal(current_x_sorting, np.arange(len(x_plot_main_final))):
             x_plot_main_final = x_plot_main_final[current_x_sorting]
             y_plot_main_final = y_plot_main_final[current_x_sorting]
    
    # Handle NaNs in final outputs
    if np.any(np.isnan(x_plot_main_final)) : x_plot_main_final = np.nan_to_num(x_plot_main_final)
    if np.any(np.isnan(y_plot_main_final)) : y_plot_main_final = np.nan_to_num(y_plot_main_final)
    if std_plot_final_interp is not None and np.any(np.isnan(std_plot_final_interp)):
        std_plot_final_interp = np.nan_to_num(std_plot_final_interp)

    return x_plot_main_final, y_plot_main_final, std_plot_final_interp


def plot_data_with_std_shade(plot_title, xlabel, ylabel, output_filename, datasets,
                             x_is_time=False, x_is_million_steps=False, x_max_limit=None, y_max_limit=None, y_min_limit=None):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    legend_handles_labels = {}
    max_x_val_observed = 0
    min_x_val_observed = float('inf')
    all_y_data_for_ylim = []

    for x_data, y_data, std_data, label, color, linestyle in datasets:
        if not (x_data.size > 0 and y_data.size > 0 and len(x_data) == len(y_data)):
            # print(f"Skipping dataset '{label}' due to empty or mismatched data.")
            continue
        
        x_plot = x_data / 1e6 if x_is_million_steps else x_data
        
        if x_plot.size > 0:
            max_x_val_observed = max(max_x_val_observed, np.max(x_plot))
            min_x_val_observed = min(min_x_val_observed, np.min(x_plot))
            all_y_data_for_ylim.extend(y_data)
            if std_data is not None and std_data.size == y_data.size:
                all_y_data_for_ylim.extend(y_data - std_data)
                all_y_data_for_ylim.extend(y_data + std_data)

        # Filter data by x_max_limit
        x_plot_filtered, y_data_filtered, std_data_filtered = x_plot, y_data, std_data
        if x_max_limit is not None and x_plot.size > 0 :
            mask = x_plot <= x_max_limit
            # Extend mask to include one point beyond limit for smoother line end if available
            if np.any(mask) and x_plot_filtered.size > 0 and x_plot_filtered[-1] < x_max_limit and x_plot_filtered[-1] > x_max_limit * 0.9 : # Heuristic to check if close to end
                 last_true_idx = np.where(mask)[0][-1]
                 if last_true_idx + 1 < len(mask): # If there is a point after the last true one
                      mask[last_true_idx + 1] = True # Include it

            x_plot_filtered = x_plot[mask]
            y_data_filtered = y_data[mask]
            if std_data is not None and std_data.size > 0 :
                std_data_filtered = std_data[mask[:len(std_data)] if len(mask) <= len(std_data) else mask] # Ensure mask isn't too long for std_data
            else: std_data_filtered = None
        
        if not (x_plot_filtered.size > 0 and y_data_filtered.size > 0):
            # print(f"Skipping dataset '{label}' after x_max_limit filtering.")
            continue

        line, = plt.plot(x_plot_filtered, y_data_filtered, color=color, linestyle=linestyle, alpha=0.95, linewidth=2.0, label=label)
        if label not in legend_handles_labels: # Avoid duplicate legend entries if somehow a label is repeated
            legend_handles_labels[label] = line

        if std_data_filtered is not None and std_data_filtered.size == y_data_filtered.size:
            plt.fill_between(x_plot_filtered, 
                             np.nan_to_num(y_data_filtered - std_data_filtered), 
                             np.nan_to_num(y_data_filtered + std_data_filtered), 
                             color=color, alpha=0.20)

    plt.title(plot_title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    if legend_handles_labels:
        legend = plt.legend(legend_handles_labels.values(), legend_handles_labels.keys(), fontsize=10, loc='best')
        if legend: legend.get_frame().set_alpha(0.95)

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # X-axis limits
    left_xlim_val = 0
    if min_x_val_observed != float('inf') and min_x_val_observed < 0 : # Handle negative x if they occur
        left_xlim_val = min_x_val_observed
    current_xlim_right = x_max_limit if x_max_limit is not None else (max_x_val_observed if max_x_val_observed > 0 else 1) 
    if current_xlim_right <= left_xlim_val : current_xlim_right = left_xlim_val + 1 # Ensure right > left
    plt.xlim(left=left_xlim_val, right=current_xlim_right)

    # Y-axis limits
    if y_min_limit is not None and y_max_limit is not None:
        plt.ylim(y_min_limit, y_max_limit)
    elif all_y_data_for_ylim:
        y_min_plot_data = np.nanmin(all_y_data_for_ylim)
        y_max_plot_data = np.nanmax(all_y_data_for_ylim)
        if not (np.isnan(y_min_plot_data) or np.isnan(y_max_plot_data)) and y_max_plot_data > y_min_plot_data :
            y_padding = (y_max_plot_data - y_min_plot_data) * 0.05
            y_padding = max(y_padding, 0.01) if y_max_plot_data - y_min_plot_data < 0.2 else y_padding # Min padding for small ranges
            plt.ylim(y_min_plot_data - y_padding, y_max_plot_data + y_padding)
        elif not (np.isnan(y_min_plot_data) or np.isnan(y_max_plot_data)): # If min=max (e.g. flat line)
             plt.ylim(y_min_plot_data - 0.1, y_max_plot_data + 0.1) # Default small padding


    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, output_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved: {save_path}")

def load_all_experiment_data(config_names_list):
    print("--- Loading All Experiment Data ---")
    all_data = {}
    for config_name in config_names_list:
        algo = infer_algo_from_config_name(config_name)
        if not algo:
            print(f"Could not infer algorithm for config: {config_name}. Skipping.")
            continue

        folder_to_search = config_name
        algo_to_search = algo
        
        folder = find_latest_experiment_folder(EXPERIMENTS_DIR, folder_to_search, algo_to_search)
        
        # Special handling for "default_mapping" alias
        if not folder and config_name == "default_mapping":
            original_name_for_alias = "default_sac_mlp" # default_mapping is an alias for default_sac_mlp
            # print(f"Info: Trying alias '{original_name_for_alias}' for '{config_name}' with algo '{algo_to_search}'")
            folder = find_latest_experiment_folder(EXPERIMENTS_DIR, original_name_for_alias, algo_to_search)

        if folder:
            # print(f"Found folder for {config_name} (algo {algo}): {folder}")
            event_files = glob.glob(os.path.join(folder, "tensorboard", "events.out.tfevents.*"))
            if event_files:
                tags_to_extract = [REWARD_AVG_100_TAG, REWARD_EPISODE_TAG, METRIC_AVG_EP_TAG, METRIC_END_EP_TAG]
                if algo == ALGO_SAC:
                    tags_to_extract.extend([SAC_ENV_STEPS_TAG_EPISODE_BASED, SAC_TOTAL_UPDATES_TAG_EPISODE_BASED])
                
                data = extract_scalar_data(event_files[0], tags_to_extract)
                data["_algo"] = algo
                data["_config_name"] = config_name # Store original config name for reference
                all_data[config_name] = data
            else:
                print(f"Error: No event file in {os.path.join(folder, 'tensorboard')} for {config_name}")
                all_data[config_name] = None # Mark as None if no data found
        else:
            print(f"Error: No experiment folder found for {config_name} (algo {algo})")
            all_data[config_name] = None # Mark as None if no folder found
            
    print("--- Data Loading Complete ---")
    return all_data

# --- Plotting Functions for Specific Questions ---

def plot_general_comparison(all_exp_data, y_source_tag, y_label, file_suffix, std_source_tag, y_min=None, y_max=None, y_rolling_avg_window=None):
    print(f"\n--- Plotting General Algorithm Comparison ({y_label}) ---")
    configs_for_plot = ["default_sac_mlp", "default_ppo_mlp", "default_sac_rnn", "default_ppo_rnn"]
    
    # Handle default_mapping alias
    if "default_mapping" in all_exp_data and all_exp_data["default_mapping"] is not None:
        if "default_sac_mlp" not in configs_for_plot: # If default_sac_mlp isn't explicitly listed
            configs_for_plot.append("default_mapping")
        elif all_exp_data.get("default_sac_mlp") is None: # If default_sac_mlp is listed but has no data, replace with default_mapping
            configs_for_plot = [c if c != "default_sac_mlp" else "default_mapping" for c in configs_for_plot]
            
    datasets_steps = []
    datasets_time = []
    
    color_map = {
        "default_sac_mlp": palette[1], 
        "default_mapping": palette[1], # Same color for alias
        "default_ppo_mlp": palette[0],
        "default_sac_rnn": palette[3],
        "default_ppo_rnn": palette[2]
    }
    label_map = {
        "default_sac_mlp": "SAC MLP",
        "default_mapping": "SAC MLP (Default)", # Distinguish if both are somehow present
        "default_ppo_mlp": "PPO MLP",
        "default_sac_rnn": "SAC RNN",
        "default_ppo_rnn": "PPO RNN"
    }

    for i, config_key in enumerate(configs_for_plot):
        exp_data = all_exp_data.get(config_key)
        if not exp_data:
            # print(f"Warning: No data for '{config_key}' in general comparison.")
            continue

        label = label_map.get(config_key, config_key)
        color = color_map.get(config_key, palette[i % len(palette)]) # Fallback color

        x_steps, y_steps, std_steps = get_plot_data(exp_data, y_source_tag, "env_steps", std_source_tag, y_rolling_avg_window)
        if x_steps.size > 0:
            datasets_steps.append((x_steps, y_steps, std_steps, label, color, CONSISTENT_LINESTYLE))

        x_time, y_time, std_time = get_plot_data(exp_data, y_source_tag, "time", std_source_tag, y_rolling_avg_window)
        if x_time.size > 0:
            datasets_time.append((x_time, y_time, std_time, label, color, CONSISTENT_LINESTYLE))
            
    if datasets_steps:
        plot_data_with_std_shade(f"Algorithm Comparison: {y_label} vs Environment Steps", 
                                 "Environment Steps (Millions)", y_label,
                                 f"general_compare_{file_suffix}_vs_steps.png", datasets_steps, 
                                 x_is_million_steps=True, x_max_limit=STEPS_PLOT_X_MAX_MILLIONS, 
                                 y_min_limit=y_min, y_max_limit=y_max)
    if datasets_time:
        plot_data_with_std_shade(f"Algorithm Comparison: {y_label} vs Running Time", 
                                 "Running Time (hours)", y_label,
                                 f"general_compare_{file_suffix}_vs_time.png", datasets_time, 
                                 x_is_time=True, x_max_limit=TIME_PLOT_X_MAX_HOURS, 
                                 y_min_limit=y_min, y_max_limit=y_max)

def plot_hyperparam_comparison(all_exp_data, base_config_key, param_variations_with_values, 
                               param_name_short, y_source_tag, y_label, file_suffix_base,
                               std_source_tag, y_min=None, y_max=None, y_rolling_avg_window=None):
    print(f"\n--- Plotting {param_name_short} Hyperparameter Comparison for {base_config_key} ({y_label}) ---")
    
    base_exp_data = all_exp_data.get(base_config_key)
    if not base_exp_data:
        # Try alias if base_config_key is default_sac_mlp and it's missing, but default_mapping exists
        if base_config_key == "default_sac_mlp" and "default_mapping" in all_exp_data and all_exp_data["default_mapping"] is not None:
            base_exp_data = all_exp_data["default_mapping"]
            # print(f"Info: Using 'default_mapping' data for base '{base_config_key}'.")
        else:
            print(f"Base config data for '{base_config_key}' not found. Skipping plot for {param_name_short}.")
            return

    datasets_steps = []
    datasets_time = []
    
    num_lines = len(param_variations_with_values) + 1 # +1 for the base config
    current_palette = sns.color_palette("husl", num_lines) # Use a palette that distributes colors well

    base_label_detail = "Default" # Default label for the base configuration
    base_label = f"{param_name_short}: {base_label_detail}"
    base_color = current_palette[0]

    x_steps_base, y_steps_base, std_steps_base = get_plot_data(base_exp_data, y_source_tag, "env_steps", std_source_tag, y_rolling_avg_window)
    if x_steps_base.size > 0:
        datasets_steps.append((x_steps_base, y_steps_base, std_steps_base, base_label, base_color, CONSISTENT_LINESTYLE))

    x_time_base, y_time_base, std_time_base = get_plot_data(base_exp_data, y_source_tag, "time", std_source_tag, y_rolling_avg_window)
    if x_time_base.size > 0:
        datasets_time.append((x_time_base, y_time_base, std_time_base, base_label, base_color, CONSISTENT_LINESTYLE))

    # Parameter variations
    for i, (var_key, value_display_str) in enumerate(param_variations_with_values):
        var_data = all_exp_data.get(var_key)
        if not var_data:
            # print(f"Warning: Data for variation key '{var_key}' not found. Skipping.")
            continue
        
        var_label = f"{param_name_short}: {value_display_str}"
        var_color = current_palette[i + 1] # Offset by 1 due to base config

        x_steps_var, y_steps_var, std_steps_var = get_plot_data(var_data, y_source_tag, "env_steps", std_source_tag, y_rolling_avg_window)
        if x_steps_var.size > 0:
            datasets_steps.append((x_steps_var, y_steps_var, std_steps_var, var_label, var_color, CONSISTENT_LINESTYLE))

        x_time_var, y_time_var, std_time_var = get_plot_data(var_data, y_source_tag, "time", std_source_tag, y_rolling_avg_window)
        if x_time_var.size > 0:
            datasets_time.append((x_time_var, y_time_var, std_time_var, var_label, var_color, CONSISTENT_LINESTYLE))
    
    algo_name_for_file = base_exp_data['_algo']
    # Infer architecture (MLP/RNN) from base_config_key for file naming
    arch_name_for_file = 'rnn' if 'rnn' in base_config_key else 'mlp'
    
    plot_main_title = f"{algo_name_for_file.upper()} {arch_name_for_file.upper()}: {param_name_short} Comparison"

    if len(datasets_steps) > 1: # Only plot if we have more than just the base (i.e., at least one variation was plotted)
        plot_data_with_std_shade(f"{plot_main_title} - {y_label} vs Steps",
                                 "Environment Steps (Millions)", y_label,
                                 f"{algo_name_for_file}_{arch_name_for_file}_{file_suffix_base}_vs_steps.png",
                                 datasets_steps, x_is_million_steps=True, x_max_limit=STEPS_PLOT_X_MAX_MILLIONS,
                                 y_min_limit=y_min, y_max_limit=y_max)
    if len(datasets_time) > 1:
        plot_data_with_std_shade(f"{plot_main_title} - {y_label} vs Time",
                                 "Running Time (hours)", y_label,
                                 f"{algo_name_for_file}_{arch_name_for_file}_{file_suffix_base}_vs_time.png",
                                 datasets_time, x_is_time=True, x_max_limit=TIME_PLOT_X_MAX_HOURS,
                                 y_min_limit=y_min, y_max_limit=y_max)

# --- Main Script ---
if __name__ == "__main__":
    all_experiment_data = load_all_experiment_data(ALL_CONFIG_NAMES)

    # metrics_to_plot format:
    # (y_source_tag, y_axis_label, file_label_suffix, std_source_tag_for_plot, y_min_val, y_max_val, y_rolling_avg_window_for_plot [optional])
    metrics_to_plot = [
        (REWARD_AVG_100_TAG, "Avg Return (100Ep)", "reward_avg100", REWARD_EPISODE_TAG, None, None, None), # No rolling avg for this one
        (METRIC_END_EP_TAG, "Avg Performance Metric (100Ep Window)", "metric_avg100ep", METRIC_END_EP_TAG, -0.05, 1.05, METRIC_ROLLING_AVG_WINDOW) # Apply rolling avg
    ]

    for plot_spec in metrics_to_plot:
        y_source_tag, y_axis_label, file_label_suffix, std_source_tag_for_plot, y_min_val, y_max_val = plot_spec[:6]
        y_rolling_avg_window_for_plot = plot_spec[6] if len(plot_spec) > 6 else None

        plot_general_comparison(all_experiment_data, y_source_tag, y_axis_label, file_label_suffix,
                                std_source_tag=std_source_tag_for_plot, y_min=y_min_val, y_max=y_max_val,
                                y_rolling_avg_window=y_rolling_avg_window_for_plot)

        # SAC MLP Hyperparameters & PER
        sac_mlp_hyperparams = {
            "Actor LR": ([("sac_mlp_actor_lr_low", "1e-5"), ("sac_mlp_actor_lr_high", "1e-4")], "sac_actor_lr"),
            "Critic LR": ([("sac_mlp_critic_lr_low", "1e-5"), ("sac_mlp_critic_lr_high", "1e-4")], "sac_critic_lr"),
            "Gamma": ([("sac_mlp_gamma_low", "0.95"), ("sac_mlp_gamma_high", "0.999")], "sac_gamma"),
            "Tau": ([("sac_mlp_tau_low", "0.001"), ("sac_mlp_tau_high", "0.01")], "sac_tau"),
            "Hidden Dims": ([("sac_mlp_hidden_dims_small", "[64,64]"), ("sac_mlp_hidden_dims_large", "[256,256]")], "sac_hidden_dims"),
            "PER": ([("sac_mlp_per", "Enabled")], "sac_per_vs_default")
        }
        for param_short_name, (variations_with_values, file_suffix) in sac_mlp_hyperparams.items():
            plot_hyperparam_comparison(all_experiment_data, "default_sac_mlp", variations_with_values,
                                       param_short_name, y_source_tag, y_axis_label, f"{file_suffix}_{file_label_suffix}",
                                       std_source_tag=std_source_tag_for_plot, y_min=y_min_val, y_max=y_max_val,
                                       y_rolling_avg_window=y_rolling_avg_window_for_plot)

        # PPO MLP Hyperparameters
        ppo_mlp_hyperparams = {
            "Actor LR": ([("ppo_mlp_actor_lr_low", "1e-5"), ("ppo_mlp_actor_lr_high", "1e-4")], "ppo_actor_lr"),
            "GAE Lambda": ([("ppo_mlp_gae_lambda_low", "0.90"), ("ppo_mlp_gae_lambda_high", "0.99")], "ppo_gae_lambda"),
            "Policy Clip": ([("ppo_mlp_policy_clip_low", "0.1"), ("ppo_mlp_policy_clip_high", "0.3")], "ppo_policy_clip"),
            "Entropy Coef": ([("ppo_mlp_entropy_coef_low", "0.005"), ("ppo_mlp_entropy_coef_high", "0.5")], "ppo_entropy_coef"),
            "Hidden Dim": ([("ppo_mlp_hidden_dim_small", "128"), ("ppo_mlp_hidden_dim_large", "512")], "ppo_hidden_dim")
        }
        for param_short_name, (variations_with_values, file_suffix) in ppo_mlp_hyperparams.items():
            plot_hyperparam_comparison(all_experiment_data, "default_ppo_mlp", variations_with_values,
                                       param_short_name, y_source_tag, y_axis_label, f"{file_suffix}_{file_label_suffix}",
                                       std_source_tag=std_source_tag_for_plot, y_min=y_min_val, y_max=y_max_val,
                                       y_rolling_avg_window=y_rolling_avg_window_for_plot)
        
        # SAC RNN Hyperparameters
        sac_rnn_hyperparams = {
            "RNN Hidden Size": ([("sac_rnn_rnn_hidden_size_small", "32"), ("sac_rnn_rnn_hidden_size_big", "128")], "sac_rnn_hidden")
        }
        for param_short_name, (variations_with_values, file_suffix) in sac_rnn_hyperparams.items():
            plot_hyperparam_comparison(all_experiment_data, "default_sac_rnn", variations_with_values,
                                       param_short_name, y_source_tag, y_axis_label, f"{file_suffix}_{file_label_suffix}",
                                       std_source_tag=std_source_tag_for_plot, y_min=y_min_val, y_max=y_max_val,
                                       y_rolling_avg_window=y_rolling_avg_window_for_plot)

    print("\nPlotting script finished.")