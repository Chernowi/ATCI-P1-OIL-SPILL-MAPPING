import os
import sys

# Attempt to import CONFIGS.
# (Import logic remains the same)
try:
    # First try to import from parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from configs import CONFIGS
except ImportError:
    # Then try from src directory
    src_dir = os.path.join(parent_dir, "src")
    if os.path.isdir(src_dir):
        sys.path.insert(0, src_dir)
        try:
            from configs import CONFIGS
        except ImportError:
            print("Error: Could not import CONFIGS from '../configs.py' or '../src/configs.py'.")
            print("Please ensure this script is run from the scripts directory and configs.py is in the parent directory.")
            sys.exit(1)
    else:
        print("Error: Could not import CONFIGS.")
        print("Please ensure configs.py exists in the parent directory of the scripts folder.")
        sys.exit(1)


# SLURM template - strictly using your provided structure
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=nct328
#SBATCH --qos=acc
#SBATCH --time=01-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nct/nct01026/ATCI-P1-BLOBTRACER/
#SBATCH --output={output_log_path}
#SBATCH --error={error_log_path}

module purge

module load  impi  intel  hdf5  mkl  python/3.12.1-gcc
#module load EB/apps EB/install cuda/12.6 cudnn/9.6.0-cuda12

cd ~/ATCI-P1-BLOBTRACER/
python src/bsc_main.py -c {config_key}
"""

# --- Define these paths using FORWARD SLASHES as they are for the Linux target ---
BASE_PROJECT_DIR_LINUX = "/home/nct/nct01026/ATCI-P1-BLOBTRACER"
BASE_LOG_DIR_LINUX = f"{BASE_PROJECT_DIR_LINUX}/out_logs" # Use f-string or string concat with '/'

# --- Define these paths using os.path.join for local operations by this script ---
# This script's current directory
CURRENT_SCRIPT_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_OUTPUT_DIR_NAME = "generated_slurm_sh_scripts"
# Local path to where .sh files will be saved by this script
OUTPUT_PATH_FOR_SH_FILES_LOCAL = os.path.join(CURRENT_SCRIPT_LOCAL_DIR, SCRIPT_OUTPUT_DIR_NAME)


def sanitize_for_path_and_job_name(name_str):
    name_str = name_str.replace("_", "-")
    sanitized = ''.join(c if c.isalnum() or c == '-' else '' for c in name_str)
    return sanitized[:30]

def main():
    if not os.path.exists(OUTPUT_PATH_FOR_SH_FILES_LOCAL):
        os.makedirs(OUTPUT_PATH_FOR_SH_FILES_LOCAL)
        print(f"Created directory for .sh files: {OUTPUT_PATH_FOR_SH_FILES_LOCAL}")

    config_keys = sorted(list(CONFIGS.keys()))

    if not config_keys:
        print("No configurations found in CONFIGS dictionary from configs.py. Exiting.")
        return

    all_sbatch_commands = []

    for config_key_original in config_keys:
        print(f"Generating SLURM .sh script for config: {config_key_original}")

        log_and_job_name_part = sanitize_for_path_and_job_name(config_key_original)
        sbatch_job_name = f"dl-{log_and_job_name_part}"

        # --- Construct Linux paths for the SLURM script ---
        log_dir_for_this_config_linux = f"{BASE_LOG_DIR_LINUX}/{log_and_job_name_part}"
        output_log_linux = f"{log_dir_for_this_config_linux}/job_output.log"
        error_log_linux = f"{log_dir_for_this_config_linux}/job_error.log"

        # --- Create the log directory locally if needed (using os.path.join for local FS) ---
        # This is so the Python script running on (e.g.) Windows can create the dir structure
        # that the Linux job will expect.
        local_log_dir_for_this_config = os.path.join(
            BASE_PROJECT_DIR_LINUX.replace("/", os.sep), # Make base project dir local
            "out_logs",
            log_and_job_name_part
        )
        # If generate_launchers.py is run from *within* BASE_PROJECT_DIR_LINUX, then:
        # local_log_dir_for_this_config = os.path.join("out_logs", log_and_job_name_part)

        # For simplicity, assuming this script is at the project root or can create dirs relative to it.
        # We will use the Linux path and let the `mkdir -p` (if re-added to SLURM) or manual creation handle it on HPC.
        # OR, this script creates the directory structure using local paths *before* job submission
        # For now, let's assume the local script *can* create this structure if it's meant to be
        # pre-created on the system where this Python script runs.
        # If the target `BASE_PROJECT_DIR_LINUX` is only on HPC, then this script cannot create those dirs.
        # Let's assume the `mkdir -p` within the SLURM script is the most robust way if paths are absolute HPC paths.
        # Since `mkdir` was removed from template, this script *must* create them if they are absolute on HPC.
        # Re-evaluating: The `#SBATCH --chdir` sets the CWD. Log paths are relative to that, or absolute.
        # Your template has absolute paths for --output and --error.

        # Let's create the directory that will be used on HPC, using this Python script
        # This assumes the full path is accessible or can be created from where this script runs.
        # This is generally NOT the case if this script runs on Windows and target is Linux absolute path.
        # The best approach is for the SLURM script to create its own log directory.
        # Since you removed `mkdir`, I will make this script attempt to create the local equivalent
        # of the HPC directory structure if the script is run in the project root.

        # Path for this script to create log dirs locally.
        # This path is constructed based on the assumption that generate_launchers.py
        # is in BASE_PROJECT_DIR_LINUX when it runs.
        log_dir_to_create_locally = os.path.join(
            os.getcwd(), # Assumes script is run from project root
            "out_logs",
            log_and_job_name_part
        )
        if not os.path.exists(log_dir_to_create_locally):
            try:
                os.makedirs(log_dir_to_create_locally)
            except OSError as e:
                print(f"  Warning: Could not create local log directory {log_dir_to_create_locally}: {e}")


        script_content = SLURM_TEMPLATE.format(
            job_name=sbatch_job_name,
            output_log_path=output_log_linux, # Use Linux path
            error_log_path=error_log_linux,   # Use Linux path
            config_key=config_key_original
        )

        sh_script_filename_base = f"run_{sanitize_for_path_and_job_name(config_key_original)}.sh"
        # Local path for saving the .sh file
        sh_script_full_path_local = os.path.join(OUTPUT_PATH_FOR_SH_FILES_LOCAL, sh_script_filename_base)

        with open(sh_script_full_path_local, "w", newline='\n') as f:
            f.write(script_content)
        os.chmod(sh_script_full_path_local, 0o755)

        # The sbatch command refers to the script's location relative to where sbatch is run.
        # If sbatch is run from BASE_PROJECT_DIR_LINUX, then the path to script is:
        # SCRIPT_OUTPUT_DIR_NAME (relative) / sh_script_filename_base
        sbatch_command = f"sbatch -A nct_328 -q acc_training {SCRIPT_OUTPUT_DIR_NAME}/{sh_script_filename_base}"
        all_sbatch_commands.append(sbatch_command)

    print(f"\nSuccessfully generated {len(config_keys)} SLURM .sh scripts in: {OUTPUT_PATH_FOR_SH_FILES_LOCAL}")
    print(f"Log files will be written to subdirectories under: {BASE_LOG_DIR_LINUX}")
    print("  (Ensure these log directories exist or can be created on the HPC before jobs run if mkdir is not in SLURM script)")


    print("\n--- List of sbatch commands to run (assuming execution from project root) ---")
    for cmd in all_sbatch_commands:
        print(cmd)

    commands_file_path = os.path.join(CURRENT_SCRIPT_LOCAL_DIR, "all_sbatch_submission_commands.txt")
    with open(commands_file_path, "w", newline='\n') as f:
        for cmd in all_sbatch_commands:
            f.write(cmd + "\n")
    print(f"\nList of sbatch commands also saved to: {commands_file_path}")

    print("\nExample of how to run all commands from the text file (from the project root):")
    print(f"  while IFS= read -r line; do eval \"$line\"; sleep 1; done < {commands_file_path}")


if __name__ == "__main__":
    main()