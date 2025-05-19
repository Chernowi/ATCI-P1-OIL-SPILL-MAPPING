# RL Agent for Oil Spill Mapping

## Overview

This project implements Reinforcement Learning (RL) agents to control an autonomous vehicle for mapping oil spills. The agent learns to navigate an environment, use its sensors to detect oil, and build an estimate of the spill's shape using a Convex Hull-based mapper. The goal is to efficiently and accurately map the extent of the spill.

The project supports two popular RL algorithms: Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO), with options for both Multi-Layer Perceptron (MLP) and Recurrent Neural Network (RNN - LSTM/GRU) architectures.

## Features

*   **RL Algorithms**:
    *   Soft Actor-Critic (SAC)
    *   Proximal Policy Optimization (PPO)
*   **Network Architectures**:
    *   Multi-Layer Perceptrons (MLPs)
    *   Recurrent Neural Networks (LSTMs or GRUs) for handling state trajectories.
*   **Advanced RL Techniques**:
    *   State and Reward Normalization
    *   Prioritized Experience Replay (PER) for SAC
    *   Generalized Advantage Estimation (GAE) for PPO
    *   Auto-tuning of SAC's temperature parameter (alpha)
*   **Configurable Environment**:
    *   Customizable world size, agent speed, sensor properties, oil spill characteristics (size, location, randomness).
    *   Detailed reward shaping options.
*   **Spill Mapping**:
    *   Convex Hull-based mapper to estimate spill boundaries from sensor data.
    *   Performance metrics based on the accuracy of the spill estimate.
*   **Training & Evaluation**:
    *   Comprehensive training script with TensorBoard logging.
    *   Separate scripts for evaluating trained models.
    *   Experiment tracking: models, configurations, and logs are saved in unique, timestamped directories.
*   **Visualization**:
    *   Live rendering of the environment during evaluation.
    *   Ability to save evaluation episodes as GIFs or MP4 videos (requires FFMpeg for MP4).
*   **Detailed Configuration**:
    *   Centralized configuration system (`src/configs.py`) for all parameters.
    *   Extensive hyperparameter guide (`hyperparameter_guide.md`).

## Project Structure

```
.
├── README.md                   # This file
├── hyperparameter_guide.md     # Detailed guide to all configuration parameters
├── requirements.txt            # Python dependencies
└── src/                        # Source code
    ├── main.py                 # Main script for training agents
    ├── PPO.py                  # PPO algorithm implementation
    ├── SAC.py                  # SAC algorithm implementation
    ├── configs.py              # All configuration classes and default settings
    ├── mapper.py               # Oil spill mapper implementation
    ├── utils.py                # Utility functions (e.g., RunningMeanStd)
    ├── visualization.py        # Environment visualization logic
    ├── world.py                # Oil spill environment simulation
    ├── world_objects.py        # Basic objects for the world (Location, Velocity)
    ├── run_ppo_experiment.py   # Script to evaluate a trained PPO model
    └── run_sac_experiment.py   # Script to evaluate a trained SAC model
```

## Setup

### Prerequisites

*   Python 3.8 or higher (this project uses the `venv` module for virtual environments, which is standard).
*   pip (Python package installer)

### Installation Steps

1.  **Clone the repository (if applicable)**:
    ```bash
    git clone https://github.com/Chernowi/ATCI-P1-OIL-SPILL-MAPPING
    cd ATCI-P1-OIL-SPILL-MAPPING
    ```

2.  **Create a virtual environment**:
    It's highly recommended to use a virtual environment to manage project dependencies. This ensures that the project's dependencies don't interfere with other Python projects on your system.
    From the root directory of the project (where `requirements.txt` is located):
    ```bash
    python -m venv venv
    ```
    This command creates a new directory named `venv` which will contain the Python interpreter and libraries for this project.

3.  **Activate the virtual environment**:
    *   **On Windows**:
        ```bash
        venv\Scripts\activate
        ```
    *   **On macOS and Linux**:
        ```bash
        source venv/bin/activate
        ```
    Your command prompt should now indicate that you are in the `(venv)` environment.

4.  **Install dependencies**:
    With the virtual environment activated, install the required packages listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    **Note on `requirements.txt`**: This file lists all necessary Python packages. Key dependencies include:
    *   `torch` (PyTorch)
    *   `numpy`
    *   `matplotlib` (for visualization)
    *   `imageio` and `Pillow` (for GIF creation)
    *   `scipy` (for ConvexHull/Delaunay in mapper)
    *   `tqdm` (for progress bars)
    *   `pydantic` (for configuration management)
    *   `tensorboard` (for logging)

5.  **Visualization Libraries & FFMpeg (Optional but Recommended for full features)**:
    *   The `requirements.txt` file should install `matplotlib`, `imageio`, and `Pillow`.
    *   To save evaluation episodes as **MP4 videos**, you need to have **FFMpeg** installed and accessible in your system's PATH. If FFMpeg is not found, the visualization will fall back to saving GIFs. Installation of FFMpeg is system-dependent (e.g., using `apt` on Debian/Ubuntu, `brew` on macOS, or downloading binaries for Windows).

## Configuration

All agent, environment, training, and evaluation parameters are managed through `src/configs.py`. This file defines Pydantic models for structured configuration.

*   **Default Configurations**: `src/configs.py` provides several default configurations (e.g., `default_sac_mlp`, `default_ppo_rnn`) stored in the `CONFIGS` dictionary.
*   **Selecting a Configuration**: You can select a configuration by its name when running training or evaluation scripts using the `--config` or `-c` argument.
*   **Customizing Configurations**: You can modify existing configurations or add new ones directly in `src/configs.py`.
*   **Hyperparameter Tuning**: For a detailed explanation of each parameter, its purpose, and tuning advice, please refer to the [**hyperparameter_guide.md**](./hyperparameter_guide.md) file.

## Training

The main script for training agents is `src/main.py`. Ensure your virtual environment is activated before running any scripts.

**Basic Usage**:
```bash
python src/main.py --config <config_name> --algorithm <sac_or_ppo>
```

**Key Command-Line Arguments for `src/main.py`**:

*   `--config` / `-c`: Name of the configuration to use from `src/configs.py` (e.g., `default_sac_mlp`, `ppo_rnn_mapping`). Default: `default_mapping`.
*   `--algorithm` / `-a`: RL algorithm to use (`sac` or `ppo`). This overrides the algorithm specified in the chosen config file if different.
*   `--device` / `-d`: CUDA device to use (e.g., `cuda:0`, `cuda:1`, `cpu`). If not specified, uses the device from the config, defaulting to `cuda:0` if available, otherwise `cpu`.
*   `--evaluate` / `--no-evaluate`: Whether to run an evaluation phase after training (default is `True`). Use `--no-evaluate` to skip.

**Example Training Commands**:

*   Train a SAC agent with MLP using the `default_sac_mlp` configuration:
    ```bash
    python src/main.py --config default_sac_mlp --algorithm sac
    ```
*   Train a PPO agent with RNN using the `default_ppo_rnn` configuration on `cuda:1` and skip evaluation:
    ```bash
    python src/main.py --config default_ppo_rnn --algorithm ppo --device cuda:1 --no-evaluate
    ```

**Training Output**:
Training results, including saved models, TensorBoard logs, and the effective configuration file (`config.json`), are stored in uniquely named subdirectories within the `experiments/` folder. For example:
`experiments/default_sac_mlp_sac_1678886400/`
    `├── models/`
    `├── tensorboard/`
    `└── config.json`

## Evaluation / Running Experiments

To evaluate a previously trained model, use the dedicated experiment scripts: `src/run_sac_experiment.py` or `src/run_ppo_experiment.py`. Ensure your virtual environment is activated.

**Key Command-Line Arguments for experiment scripts**:

*   `--config` / `-c`: Configuration name used during training (or a compatible one for architecture). This helps set up the agent structure correctly.
*   `--model` / `-m` (Required): Full path to the trained model checkpoint file (`.pt`).
*   `--episodes` / `-e`: Number of evaluation episodes (overrides config).
*   `--steps` / `-s`: Maximum steps per evaluation episode (overrides config).
*   `--render` / `--no-render`: Enable or disable rendering (overrides config).

**Example Evaluation Commands**:

*   Evaluate a trained SAC model:
    ```bash
    python src/run_sac_experiment.py --config default_sac_mlp --model experiments/default_sac_mlp_sac_timestamp/models/sac_final_epXXXX_updatesYYYY.pt --render
    ```
*   Evaluate a trained PPO model for 10 episodes without rendering:
    ```bash
    python src/run_ppo_experiment.py --config default_ppo_rnn --model experiments/default_ppo_rnn_ppo_timestamp/models/ppo_rnn_final_epXXXX_stepYYYY.pt --episodes 10 --no-render
    ```

**Evaluation Output**:
*   Console output will show per-episode metrics (reward, success, final mapping metric).
*   If rendering is enabled, GIF/MP4 files will be saved to the directory specified in `VisualizationConfig.save_dir` (default: `mapping_snapshots/`).

## TensorBoard

TensorBoard is used for logging training progress. Ensure your virtual environment is activated.

1.  Navigate to the root directory of the project in your terminal.
2.  Run TensorBoard, pointing it to the `experiments` directory (or a specific experiment's `tensorboard` subfolder):
    ```bash
    tensorboard --logdir experiments
    ```
3.  Open your web browser and go to `http://localhost:6006` (or the URL provided by TensorBoard).

You can monitor metrics like episode rewards, losses (actor, critic, alpha), entropy, performance metrics, etc.

## Implemented Algorithms

*   **Soft Actor-Critic (SAC)**: An off-policy, maximum entropy actor-critic algorithm known for its sample efficiency and stability. Supports MLP and RNN (LSTM/GRU) variants, Prioritized Experience Replay (PER), and state/reward normalization.
*   **Proximal Policy Optimization (PPO)**: An on-policy actor-critic algorithm known for its robust performance across a wide range of tasks. Supports MLP and RNN (LSTM/GRU) variants, Generalized Advantage Estimation (GAE), and state/reward normalization.

## Key Components

*   **`src/world.py`**: Defines the oil spill environment, including agent dynamics, sensor model, spill generation, and reward calculation. The state representation includes normalized coordinates and heading, and optionally a trajectory of past states, actions, and rewards.
*   **`src/mapper.py`**: Implements the `Mapper` class, which uses sensor readings to estimate the oil spill's convex hull.
*   **`src/visualization.py`**: Handles the rendering of the environment state using Matplotlib, and saving of evaluation episodes as GIFs or MP4s.

## Hyperparameter Tuning

Choosing the right hyperparameters is crucial for successful RL training. A comprehensive guide to all configuration parameters, their effects, and tuning advice can be found in [**hyperparameter_guide.md**](./hyperparameter_guide.md).
