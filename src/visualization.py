import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import os
import time
import numpy as np
import math
from world import World # Mapping world
from world_objects import Location
import imageio.v2 as imageio
from PIL import Image
import glob
from configs import VisualizationConfig
from typing import List

# Global trajectory storage - reset using reset_trajectories()
_agent_trajectory: List[tuple[float, float]] = [] # Stores UNNORMALIZED coordinates


def visualize_world(world: World, vis_config: VisualizationConfig, filename: str = None, show_trajectories: bool = True):
    """
    Visualize the oil spill mapping world state (point clouds, hull) and save it to a file.
    Operates on UNNORMALIZED coordinates for plotting.

    Args:
        world (World): The World object to visualize.
        vis_config (VisualizationConfig): Configuration for visualization settings.
        filename (str, optional): Optional filename for saving the plot (without directory).
        show_trajectories (bool): Whether to show the agent trajectory.

    Returns:
        str: Full path to the saved image file, or None if saving failed.
    """
    global _agent_trajectory
    max_trajectory_points = vis_config.max_trajectory_points

    # --- Update Trajectory (using unnormalized coordinates) ---
    if world.agent and world.agent.location:
        # Store UNNORMALIZED location
        _agent_trajectory.append((world.agent.location.x, world.agent.location.y))
        if len(_agent_trajectory) > max_trajectory_points:
            _agent_trajectory = _agent_trajectory[-max_trajectory_points:]

    # --- Prepare Figure ---
    save_dir = vis_config.save_dir
    try:
        if not os.path.exists(save_dir): os.makedirs(save_dir)
    except OSError as e: print(f"Error creating viz dir {save_dir}: {e}"); return None

    fig, ax = plt.subplots(figsize=vis_config.figure_size)

    # --- Plot Trajectory (Unnormalized) ---
    if show_trajectories and len(_agent_trajectory) > 1:
        agent_traj_x, agent_traj_y = zip(*_agent_trajectory)
        ax.plot(agent_traj_x, agent_traj_y, 'g-', # Changed color to green
                linewidth=1.0, alpha=0.6, label='Agent Traj.')

    # --- Plot True Oil Points ---
    if vis_config.plot_oil_points and world.true_oil_points:
        oil_x, oil_y = zip(*[(p.x, p.y) for p in world.true_oil_points])
        ax.scatter(oil_x, oil_y, color='black', marker='.', s=vis_config.point_marker_size, alpha=0.7, label='True Oil')

    # --- Plot True Water Points ---
    if vis_config.plot_water_points and world.true_water_points:
        water_x, water_y = zip(*[(p.x, p.y) for p in world.true_water_points])
        ax.scatter(water_x, water_y, color='lightblue', marker='.', s=vis_config.point_marker_size, alpha=0.5, label='Water')

    # --- Plot Estimated Oil Spill Hull ---
    if world.mapper and world.mapper.hull_vertices is not None:
        # hull_vertices is already ordered for polygon plotting
        hull_poly = Polygon(world.mapper.hull_vertices,
                            edgecolor='red', facecolor='red', alpha=0.2, # Filled transparent red
                            linewidth=1.5, linestyle='--',
                            label=f'Est. Hull (Pts In: {world.performance_metric:.2%})')
        ax.add_patch(hull_poly)
        # Optional: plot oil sensor locations used for hull
        # if world.mapper.oil_sensor_locations:
        #     sensor_x, sensor_y = zip(*[(p.x, p.y) for p in world.mapper.oil_sensor_locations])
        #     ax.scatter(sensor_x, sensor_y, color='darkred', marker='x', s=20, label='Oil Sensors Used')


    # --- Plot Agent (Unnormalized) ---
    if world.agent and world.agent.location:
        ax.scatter(world.agent.location.x, world.agent.location.y,
                   color='blue', marker='o', s=60, zorder=5, label='Agent') # Increased size slightly
        # Add heading indicator
        heading = world.agent.get_heading()
        arrow_len = 3.0 # Length in world units
        ax.arrow(world.agent.location.x, world.agent.location.y,
                 arrow_len * math.cos(heading), arrow_len * math.sin(heading),
                 head_width=1.0, head_length=1.5, fc='blue', ec='blue', alpha=0.7, zorder=5)

    # --- Plot Sensors and Radii (Unnormalized) ---
    if world.agent:
        sensor_locs, sensor_reads = world._get_sensor_readings() # Get current readings (unnormalized locations)
        for i, loc in enumerate(sensor_locs):
            is_oil = sensor_reads[i]
            color = vis_config.sensor_color_oil if is_oil else vis_config.sensor_color_water
            # Plot sensor location
            ax.scatter(loc.x, loc.y,
                       color=color, marker='s', s=vis_config.sensor_marker_size,
                       edgecolors='black', linewidth=0.5, zorder=4, label='Sensors' if i == 0 else "")
            # Plot sensor radius
            sensor_circle = Circle((loc.x, loc.y), world.sensor_radius,
                                   edgecolor=color, facecolor='none',
                                   linewidth=0.5, linestyle=':', alpha=0.4, zorder=3)
            ax.add_patch(sensor_circle)


    # --- Axis Setup ---
    world_w, world_h = world.world_size
    ax.set_xlabel('X Coordinate (Unnormalized)')
    ax.set_ylabel('Y Coordinate (Unnormalized)')
    title_info1 = f"Step: {world.current_step}, Reward: {world.reward:.3f}"
    title_info2 = f"Metric (Pts In): {world.performance_metric:.3f}"
    if world.current_seed is not None: title_info1 += f", Seed: {world.current_seed}"
    ax.set_title(f'Oil Spill Mapping (PointCloud)\n{title_info1} | {title_info2}')

    # Use fixed world boundaries plus padding
    padding = 5.0 # Padding in world units
    ax.set_xlim(-padding, world_w + padding)
    ax.set_ylim(-padding, world_h + padding)

    ax.set_aspect('equal', adjustable='box')
    # Place legend outside plot area
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # --- Save Figure ---
    if filename is None:
        timestamp = int(time.time())
        filename = f"world_map_{timestamp}.png"

    full_path = os.path.join(save_dir, filename)
    try:
        plt.savefig(full_path, bbox_inches='tight') # Use bbox_inches='tight'
        plt.close(fig)
        return full_path
    except Exception as e:
        print(f"Error saving visualization to {full_path}: {e}")
        plt.close(fig)
        return None


def reset_trajectories():
    """Reset stored agent trajectory data."""
    global _agent_trajectory
    _agent_trajectory = []


def save_gif(output_filename: str, vis_config: VisualizationConfig, frame_paths: list, delete_frames: bool = True):
    """
    Create a GIF from a list of frame image paths.
    """
    if not frame_paths:
        print("No frame paths provided to create GIF."); return None

    save_dir = vis_config.save_dir
    duration = vis_config.gif_frame_duration
    output_path = os.path.join(save_dir, output_filename)

    print(f"Creating GIF from {len(frame_paths)} frames: {output_path}")
    try:
        images = []
        valid_frame_paths = []
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                 images.append(imageio.imread(frame_path))
                 valid_frame_paths.append(frame_path) # Only keep track of files actually read
            else: print(f"Warning: Frame file not found during GIF creation: {frame_path}")
        if not images: print("Error: No valid frames found for GIF."); return None

        imageio.mimsave(output_path, images, duration=duration) # Use keyword 'duration'
        print(f"GIF saved successfully.")

        if delete_frames:
            deleted_count = 0
            # Iterate over the frames actually used in the GIF
            for frame_path in valid_frame_paths:
                try:
                    if os.path.exists(frame_path): os.remove(frame_path); deleted_count += 1
                except OSError as e: print(f"Warn: Could not delete frame {frame_path}: {e}")
            if deleted_count > 0: print(f"Deleted {deleted_count} frame files.")
        return output_path
    except Exception as e: print(f"Error creating GIF {output_path}: {e}"); return None