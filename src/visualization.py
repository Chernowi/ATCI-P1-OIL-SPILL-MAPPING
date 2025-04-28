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
_agent_trajectory: List[tuple[float, float]] = []


def visualize_world(world: World, vis_config: VisualizationConfig, filename: str = None, show_trajectories: bool = True):
    """
    Visualize the oil spill mapping world state and save it to a file.

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

    # --- Update Trajectory ---
    if world.agent and world.agent.location:
        _agent_trajectory.append((world.agent.location.x, world.agent.location.y))
        if len(_agent_trajectory) > max_trajectory_points:
            _agent_trajectory = _agent_trajectory[-max_trajectory_points:]

    # --- Prepare Figure ---
    save_dir = vis_config.save_dir
    try:
        if not os.path.exists(save_dir): os.makedirs(save_dir)
    except OSError as e: print(f"Error creating viz dir {save_dir}: {e}"); return None

    fig, ax = plt.subplots(figsize=vis_config.figure_size)

    # --- Plot Trajectory ---
    if show_trajectories and len(_agent_trajectory) > 1:
        agent_traj_x, agent_traj_y = zip(*_agent_trajectory)
        ax.plot(agent_traj_x, agent_traj_y, 'b-',
                linewidth=1.0, alpha=0.5, label='Agent Traj.')

    # --- Plot True Oil Spill ---
    if world.true_spill:
        true_spill_patch = Circle((world.true_spill.center.x, world.true_spill.center.y),
                                  world.true_spill.radius,
                                  color='gray', alpha=0.4, label=f'True Spill (R={world.true_spill.radius:.1f})')
        ax.add_patch(true_spill_patch)

    # --- Plot Estimated Oil Spill ---
    if world.mapper and world.mapper.estimated_spill:
        est_spill = world.mapper.estimated_spill
        est_spill_patch = Circle((est_spill.center.x, est_spill.center.y),
                                 est_spill.radius,
                                 edgecolor='red', facecolor='none', linewidth=1.5, linestyle='--',
                                 label=f'Est. Spill (R={est_spill.radius:.1f}, IoU={world.iou:.3f})')
        ax.add_patch(est_spill_patch)
        # Plot center of estimated spill
        ax.scatter(est_spill.center.x, est_spill.center.y, color='red', marker='x', s=50, label='Est. Center')


    # --- Plot Agent ---
    if world.agent and world.agent.location:
        ax.scatter(world.agent.location.x, world.agent.location.y,
                   color='blue', marker='o', s=80, label='Agent')
        # Add heading indicator
        heading = world.agent.get_heading()
        ax.arrow(world.agent.location.x, world.agent.location.y,
                 2.0 * math.cos(heading), 2.0 * math.sin(heading), # Arrow length 2.0 units
                 head_width=0.8, head_length=1.0, fc='blue', ec='blue', alpha=0.7)

    # --- Plot Sensors ---
    if world.agent and world.true_spill:
        sensor_locs, sensor_reads = world._get_sensor_readings() # Get current readings
        for i, loc in enumerate(sensor_locs):
            is_oil = sensor_reads[i]
            color = vis_config.sensor_color_oil if is_oil else vis_config.sensor_color_water
            edge_color = 'black' if is_oil else 'blue'
            ax.scatter(loc.x, loc.y,
                       color=color, marker='s', s=vis_config.sensor_marker_size,
                       edgecolors=edge_color, linewidth=0.5, label='Sensors' if i == 0 else "")

    # --- Plot Mapper Points (Optional - can be slow if many points) ---
    # Uncomment if needed for debugging, but might clutter the plot
    # if world.mapper:
    #     if world.mapper.oil_points:
    #         oil_x, oil_y = zip(*[(p.x, p.y) for p in world.mapper.oil_points])
    #         ax.scatter(oil_x, oil_y, color='black', marker='.', s=5, alpha=0.6, label='Oil Hits')
    #     if world.mapper.water_points:
    #         water_x, water_y = zip(*[(p.x, p.y) for p in world.mapper.water_points])
    #         ax.scatter(water_x, water_y, color='cyan', marker='.', s=5, alpha=0.6, label='Water Hits')

    # --- Axis Setup ---
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    title_info = f"Step: {world.current_step}, Reward: {world.reward:.3f}, IoU: {world.iou:.3f}"
    ax.set_title(f'Oil Spill Mapping\n{title_info}')

    # Determine plot bounds dynamically
    points_x = [world.agent.location.x] if world.agent else []
    points_y = [world.agent.location.y] if world.agent else []
    if world.true_spill:
        points_x.extend([world.true_spill.center.x - world.true_spill.radius, world.true_spill.center.x + world.true_spill.radius])
        points_y.extend([world.true_spill.center.y - world.true_spill.radius, world.true_spill.center.y + world.true_spill.radius])
    if world.mapper and world.mapper.estimated_spill:
        est = world.mapper.estimated_spill
        points_x.extend([est.center.x - est.radius, est.center.x + est.radius])
        points_y.extend([est.center.y - est.radius, est.center.y + est.radius])
    if show_trajectories and _agent_trajectory:
        traj_x, traj_y = zip(*_agent_trajectory)
        points_x.extend(traj_x)
        points_y.extend(traj_y)

    if not points_x or not points_y:
        min_x, max_x, min_y, max_y = -50, 50, -50, 50 # Default bounds
    else:
        min_x, max_x = min(points_x), max(points_x)
        min_y, max_y = min(points_y), max(points_y)

    # Add padding
    range_x = max(max_x - min_x, 10.0) # Ensure minimum range
    range_y = max(max_y - min_y, 10.0)
    padding_x = range_x * 0.15
    padding_y = range_y * 0.15
    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, max_y + padding_y)

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
        plt.savefig(full_path)
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
    Create a GIF from a list of frame image paths. (Unchanged from original)
    """
    if not frame_paths:
        print("No frame paths provided to create GIF."); return None

    save_dir = vis_config.save_dir
    duration = vis_config.gif_frame_duration
    output_path = os.path.join(save_dir, output_filename)

    print(f"Creating GIF from {len(frame_paths)} frames: {output_path}")
    try:
        images = []
        for frame_path in frame_paths:
            if os.path.exists(frame_path): images.append(imageio.imread(frame_path))
            else: print(f"Warning: Frame file not found: {frame_path}")
        if not images: print("Error: No valid frames found."); return None

        imageio.mimsave(output_path, images, duration=duration)
        print(f"GIF saved successfully.")

        if delete_frames:
            deleted_count = 0
            for frame_path in frame_paths:
                try:
                    if os.path.exists(frame_path): os.remove(frame_path); deleted_count += 1
                except OSError as e: print(f"Warn: Could not delete frame {frame_path}: {e}")
            if deleted_count > 0: print(f"Deleted {deleted_count} frame files.")
        return output_path
    except Exception as e: print(f"Error creating GIF {output_path}: {e}"); return None

