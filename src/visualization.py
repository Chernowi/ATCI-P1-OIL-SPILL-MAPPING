# In visualization.py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import os
import time
import numpy as np
import math
from world import World # Mapping world
from world_objects import Location
import imageio.v2 as imageio # Keep for GIF
from PIL import Image 
import glob
from configs import VisualizationConfig
from typing import List

_agent_trajectory: List[tuple[float, float]] = []

def visualize_world(world: World, vis_config: VisualizationConfig, fig, ax, show_trajectories: bool = True):
    """
    Visualize the oil spill mapping world state on the given fig and ax.
    Operates on UNNORMALIZED coordinates for plotting.
    Does NOT save the figure.
    """
    global _agent_trajectory
    max_trajectory_points = vis_config.max_trajectory_points

    if world.agent and world.agent.location:
        _agent_trajectory.append((world.agent.location.x, world.agent.location.y))
        if len(_agent_trajectory) > max_trajectory_points:
            _agent_trajectory = _agent_trajectory[-max_trajectory_points:]

    ax.clear() # Clear previous frame's artists

    if show_trajectories and len(_agent_trajectory) > 1:
        agent_traj_x, agent_traj_y = zip(*_agent_trajectory)
        ax.plot(agent_traj_x, agent_traj_y, 'g-',
                linewidth=1.0, alpha=0.6, label='Agent Traj.')

    if vis_config.plot_oil_points and world.true_oil_points:
        oil_x, oil_y = zip(*[(p.x, p.y) for p in world.true_oil_points])
        ax.scatter(oil_x, oil_y, color='black', marker='.', s=vis_config.point_marker_size, alpha=0.7, label='True Oil')

    if vis_config.plot_water_points and world.true_water_points:
        water_x, water_y = zip(*[(p.x, p.y) for p in world.true_water_points])
        ax.scatter(water_x, water_y, color='lightblue', marker='.', s=vis_config.point_marker_size, alpha=0.5, label='Water')

    if world.mapper and world.mapper.hull_vertices is not None:
        hull_poly = Polygon(world.mapper.hull_vertices,
                            edgecolor='red', facecolor='red', alpha=0.2,
                            linewidth=1.5, linestyle='--',
                            label=f'Est. Hull (Pts In: {world.performance_metric:.2%})')
        ax.add_patch(hull_poly)

    if world.agent and world.agent.location:
        ax.scatter(world.agent.location.x, world.agent.location.y,
                   color='blue', marker='o', s=60, zorder=5, label='Agent')
        heading = world.agent.get_heading()
        arrow_len = 3.0
        ax.arrow(world.agent.location.x, world.agent.location.y,
                 arrow_len * math.cos(heading), arrow_len * math.sin(heading),
                 head_width=1.0, head_length=1.5, fc='blue', ec='blue', alpha=0.7, zorder=5)

    if world.agent:
        sensor_locs, sensor_reads = world._get_sensor_readings()
        for i, loc in enumerate(sensor_locs):
            is_oil = sensor_reads[i]
            color = vis_config.sensor_color_oil if is_oil else vis_config.sensor_color_water
            ax.scatter(loc.x, loc.y,
                       color=color, marker='s', s=vis_config.sensor_marker_size,
                       edgecolors='black', linewidth=0.5, zorder=4, label='Sensors' if i == 0 else "")
            sensor_circle = Circle((loc.x, loc.y), world.sensor_radius,
                                   edgecolor=color, facecolor='none',
                                   linewidth=0.5, linestyle=':', alpha=0.4, zorder=3)
            ax.add_patch(sensor_circle)

    world_w, world_h = world.world_size
    ax.set_xlabel('X Coordinate (Unnormalized)')
    ax.set_ylabel('Y Coordinate (Unnormalized)')
    title_info1 = f"Step: {world.current_step}, Reward: {world.reward:.3f}"
    title_info2 = f"Metric (Pts In): {world.performance_metric:.3f}"
    if world.current_seed is not None: title_info1 += f", Seed: {world.current_seed}"
    ax.set_title(f'Oil Spill Mapping\n{title_info1} | {title_info2}')

    padding = 5.0
    ax.set_xlim(-padding, world_w + padding)
    ax.set_ylim(-padding, world_h + padding)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout(rect=[0, 0, 0.85, 1]) # Apply to figure

def reset_trajectories():
    global _agent_trajectory
    _agent_trajectory = []

def save_gif(output_filename: str, vis_config: VisualizationConfig, frame_paths: list):
    if not frame_paths:
        print("No frame paths provided to create GIF."); return None

    save_dir = vis_config.save_dir
    # Use video_fps for consistency, calculate duration for imageio
    frame_duration_sec = 1.0 / vis_config.video_fps
    output_path = os.path.join(save_dir, output_filename)

    print(f"Creating GIF from {len(frame_paths)} frames: {output_path} (Duration per frame: {frame_duration_sec:.3f}s)")
    try:
        images = []
        valid_frame_paths = []
        target_size = None

        for i, frame_path in enumerate(frame_paths):
            if os.path.exists(frame_path):
                try:
                    img_pil = Image.open(frame_path).convert('RGB')
                    if target_size is None:
                        target_size = img_pil.size
                    if img_pil.size != target_size:
                        img_pil = img_pil.resize(target_size, Image.Resampling.LANCZOS)
                    images.append(np.array(img_pil))
                    valid_frame_paths.append(frame_path)
                except Exception as read_err:
                     print(f"Error reading or processing frame {frame_path}: {read_err}")
            else: print(f"Warning: Frame file not found during GIF creation: {frame_path}")

        if not images: print("Error: No valid frames found/processed for GIF."); return None
        
        first_shape = images[0].shape
        for idx, img_arr in enumerate(images):
            if img_arr.shape != first_shape:
                print(f"CRITICAL ERROR: Frame {idx} shape {img_arr.shape} mismatch with first frame {first_shape}. Aborting GIF save.")
                return None

        imageio.mimsave(output_path, images, duration=frame_duration_sec * 1000) # imageio duration is in ms or list of durations
        print(f"GIF saved successfully.")

        if vis_config.delete_png_frames:
            deleted_count = 0
            for frame_path in valid_frame_paths:
                try:
                    if os.path.exists(frame_path): os.remove(frame_path); deleted_count += 1
                except OSError as e: print(f"Warn: Could not delete frame {frame_path}: {e}")
            if deleted_count > 0: print(f"Deleted {deleted_count} PNG frame files.")
        return output_path
    except Exception as e:
        print(f"Error creating GIF {output_path}: {e}")
        if 'images' in locals() and images:
             shapes = [img.shape for img in images]
             print(f"Frame shapes before error: {shapes}")
        return None