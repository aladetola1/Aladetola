import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pandas as pd
from matplotlib import ticker
import argparse
import sys
import time
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict, Counter
from visualization.vis_tools import load_run_output_files, get_folder_path_from_str, init_env, \
    load_all_infos, load_all_fitnesses, get_final_6dof_pose_from_infos

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--runs", help="The directory containing runs", type=str, required=True)

	return parser.parse_args()


def get_path_kwargs():
	args = arg_parser()

	return {
		'run_folder_path': args.runs,
	}

def pose_list_to_dict(pose_list):

    pose_dicts = []

    for pose in pose_list:
        x, y, z, roll, pitch, yaw = pose

        pose_dicts.append({
            'xyz': [x, y, z],
            'euler_rpy': [roll, pitch, yaw],
            'quaternions': None
        })
    return pose_dicts

def load_data(data_path):
    fitness_list =[]
    final_pose_list = []
    
    folder = get_folder_path_from_str(data_path)
    infos = load_all_infos(folder)
    fitnesses = load_all_fitnesses(folder)

    all_6dof_poses = get_final_6dof_pose_from_infos(infos)

    fitness_list = fitnesses
    final_pose_list = pose_list_to_dict(all_6dof_poses)

    min_val = min(fitness_list)
    shift_value = - min_val if min_val < 0 else 0
    fitness_list = [val + shift_value for val in fitness_list]

    return fitness_list, final_pose_list


def get_point_cloud(env) :

    transformed_pc, T_world_obj = env.get_obj_mesh_vertice_points_in_world_frame()
    
    if env.is_debug_mode():
        time.sleep(1)
        env.delete_debug_bodies()

    z = transformed_pc[:, 2]
    z_min, z_max = np.min(z), np.max(z)
    z_range = z_max - z_min

    # Seuil absolu : 10% du range au-dessus du z_min
    z_threshold = z_min + 0.1 * z_range

    # Sélectionne les points dont la hauteur est ≤ seuil
    selected_points = transformed_pc[z <= z_threshold]

    selected_points_h = np.hstack((selected_points, np.ones((selected_points.shape[0], 1))))  # Nx4
    selected_points_obj_h = (T_world_obj @ selected_points_h.T).T  # Nx4
    selected_points_obj = selected_points_obj_h[:, :3]

    return selected_points_obj

def point_cloud_values(env, fitness_list, final_pose_list, precision=6):

    point_fitness_dict = {}
    pc_obj_frame  = env.get_obj_mesh_vertice_points_in_obj_frame()
    triangles = env.get_obj_point_cloud_in_obj_frame()


    for pt in  pc_obj_frame :
        key = tuple(np.round(pt, precision))
        point_fitness_dict[key] = {
            'coord': pt,
            'fitness': 0,
            'count': 0
        }

    face_counter = Counter()
    face_pose_indices = defaultdict(list)
    face_keys = [] 
    similarity_threshold = 0.9

    for i, pose in enumerate(final_pose_list):

        replay_final_6dof_poses(env, pose)
        pc = get_point_cloud(env)

        if pc is None or len(pc) == 0:
            continue

        rounded_pc = set(tuple(np.round(pt, precision)) for pt in pc)

        matched = False
        for ref_key in face_keys:
            common = len(rounded_pc.intersection(ref_key))
            overlap_ratio = common / max(len(ref_key), len(rounded_pc))

            if overlap_ratio >= similarity_threshold:
                face_counter[ref_key] += 1
                face_pose_indices[ref_key].append(i)
                matched = True
                break

        if not matched:
            new_key = frozenset(rounded_pc)
            face_keys.append(new_key)
            face_counter[new_key] += 1
            face_pose_indices[new_key].append(i)

        for pt in pc:
            # Arrondi pour gérer les flottants et regrouper les points similaires
            key = tuple(np.round(pt, precision))

            # Accumulation de la fitness
            if key in point_fitness_dict:
                point_fitness_dict[key]['fitness'] += fitness_list[i]
                point_fitness_dict[key]['count'] += 1

    # convert to arry     
    points = []
    fitness = []
    count = []

    for entry in point_fitness_dict.values():
        points.append(entry['coord'])
        fitness.append(entry['fitness'])
        count.append(entry['count'])

    total_counts = sum(face_counter.values())  
    print(f"Nombre de faces uniques rencontrées : {len(face_counter)}")
    print(f"Nombre total de faces vues : {total_counts}")

    for face, count in face_counter.items():
        percentage = (count / total_counts) * 100
        print(f"- Face vue {count} fois({percentage:.2f}%)")

    x_face = face_counter.most_common()[0]      #la xeme face plus fréquente 
    keys, count = x_face

    # for idx in face_pose_indices[keys]:
    #     pose = final_pose_list[idx]
    #     replay_final_6dof_poses(env_d, pose)
    #     time.sleep(1)
    
    return np.array(points), triangles, np.array(fitness), np.array(count)       
          
def replay_final_6dof_poses(env, pose) :     
    env.reset()
    env.set_6dof_obj_pose(pose)
        

def display_3d_heatmap_1(points,triangles, fitness, title='', export_path=None) :
    vertices = points
    mesh_triangles = [vertices[triangle] for triangle in triangles]

    # Calcul de la fitness par triangle (moyenne des 3 vertices)
    triangle_fitness = np.mean(fitness[triangles], axis=1)
    #min_norm_value = np.min(triangle_fitness)
    min_norm_value = 0

    norm_fitness = (triangle_fitness - min_norm_value ) / (np.max(triangle_fitness) - min_norm_value )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Choose colormap: e.g., 'viridis', 'plasma', 'hot', etc.
    cmap = plt.cm.viridis
    colors = cmap(norm_fitness)

    mesh_collection = Poly3DCollection(mesh_triangles, facecolors=colors, edgecolor='k', linewidths=0.01, alpha=0.8)
    ax.add_collection3d(mesh_collection)
    # Auto scale to the mesh size
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    #sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=5)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(fitness)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
    cbar.set_label("Normalized Fitness")

    if export_path is not None:
        export_path = export_path + '/' if export_path[-1] != '/' else export_path
        clean_title = export_path + title.replace(' ', '_').replace('|', '_')

        # 1. Export as PNG image
        fig_path = clean_title + '.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else :
        plt.show()
    plt.close()

def display_3d_heatmap(points, triangles, fitness, title='', export_path=None, gamma=0.3):
    """
    Affiche un heatmap 3D des triangles d'un mesh avec coloration selon la fitness.
    La normalisation commence à 0 et les petites valeurs sont accentuées pour être plus visibles.
    
    Args:
        points: np.array(N,3) des vertices
        triangles: np.array(M,3) indices des triangles
        fitness: np.array(N) fitness par vertex
        gamma: float, facteur de contraste pour les petites valeurs (<1 accentue les faibles valeurs)
    """
    vertices = points
    mesh_triangles = [vertices[triangle] for triangle in triangles]

    # Normalisation de la fitness de 0 à 1
    fitness = np.array(fitness)
    max_val = np.max(fitness)
    norm_fitness = fitness / (max_val + 1e-8)

    # Accentuer les petites valeurs (gamma correction)
    norm_fitness = norm_fitness ** gamma

    # Calcul de la fitness par triangle (moyenne des 3 vertices)
    triangle_fitness = np.min(norm_fitness[triangles], axis=1)

    # Création du plot 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.cm.viridis
    colors = cmap(triangle_fitness)

    mesh_collection = Poly3DCollection(mesh_triangles, facecolors=colors, edgecolor='k', linewidths=0.01, alpha=0.9)
    ax.add_collection3d(mesh_collection)

    # Ajustement automatique des axes
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Labels et titre simples
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)

    # Moins de ticks sur X et Y
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))

    # Colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(triangle_fitness)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
    cbar.set_label("Normalized Fitness", fontsize=12)

    # Export si nécessaire
    if export_path is not None:
        export_path = export_path.rstrip('/') + '/'
        clean_title = export_path + title.replace(' ', '_').replace('|', '_')
        fig_path = clean_title + '.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"{fig_path} has been successfully exported.")

    plt.show()
    plt.close()


def heat_map_routine(run_folder_path):

    # Export path
    output_path = os.path.join(run_folder_path, "heat_map")
    os.makedirs(output_path, exist_ok=True)

    run_folder = Path(run_folder_path)
    run_details, run_infos, cfg = load_run_output_files(run_folder)
    env = init_env(cfg, display=False)

    fitness_list, final_pose_list = load_data(run_folder_path)

    points, triangles, fitness, counts = point_cloud_values(env, fitness_list, final_pose_list)

    display_3d_heatmap(points, triangles, fitness, title = "Fitness Heatmap", export_path=output_path)

    plt.close('all')

def main():

    kwargs = get_path_kwargs()
    heat_map_routine(**kwargs)

if __name__ == "__main__":
    sys.exit(main())
