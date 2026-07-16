import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import argparse
import sys
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from visualization.vis_tools import load_run_output_files, get_folder_path_from_str, init_env, \
	load_all_infos, load_all_fitnesses, get_final_6dof_pose_from_infos, load_all_flags
from multiprocessing import Pool
from scipy.spatial import KDTree
import time
def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--runs", help="The directory containing runs", type=str, required=True)
	parser.add_argument("-ll", "--parallelize", action="store_true", help="Trigger multiprocessing parallelization.")
	parser.add_argument("-rp", "--robust-poses", action="store_true",
						help="Display only robust poses.")
	return parser.parse_args()


def get_path_kwargs():
	args = arg_parser()

	return {
		'run_folder_path': args.runs,
		'parallelize': args.parallelize,
		'robust_poses': args.robust_poses,
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

def load_data(args):
	fitness_list =[]
	final_pose_list = []
	
	folder = get_folder_path_from_str(args['run_folder_path'])
	infos = load_all_infos(folder)
	fitnesses = load_all_fitnesses(folder)

	all_6dof_poses = get_final_6dof_pose_from_infos(infos)
	if args.get('robust_poses', False) :
		is_robust_pose_flag = load_all_flags(folder)
		for ind, pose in enumerate(all_6dof_poses):
			if not is_robust_pose_flag[ind]:
				continue  
			fitness_list.append(fitnesses[ind])
			final_pose_list.append(pose_list_to_dict([pose])[0])
	else :
		fitness_list = fitnesses
		final_pose_list = pose_list_to_dict(all_6dof_poses)

	min_val = min(fitness_list)
	shift_value = - min_val if min_val < 0 else 0
	fitness_list = [val + shift_value for val in fitness_list]

	return fitness_list, final_pose_list


def smooth_counts(points, counts, radius=0.01):
	tree = KDTree(points)
	smoothed = np.zeros_like(counts, dtype=float)

	for i, pt in enumerate(points):
		idx = tree.query_ball_point(pt, r=radius)
		smoothed[i] = np.mean(counts[idx])  
	return smoothed
	  
def display_3d_heatmap(env, points, counts, title='', export_path=None) :
	triangles = env.get_obj_point_cloud_in_obj_frame()

	counts = smooth_counts(points, counts)

	vertices = points
	mesh_triangles = [vertices[triangle] for triangle in triangles]

	triangle_counts = np.sum(counts[triangles], axis=1)
	
	# Normalisation entre 0 et 1
	if np.max(triangle_counts) > 0:
		norm_counts = triangle_counts / np.max(triangle_counts)
	else:
		norm_counts = np.zeros_like(triangle_counts)
	gamma = 0.8
	norm_counts = norm_counts ** gamma

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	cmap = plt.cm.viridis
	colors = cmap(norm_counts)
	
	mesh_collection = Poly3DCollection(mesh_triangles, facecolors=colors, edgecolor='none', linewidths=0.01, alpha=0.7)
	ax.add_collection3d(mesh_collection)
	# Auto scale to the mesh size
	scale = vertices.flatten()
	ax.auto_scale_xyz(scale, scale, scale)

	#sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=5)
	
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")

	mappable = plt.cm.ScalarMappable(cmap=cmap)
	mappable.set_array(triangle_counts)
	cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
	cbar.set_label("Fitness")

	if export_path is not None:
		export_path = export_path + '/' if export_path[-1] != '/' else export_path
		clean_title = export_path + title.replace(' ', '_').replace('|', '_')

		views = [(30, 45), (30, 135), (60, 225), (90, 0), (-70, 0) ]  

		for i, (elev, azim) in enumerate(views):
			ax.view_init(elev=elev, azim=azim)
			fig_path = f"{clean_title}_view{i}.png"
			plt.savefig(fig_path, dpi=300)
			print(f"{fig_path} has been successfully exported.")
	else :
		plt.show()

	plt.close()

def replay_final_6dof_poses(env, pose) :     
	env.reset()
	env.set_6dof_obj_pose(pose)

def get_point_cloud(env) :
	contact_points = env.get_contact_points()

	# Extraire les points de contact sur l'objet
	obj_contact_points = []
	for contact in contact_points:
		# Vérifier si le contact concerne l'objet
		if contact[1] == env.obj_id or contact[2] == env.obj_id:
			# Utiliser le point sur l'objet
			if contact[1] == env.obj_id:
				point = contact[5]  # Position sur le premier corps (objet)
			else:
				point = contact[6]  # Position sur le second corps
			obj_contact_points.append(point)
	
	if not obj_contact_points:
		return np.array([])
	
	transformed_pc, T_world_obj = env.get_obj_mesh_vertice_points_in_world_frame()

	threshold = 0.02

	selected_points = []

	# Utiliser KDTree pour une recherche plus efficace
	tree = KDTree(transformed_pc)
	
	for contact in obj_contact_points:
		# Trouver les points les plus proches
		distances, indices = tree.query(contact, k=5, distance_upper_bound=threshold)
		valid_indices = indices[distances != np.inf]
		if len(valid_indices) > 0:
			selected_points.extend(transformed_pc[valid_indices])
	
	if not selected_points:
		return np.array([])

	selected_points_np = np.vstack(selected_points)
	selected_points_h = np.hstack((selected_points_np, np.ones((selected_points_np.shape[0], 1))))
	selected_points_obj_h = (T_world_obj @ selected_points_h.T).T 
	selected_points_obj = selected_points_obj_h[:, :3]

	return selected_points_obj

#  EVALUATION WITH FITNESS : VARIANCE
# def point_cloud_evaluation(args):
#     env, fitness, final_pose = args
#     precision=6

#     point_fitness_dict = {}
#     pc_obj_frame  = env.get_obj_mesh_vertice_points_in_obj_frame()


#     for pt in  pc_obj_frame :
#         key = tuple(np.round(pt, precision))
#         point_fitness_dict[key] = {
#             'coord': pt,
#             'fitness': 0,
#             'count': 0
#         }

#     replay_final_6dof_poses(env, final_pose)

#     if not env.is_there_contacts() :
#         return {}

#     pc = get_point_cloud(env)

#     if pc is None or len(pc) == 0:
#         return {}

#     for pt in pc:
#         key = tuple(np.round(pt, precision))

#         if key in point_fitness_dict:
#             point_fitness_dict[key]['fitness'] += fitness
#             point_fitness_dict[key]['count'] += 1

#     points = []
#     fitness = []
#     count = []

#     for entry in point_fitness_dict.values():
#         points.append(entry['coord'])
#         fitness.append(entry['fitness'])
#         count.append(entry['count'])

#     return {
#         'points': np.array(points),
#         'fitness': np.array(fitness),
#     }

def point_cloud_evaluation(args):
	env, fitness, final_pose = args
	precision=6

	all_contact_points = []
	
	replay_final_6dof_poses(env, final_pose)
	env.visualize_contacts_with_normals()
	#time.sleep(1)
	if env.is_there_contacts():
		pc = get_point_cloud(env)
		if pc is not None and len(pc) > 0:
			all_contact_points.append(pc)

	# all_contact_points est une liste de tableaux numpy, on doit les concaténer
	if len(all_contact_points) > 0:
		combined_pc = np.vstack(all_contact_points)
	else:
		combined_pc = np.array([])

	# Initialiser le dictionnaire avec tous les points du mesh
	pc_obj_frame = env.get_obj_mesh_vertice_points_in_obj_frame()
	point_count_dict = {}
	
	for pt in pc_obj_frame:
		key = tuple(np.round(pt, precision))
		point_count_dict[key] = {
			'coord': pt,
			'count': 0
		}
	
	# Compter les occurrences pour chaque point
	for pt in combined_pc:
		key = tuple(np.round(pt, precision))
		if key in point_count_dict:
			point_count_dict[key]['count'] += 1
	
	# Préparer les données de retour
	points = []
	counts = []
	
	for entry in point_count_dict.values():
		points.append(entry['coord'])
		counts.append(entry['count'])

	return {
		'points': np.array(points),
		'counts': np.array(counts),
	}

def evaluate(evaluate_fn, args_list, multiproc_pool=None):

	if multiproc_pool is not None:
		evaluation_results = multiproc_pool.map(evaluate_fn, args_list)
	else:
		evaluation_results = list(map(evaluate_fn,args_list))

	merged_dict = {}

	for result in evaluation_results:
		if not result:
			continue
		points = result['points']
		counts = result['counts']
		for pt, c in zip(points, counts):
			key = tuple(np.round(pt, 6))  # même précision qu'avant
			if key not in merged_dict:
				merged_dict[key] = {'coord': pt, 'count': 0}
			merged_dict[key]['count'] += c

	for result in evaluation_results:
		if not result:
			continue

	merged_points = [v['coord'] for v in merged_dict.values()]
	merged_counts = [v['count'] for v in merged_dict.values()]

	return (
		np.array(merged_points),
		np.array(merged_counts),
	)
	 
def heat_map_routine(**kwargs):
	run_folder_path = kwargs['run_folder_path']
	# Export path
	output_path = os.path.join(run_folder_path, "heat_map")
	os.makedirs(output_path, exist_ok=True)

	run_folder = Path(run_folder_path)
	run_details, run_infos, cfg = load_run_output_files(run_folder)
	if kwargs['parallelize'] : 
		display = False
	else : 
		display = True

	env = init_env(cfg, display)

	fitness_list, final_pose_list = load_data(kwargs)

	args_list = [(env, fitness, final_pose) for fitness, final_pose in zip(fitness_list, final_pose_list)]

	if kwargs['parallelize'] :
		with Pool() as multiproc_pool:
			point_cloud_results = evaluate(point_cloud_evaluation, args_list, multiproc_pool)
	else :
		point_cloud_results = evaluate(point_cloud_evaluation, args_list)
	if kwargs['robust_poses'] :
		title = "Heatmap_robust_poses"
	else :
		title = "Heatmap_all_poses"
	points, counts = point_cloud_results    
	display_3d_heatmap(env, points, counts, title = title, export_path=output_path)

	plt.close('all')

def main():

	kwargs = get_path_kwargs()
	heat_map_routine(**kwargs)

if __name__ == "__main__":
	sys.exit(main())
