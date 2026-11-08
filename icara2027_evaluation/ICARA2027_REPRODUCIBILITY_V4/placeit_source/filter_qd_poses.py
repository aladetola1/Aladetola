import pdb
import time
import numpy as np
import argparse
import sys
import random
from multiprocessing import Pool

from visualization.vis_tools import load_run_output_files, load_all_inds, get_folder_path_from_str, init_env, \
	load_all_infos, load_all_fitnesses, get_final_6dof_pose_from_infos, get_last_dump_ind_file_qd, get_last_dump_ind_file
from utils.io_run_data import export_running_data

INVALID_POSE_FITNESS = - np.inf

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--runs", help="The directory containing runs", type=str, required=True)
	parser.add_argument("-ll", "--parallelize", action="store_true", help="Trigger multiprocessing parallelization.")
	parser.add_argument("-d", "--debug", action="store_true", help="display grasps and places")
	return parser.parse_args()


def get_kwargs():
	args = arg_parser()
	if args.debug:
		assert not args.parallelize, \
			'Cannot both display and parallelize computation. (choose either -ll or -d).'
		
	kwargs = {
		'run_folder_path': args.runs,
		'parallelize': args.parallelize,
		'debug': args.debug,
	}

	return kwargs

def export_results(is_robust_infos, run_folder_path):

	is_robust_list = is_robust_infos['is_robust_list']
	folder = get_folder_path_from_str(run_folder_path)
	last_ind_file = get_last_dump_ind_file(folder)

	data = np.load(last_ind_file, allow_pickle=True)
	data_dict = {key: data[key] for key in data.files}
	data_dict["is_robust_pose_flag"] = np.array(is_robust_list)

	np.savez(last_ind_file, **data_dict)
	print(f"Fichier mis à jour : {last_ind_file}")

	data_dict["robustness_fitness"]=np.array(is_robust_infos['fitness'])
	np.savez(last_ind_file, **data_dict)
	print(f"Fichier mis à jour : {last_ind_file}")

def get_fitness(obj_trajectory_data):

	if obj_trajectory_data is None :
		return INVALID_POSE_FITNESS

	positions = [pos for pos, _ in obj_trajectory_data]
	orientations = [orient for _, orient in obj_trajectory_data]

	positions = np.array(positions)
	orientations = np.array(orientations)

	# Calcul de la variance
	pos_variance = np.var(positions)
	orient_variance = np.var(orientations)

	fitness = - (pos_variance + orient_variance)

	return fitness

def is_not_robust_output_dict():
	return {
		"is_robust" : False,
		"fitness" : INVALID_POSE_FITNESS
	}

def is_pose_robust_to_perturbation(env, trajectory_data) :
	is_robust = False

	if not trajectory_data :
		return is_not_robust_output_dict()

	is_there_contact = env.is_there_contacts()

	if not is_there_contact :
		return is_not_robust_output_dict()

	fitness = - get_fitness(trajectory_data)

	if fitness > 10 :
		is_robust = False

	return {
		"is_robust" : is_robust,
		"fitness" : fitness
	}

def eval_func(args):
	env, pose_list = args

	object_6dof_final_pose = {
		'xyz': pose_list['6dof_pose'][:3].tolist(),
		'euler_rpy': pose_list['6dof_pose'][3:].tolist(),
		'quaternions': None,
	}
	robustness_per_pose = []

	for _ in range(10):
		env.reset()
		env.set_6dof_obj_pose(object_6dof_final_pose)
		env.apply_external_force_to_object_final_state()
		trajectory_data = env.drop_object()
		
		robustness_per_pose.append(is_pose_robust_to_perturbation(env, trajectory_data))

	return robustness_per_pose


def evaluate(evaluate_fn, args_list, multiproc_pool=None):

	if multiproc_pool is not None:
		evaluation_results = multiproc_pool.map(evaluate_fn,args_list)
	else:
		evaluation_results = list(map(evaluate_fn,args_list))

	is_robust_list = []
	fitness_list = []
	# if one pose have at least one test with a not robust result, the pose is not considered robust
	for result in evaluation_results:  
		# result = liste de 10 dictionnaires
		per_pose_is_robust = [r['is_robust'] for r in result]
		per_pose_fitness   = [r['fitness'] for r in result]

		# une pose est considérée robuste si TOUS ses tests sont robustes
		is_robust = all(per_pose_is_robust)

		is_robust_list.append(is_robust)
		fitness_list.append(per_pose_fitness)  # garde la liste des 10 fitness


	robustness_infos = {
		'is_robust_list' : is_robust_list,
		'fitness' : fitness_list
	}
	return robustness_infos

def filter_poses(env, all_6dof_pose_data, args):

	args_list = [(env, pose) for pose in all_6dof_pose_data.values()]
	if args['parallelize'] :
		with Pool() as multiproc_pool:
			is_robust_infos = evaluate(eval_func, args_list, multiproc_pool)
	else :
		is_robust_infos = evaluate(eval_func, args_list)

	return is_robust_infos

def get_all_6dof_data(folder, cfg):
	
	individuals = load_all_inds(folder)
	infos = load_all_infos(folder)
	fitnesses = load_all_fitnesses(folder)

	all_6dof_poses = get_final_6dof_pose_from_infos(infos)
	all_6dof_pose_data = {}

	for scs_ind_id, (ind, poses) in enumerate(zip(individuals, all_6dof_poses)):
		info = infos[scs_ind_id]

		all_6dof_pose_data[scs_ind_id] = {
			'6dof_pose': poses,
			'fitness': fitnesses[scs_ind_id],
			'infos': info,
		}

	return all_6dof_pose_data


def init_filter_6dof_poses(**args):
	run_folder_path = args['run_folder_path']
	display = args['debug']

	folder = get_folder_path_from_str(run_folder_path)

	run_details, run_infos, cfg = load_run_output_files(folder)

	env = init_env(cfg, display=display)

	all_6dof_pose_data = get_all_6dof_data(folder=folder, cfg=cfg)

	return env, all_6dof_pose_data


def filter_poses_routine(**args):

	env, all_6dof_pose_data = init_filter_6dof_poses(**args)

	is_robust_infos = filter_poses(
		env=env,
		all_6dof_pose_data=all_6dof_pose_data,
		args=args,
	)
	run_folder_path = args['run_folder_path']

	export_results(is_robust_infos, run_folder_path )
	print('Ending.')


def main():
	kwargs = get_kwargs()
	filter_poses_routine(**kwargs)


if __name__ == "__main__":
	sys.exit(main())