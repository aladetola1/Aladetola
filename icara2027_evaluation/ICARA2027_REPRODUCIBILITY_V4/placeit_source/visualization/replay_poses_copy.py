import pdb
import time
import numpy as np
import argparse
import sys
import random
import pandas as pd
from visualization.vis_tools import load_run_output_files, load_all_inds, get_folder_path_from_str, init_env, \
	load_all_infos, load_all_robustness_infos, get_all_6dof_poses_from_infos, load_all_flags, get_final_6dof_pose_from_infos



def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--runs", help="The directory containing runs", type=str, required=True)
	parser.add_argument("-si", "--shuffle-inds", action="store_true", help="Randomly shuffle inds.")
	parser.add_argument("-i", "--i_ind", help="Index of a specific ind to display.", type=int, default=None)
	parser.add_argument("-fs", "--fitness-sorted", action="store_true",
						help="Display poses from higher to lower fitness.")
	parser.add_argument("-rp", "--robust-poses", action="store_true",
						help="Display only robust poses.")
	parser.add_argument("-f", "--final-poses", action="store_true",
						help="Display only final poses.")
	
	return parser.parse_args()


def get_replay_poses_kwargs():
	args = arg_parser()

	return {
		'run_folder_path': args.runs,
		'display_flags': {
			'i_ind2display': args.i_ind,
			'shuffle_inds': args.shuffle_inds,
			'fitness_sorted': args.fitness_sorted,
			'robust_poses': args.robust_poses,
			'final_poses' : args.final_poses,
		}
	}

def play_pose(env, pose_dict, display_flags) :
	env.reset()
	env.set_6dof_obj_pose(pose_dict)

	if not display_flags['final_poses'] :
		td = env.drop_object()
	
	else :
		time.sleep(1)

def replay_6dof_specific_pose(env, all_6dof_pose_data, display_flags):

	i_ind2display = display_flags['i_ind2display']

	
	all_ids = list(all_6dof_pose_data.keys())
	all_fits = [all_6dof_pose_data[ind_id]['fitness'] for ind_id in all_ids]
	ind_ids_sorted_ascending_order = np.argsort(all_fits)[::-1]
	all_all_6dof_pose_ids = [all_ids[i] for i in ind_ids_sorted_ascending_order]
	
	
	ind_id = all_all_6dof_pose_ids[i_ind2display]
	
	obj_6dof_pose = all_6dof_pose_data[ind_id]['6dof_pose']
	fitness = all_6dof_pose_data[ind_id]['fitness']
	print(fitness)

	print(f'Displaying object placement n°{i_ind2display} | fitness={fitness}')

	pose_dict = {
		'xyz': obj_6dof_pose[:3].tolist(),
		'euler_rpy': obj_6dof_pose[3:].tolist(),
		'quaternions': None
	}

	print(pose_dict)

	n_replay = 20

	for i_replay in range(n_replay):
		play_pose(env, pose_dict, display_flags)

	print(f'Display over.')


def replay_all_6dof_poses(env, all_6dof_pose_data, display_flags):

	all_ids = list(all_6dof_pose_data.keys())
	all_fits = [float(np.mean(all_6dof_pose_data[ind_id]['fitness'])) for ind_id in all_ids]



	df = pd.DataFrame({
		"id": all_ids,
		"fitness": all_fits
	})

	intervals = [(0, 0.05), (1.05, 1.1), (2.35, 2.4)]
	selected_ids = []

	for low, high in intervals:
		candidates = df[(df["fitness"] >= low) & (df["fitness"] <= high)]["id"].tolist()
		if len(candidates) >= 10:
			selected_ids.extend(random.sample(candidates, 10))
		else:
			selected_ids.extend(candidates)

	
	if len(selected_ids) < 30:
		remaining = list(set(all_ids) - set(selected_ids))
		selected_ids.extend(random.sample(remaining, min(6 - len(selected_ids), len(remaining))))


	for scs_ind_id in selected_ids:
		obj_6dof_pose = all_6dof_pose_data[scs_ind_id]['6dof_pose']
		fitness = all_6dof_pose_data[scs_ind_id]['fitness']
		pose_dict = {
			'xyz': obj_6dof_pose[:3].tolist(),
			'euler_rpy': obj_6dof_pose[3:].tolist(),
			'quaternions': None,
			'fitness': fitness
		}
		print(pose_dict)
		play_pose(env, pose_dict, display_flags)

	print(f'Display over.')



def replay_6dof_poses(env, all_6dof_pose_data, display_flags):
	if display_flags['i_ind2display'] is not None:
		replay_6dof_specific_pose(
			env=env,
			all_6dof_pose_data=all_6dof_pose_data,
			display_flags=display_flags
		)
	else:
		replay_all_6dof_poses(
			env=env,
			all_6dof_pose_data=all_6dof_pose_data,
			display_flags=display_flags
		)


def get_all_6dof_data(folder, cfg, display_flags):
	
	individuals = load_all_inds(folder)
	infos = load_all_infos(folder)
	fitnesses = load_all_robustness_infos(folder)

	if display_flags['final_poses'] :
			all_6dof_poses = get_final_6dof_pose_from_infos(infos)
	else :
		all_6dof_poses = get_all_6dof_poses_from_infos(infos)

	all_6dof_pose_data = {}

	if display_flags['robust_poses'] :
		is_robust_pose_flag = load_all_flags(folder)
		for scs_ind_id, (ind, poses) in enumerate(zip(individuals, all_6dof_poses)):
			if not is_robust_pose_flag[scs_ind_id]:
				continue  

			all_6dof_pose_data[scs_ind_id] = {
				'6dof_pose': poses,
				'fitness': fitnesses[scs_ind_id],
				'infos': infos[scs_ind_id],
			}
	else : 
		for scs_ind_id, (ind, poses) in enumerate(zip(individuals, all_6dof_poses)):
			all_6dof_pose_data[scs_ind_id] = {
				'6dof_pose': poses,
				'fitness': fitnesses[scs_ind_id],
				'infos': infos[scs_ind_id],
			}

	return all_6dof_pose_data


def init_replay_6dof_poses(run_folder_path, display_flags, display=True):
	
	folder = get_folder_path_from_str(run_folder_path)

	run_details, run_infos, cfg = load_run_output_files(folder)

	env = init_env(cfg, display=display)

	all_6dof_pose_data = get_all_6dof_data(folder=folder, cfg=cfg, display_flags=display_flags)

	return env, all_6dof_pose_data, display_flags


def replay_poses(run_folder_path, display_flags):
	env, all_6dof_pose_data, display_flags = init_replay_6dof_poses(run_folder_path, display_flags)

	replay_6dof_poses(
		env=env,
		all_6dof_pose_data=all_6dof_pose_data,
		display_flags=display_flags
	)

	print('Ending.')


def main():
	kwargs = get_replay_poses_kwargs()
	replay_poses(**kwargs)


if __name__ == "__main__":
	sys.exit(main())
