
import argparse
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
import pdb
from collections import defaultdict
from visualization.vis_tools import load_all_robustness_infos, get_folder_path_from_str, load_all_flags, get_final_6dof_pose_from_infos, load_all_infos

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--object",
						type=str,
						required=True,
						help="Object name.")
	parser.add_argument("-r", "--runs-dir",
						type=str,
						default=str(Path(__file__).parent.parent / "runs"),
						help="Run folder directory")
	parser.add_argument("-d", "--dump-path",
						type=str,
						default=str(Path(__file__).parent.parent / "runs"),
						help="Path to the dump folders for the three methods.")
	args = parser.parse_args()
	return args

def get_color_palette():

	colors = {
		"RAND_SAMPLE": "#4363D8",        
		"ME_SCS": "#994F01",              
		"CMA_MAE": "#1AFF1B",     
		"ME_RAND": "#4D1829",         
		"PLACEIT_ME_RAND": "#F032E6", 
		"PLACEIT_ME_SCS": "#F58231",         
		"PLACEIT_CMA_MAE": "#FCDE18",     
		"PLACEIT_RAND_SAMPLE": "#56B4E9", 
		"FACE_ALIGNMENT": "#949494",      
		"PCA_ALIGNMENT": "#DC3220",
	}
	
	return colors

def plot_fitness_distribution_cma_mae(df, export_path=None, bins=100):
    """
    Plot la distribution de fitness uniquement pour l'algorithme CMA_MAE.
    """
    df_algo = df[df['algorithm'] == 'CMA_MAE']
    
    if df_algo.empty:
        print("Aucune donnée pour l'algorithme CMA_MAE.")
        return

    df = df_algo[np.isfinite(df["fitness"])]

    plt.figure(figsize=(32, 12))
	
    bin_colors = "#b82a2a"  #"#b2b2cc"
    n, bins_edges, patches = plt.hist(
		df["fitness"], bins=bins, color=bin_colors, alpha=0.7, edgecolor="black"
	)
    font_size = 42
    plt.xlabel("Fitness", fontsize=font_size)
    plt.ylabel("Individuals", fontsize=font_size, labelpad=18)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    max_count = n.max()
    x_min, x_max = plt.xlim()
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10))
    #plt.xlim(0, x_max)
    plt.xlim(0, 5.5)

    plt.ylim(0, max_count * 1.1)  # ⬅️ garde un peu de marge en haut

    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / 'fitness_distribution.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

# def plot_fitness_distribution_cma_mae(df, export_path=None, bins=100):
# 	"""
# 	Plot la distribution de fitness uniquement pour l'algorithme CMA_MAE.
# 	"""
# 	df_algo = df[df['algorithm'] == 'CMA_MAE']
	
# 	if df_algo.empty:
# 		print("Aucune donnée pour l'algorithme CMA_MAE.")
# 		return

# 	plt.figure(figsize=(12,3))
	
# 	# Histogramme

# 	df = df_algo[np.isfinite(df["fitness"])]

# 	plt.figure(figsize=(10, 6))
# 	#plt.hist(df["fitness"], bins=bins, color="#56B4E9", alpha=0.7, edgecolor="black")
# 	n, bins_edges, patches = plt.hist(df["fitness"], bins=bins, color="#b2b2cc", alpha=0.7, edgecolor="black")

# 	plt.xlabel("Fitness")
# 	plt.ylabel("Individuals")

# 	max_count = n.max()
# 	x_min, x_max = plt.xlim()
# 	plt.gca().xaxis.set_major_locator(MultipleLocator(1))  # ticks principaux tous les 0.1
# 	plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10))    # sous-ticks pour plus de précision
# 	plt.xlim(0, x_max)

# 	plt.legend()
# 	plt.tight_layout()

# 	if export_path is not None:
# 		export_path = Path(export_path)
# 		export_path.mkdir(parents=True, exist_ok=True)
# 		fig_path = export_path / 'fitness_distribution.png'
# 		plt.savefig(fig_path)
# 		print(f"{fig_path} has been successfully exported.")
# 	else:
# 		plt.show()

# 	plt.close('all')

def get_fitness_df(fitness_lists):

	fitness_records = []

	for fitness_dict in fitness_lists :
		fits = fitness_dict['fitness'] 
		fitness_records.append({
			'algorithm': fitness_dict['algorithm'],
			'fitness': fits,
			'is_robust': fitness_dict['is_robust'],
		})
		
	return pd.DataFrame(fitness_records)

def load_data(run_folder_path, algo_name):

	folder = get_folder_path_from_str(run_folder_path)

	robustness_fits = load_all_robustness_infos(folder)
	is_robust_pose_flag = load_all_flags(folder)
	fitness_records = []

	for idx, fits in enumerate(robustness_fits):
		fitness = fits.mean()
		fitness_records.append({
			'algorithm': algo_name,
			'fitness': fitness,
			'is_robust': is_robust_pose_flag[idx],
		})

	return fitness_records

def group_configs(args):
	target_object_name = args.object
	runs_dir = args.runs_dir

	grouped_paths = defaultdict(list)

	for root, _, files in os.walk(runs_dir):
		if 'config.pkl' in files:
			config_path = os.path.join(root, 'config.pkl')
			try:
				with open(config_path, 'rb') as f:
					config = pickle.load(f)
				object_name = config.get('env', {}).get('kwargs', {}).get('object_name', '')
				algorithm = str(config.get('algorithm')).split('.')[-1].split(':')[0]
				if object_name == target_object_name and algorithm == 'CMA_MAE' :
					grouped_paths[algorithm].append(os.path.dirname(config_path))

			except Exception as e:
				print(f"Error reading {config_path}: {e}")

	if not grouped_paths:
		raise ValueError(f"No config.pkl files found for object '{target_object_name}' in {runs_dir}.")

	return grouped_paths

def main():
	args = parse_args()
	grouped_paths = group_configs(args)

	fitness_list = []

	for algo, paths in grouped_paths.items():
		print(algo)
		for path in paths:
			fitness_records = load_data(path, algo)
			fitness_list.extend(fitness_records)


	if fitness_list:
		df = get_fitness_df(fitness_list)
		output_dir = os.path.join(args.dump_path, "plots")
		os.makedirs(output_dir, exist_ok=True)
		plot_fitness_distribution_cma_mae(df, output_dir)
	else:
		print(f"No data found in all_fitness_records")


if __name__ == "__main__":
	main()
