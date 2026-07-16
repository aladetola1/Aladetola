
import argparse
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
import pdb
from collections import defaultdict
from visualization.vis_tools import load_all_infos

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

def plot_fitness_comparison(df, export_path=None) :

    plt.figure(figsize=(12, 12))
    palette = get_color_palette()
    sns.lineplot(data=df, x='individual', y='fitness', hue='algorithm', palette=palette, estimator=None) 

    plt.title("Comparaison des distributions de fitness du success archive")
    plt.xlabel("Individu (rang)")
    plt.ylabel("Fitness")
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / ('fitness_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def plot_qd_scs_score_comparaison(df, export_path=None):
       
    plt.figure(figsize=(12, 12))
    palette = get_color_palette()
    sns.lineplot(data=df, x='Evaluation', y='qd_scs', hue='algorithm', palette=palette) 

    plt.title("Comparaison of QD Score of success archive")
    plt.xlabel("Evaluation")
    plt.ylabel("QD score")
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / ('qd_score_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def plot_outcome_cvg_comparaison(df, export_path=None):
        
    plt.figure(figsize=(12, 12))
    palette = get_color_palette()
    sns.lineplot(data=df, x='Evaluation', y='outcome_cvg', hue='algorithm', palette=palette) 

    plt.title("Comparaison of outcome archive couverage")
    plt.xlabel("Evaluations")
    plt.ylabel("outcome_cvg")
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / ('outcome_cvg_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def plot_scs_cvg_comparaison(df, export_path=None):
        
    plt.figure(figsize=(12, 12))
    palette = get_color_palette()
    sns.lineplot(data=df, x='Evaluation', y='scs_cvg', hue='algorithm', palette=palette) 

    plt.title("Comparaison of success archive couverage")
    plt.xlabel("Evaluations")
    plt.ylabel("success_cvg")
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / ('success_cvg_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def plot_scs_cells_comparaison(df, export_path=None):
        
    plt.figure(figsize=(12, 12))
    palette = get_color_palette()
    sns.lineplot(data=df, x='Evaluation', y='scs_cell', hue='algorithm', palette=palette) 

    plt.title("Comparaison of the number of successful cells")
    plt.xlabel("Evaluations")
    plt.ylabel("number of scs cells")
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / ('nbr_of_successeful_cells_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def plot_nbr_ivalid_inds_comparaison(df, export_path=None):
        
    plt.figure(figsize=(12, 12))
    palette = get_color_palette()
    sns.lineplot(data=df, x='Evaluation', y='nbr_invalid_inds', hue='algorithm', palette=palette) 

    plt.title("Comparison of number of invalid individuals per evaluation")
    plt.xlabel("Evaluations")
    plt.ylabel("number of invalid individuals")
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / ('nbr_of_invalid_individuals_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def plot_nbr_eq_stable_cells_comparaison(df, export_path=None):
        
    plt.figure(figsize=(8, 5))
    palette = get_color_palette()
    sns.barplot(data=df, x='algorithm', y='nbr', hue='algorithm', palette=palette) 

    plt.title("Comparison of the number of stably balanced cells")
    plt.xlabel("Algorithm")
    plt.ylabel("number of stably balanced cells")
    plt.xticks(rotation=70)
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / ('number_of_stably_balanced_cells_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def plot_ratio_eq_stable_cells_comparaison(df, export_path=None):
        
    plt.figure(figsize=(8, 5))
    palette = get_color_palette()
    sns.barplot(data=df, x='algorithm', y='ratio', hue='algorithm', palette=palette)

    plt.title("Comparison of Stable Equilibrium Ratios on all successful cells")
    plt.xlabel("Algorithm")
    plt.ylabel("ratio")
    plt.xticks(rotation=70)
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / ('ratio_stably_balanced_cell_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def plot_final_poses_variance_comparaison(df, export_path=None):
        
    plt.figure(figsize=(8, 5))
    palette = get_color_palette()
    sns.barplot(data=df, x='algorithm', y='variance', hue='algorithm', palette=palette) 

    plt.title("Comparison of final position variances")
    plt.xlabel("Algorithm")
    plt.ylabel("variance")
    plt.xticks(rotation=70)
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / ('final_6dof_poses_variance_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def plot_init_poses_variance_comparaison(df, export_path=None):
        
    plt.figure(figsize=(8, 5))
    palette = get_color_palette()
    sns.barplot(data=df, x='algorithm', y='variance', hue='algorithm', palette=palette) 

    plt.title("Comparison of initial position variances")
    plt.xlabel("Algorithm")
    plt.ylabel("variance")
    plt.xticks(rotation=70)
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / ('init_6dof_poses_variance_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def get_fitness_records(data_dict,algo_name):
    fitness_records = []
    fitness_hist = data_dict.get('fitness_hist')
    fitnes_list = sorted(fitness_hist[-1])

    for idx, fitness_value in enumerate(fitnes_list, start=1):
        fitness_records.append({
            'algorithm': algo_name,
            'fitness': fitness_value,
            'individual': idx,
        })
    return fitness_records

def get_coverage_records(data_dict,algo_name):
    coverage_records = []

    outcome_archive_cvg_hist = data_dict['outcome_archive_cvg_hist']
    evals_hist = data_dict['n_evals_hist']

    for i, cvg in enumerate(outcome_archive_cvg_hist):
        coverage_records.append({
            'algorithm': algo_name,
            'Evaluation': evals_hist[i],
            'outcome_cvg': cvg,
        })

    return coverage_records

def get_scs_coverage_records(data_dict,algo_name):
    coverage_records = []

    outcome_archive_cvg_hist = data_dict['success_archive_cvg_hist']
    evals_hist = data_dict['n_evals_hist']

    for i, cvg in enumerate(outcome_archive_cvg_hist):
        coverage_records.append({
            'algorithm': algo_name,
            'Evaluation': evals_hist[i],
            'scs_cvg': cvg,
        })

    return coverage_records

def get_nbr_scs_inds(data_dict,algo_name):
    scs_inds_records = []

    n_successful_cells_list = data_dict['n_successful_cells_list']
    evals_hist = data_dict['n_evals_hist']

    for i, ind in enumerate(n_successful_cells_list):
        scs_inds_records.append({
            'algorithm': algo_name,
            'Evaluation': evals_hist[i],
            'scs_cell': ind,
        })

    return scs_inds_records

def get_nbr_eq_stable_cells(path,algo_name):

    folder_path = Path(path)
    if len(list(folder_path.glob("**/run_details*.yaml"))) == 0:
        raise FileNotFoundError("No run found")
    infos = load_all_infos(folder_path)

    eq_stable_cells_list = [cell_info['is_equilibrium_stable'] for cell_info in infos]


    nbr = int(np.sum(eq_stable_cells_list))

    nbr_eq_stable_cells = [{
            'algorithm': algo_name,
            'nbr': nbr,
        }]

    return nbr_eq_stable_cells

def get_nbr_invalid_inds(data_dict,algo_name):

    nbr_invalid_inds_records = []
    nbr_invalid_inds_list = data_dict.get('n_invalid_cells')
    evals_hist = data_dict['n_evals_hist']

    for i, ind in enumerate(nbr_invalid_inds_list):
        nbr_invalid_inds_records.append({
            'algorithm': algo_name,
            'Evaluation': evals_hist[i],
            'nbr_invalid_inds': ind,
        })

    return nbr_invalid_inds_records

def get_ratio_eq_stable_cells(path,algo_name):

    folder_path = Path(path)
    if len(list(folder_path.glob("**/run_details*.yaml"))) == 0:
        raise FileNotFoundError("No run found")
    infos = load_all_infos(folder_path)

    eq_stable_cells_list = [cell_info['is_equilibrium_stable'] for cell_info in infos]
    scs_cells_list = [cell_info['is_success'] for cell_info in infos]
    nbr_eq_stable_cells = int(np.sum(eq_stable_cells_list))
    all_scs_cells = int(np.sum(scs_cells_list))
    ratio = nbr_eq_stable_cells/all_scs_cells
    ratio_eq_stable_cells = [{
            'algorithm': algo_name,
            'ratio': ratio,
        }]

    return ratio_eq_stable_cells

def get_dq_scs_records(data_dict,algo_name):
    qd_records = []

    qd_scs = data_dict.get('success_archive_qd_score_hist')
    evals_hist = data_dict['n_evals_hist']

    for i, qd_score in enumerate(qd_scs):
        qd_records.append({
            'algorithm': algo_name,
            'qd_scs': qd_score,
            'Evaluation': evals_hist[i],
        })
    return qd_records

def get_fitness_df(fitness_lists):

    fitness_records = []

    for fitness_dict in fitness_lists :
        fits = fitness_dict['fitness'] 
        fitness_records.append({
            'algorithm': fitness_dict['algorithm'],
            'fitness': fits,
            'individual': fitness_dict['individual'],
        })
        
    return pd.DataFrame(fitness_records)

def get_qd_df(qd_scs_list):

    qd_records = []

    for qd_dict in qd_scs_list :
        qd_score = qd_dict['qd_scs'] 
        qd_records.append({
            'algorithm': qd_dict['algorithm'],
            'qd_scs': qd_score,
            'Evaluation': qd_dict['Evaluation'],
        })
        
    return pd.DataFrame(qd_records)

def get_final_poses_variance(data_dict,algo_name):

    final_6dof_poses = data_dict.get('final_6dof_pos',[])
    positions = [pose['xyz'] for pose in final_6dof_poses]
    orientations = [pose['euler_rpy'] for pose in final_6dof_poses]

    positions = np.array(positions)
    orientations = np.array(orientations)

    pos_variance = np.var(positions)
    orient_variance = np.var(orientations)

    variance = pos_variance + orient_variance

    final_poses_varaices = [{
            'algorithm': algo_name,
            'variance': variance,
        }]

    return final_poses_varaices

def get_init_poses_variance(data_dict,algo_name):

    init_6dof_poses = data_dict.get('init_6dof_pos',[])
    positions = [pose['xyz'] for pose in init_6dof_poses]
    orientations = [pose['euler_rpy'] for pose in init_6dof_poses]

    positions = np.array(positions)
    orientations = np.array(orientations)

    pos_variance = np.var(positions)
    orient_variance = np.var(orientations)

    variance = pos_variance + orient_variance

    init_poses_varaices = [{
            'algorithm': algo_name,
            'variance': variance,
        }]

    return init_poses_varaices


def load_data(path, algo_name):

    data_path = os.path.join(path, "data_export.pkl")
 
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:

            data_dict = pickle.load(f)
            fitness_records = get_fitness_records(data_dict, algo_name)
            qd_scs_records = get_dq_scs_records(data_dict,algo_name)
            coverage_records = get_coverage_records(data_dict,algo_name)
            scs_cvg_records = get_scs_coverage_records(data_dict,algo_name)
            nbr_scs_inds_records = get_nbr_scs_inds(data_dict,algo_name)
            #nbr_eq_stable_cells = get_nbr_eq_stable_cells(path,algo_name)
            nbr_invalid_inds_records = get_nbr_invalid_inds(data_dict,algo_name)
            #ratio_eq_stable_cells = get_ratio_eq_stable_cells(path,algo_name)
            final_poses_variances = get_final_poses_variance(data_dict,algo_name)
            init_poses_variances = get_init_poses_variance(data_dict,algo_name)
            nbr_eq_stable_cells =[]
            ratio_eq_stable_cells = []
    else:
        print(f"Fichier non trouvé : {data_path}")

    return fitness_records, qd_scs_records, coverage_records, scs_cvg_records, nbr_scs_inds_records, nbr_eq_stable_cells, final_poses_variances, ratio_eq_stable_cells, init_poses_variances, nbr_invalid_inds_records

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
                if object_name == target_object_name:
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
    qd_scs_list = []
    coverage_list = []
    scs_coverage_list = []
    scs_cells_list =[]
    nbr_eq_stable_cells_list = []
    final_poses_variances_list = []
    ratio_eq_stable_cells_list = []
    init_poses_variances_list = []
    nbr_invalid_inds_list = []

    for algo, paths in grouped_paths.items():
        print(algo)
        for path in paths:
            fitness_records, qd_scs_records, coverage_records, scs_cvg_records, nbr_scs_inds, nbr_eq_stable_cells, final_poses_variances, ratio_eq_stable_cells, init_poses_variances, nbr_invalid_inds_records = load_data(path, algo)

            fitness_list.extend(fitness_records)
            qd_scs_list.extend(qd_scs_records)
            coverage_list.extend(coverage_records)
            scs_coverage_list.extend(scs_cvg_records)
            scs_cells_list.extend(nbr_scs_inds)
            nbr_eq_stable_cells_list.extend(nbr_eq_stable_cells)
            final_poses_variances_list.extend(final_poses_variances)
            init_poses_variances_list.extend(init_poses_variances)
            ratio_eq_stable_cells_list.extend(ratio_eq_stable_cells)
            nbr_invalid_inds_list.extend(nbr_invalid_inds_records)


    if fitness_list:
        df = get_fitness_df(fitness_list)
        output_dir = os.path.join(args.dump_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_fitness_comparison(df, output_dir)
    else:
        print(f"No data found in all_fitness_records")

    if qd_scs_list:
        qd_df = get_qd_df(qd_scs_list)
        output_dir = os.path.join(args.dump_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_qd_scs_score_comparaison(qd_df, output_dir)
    else:
        print(f"No data found in qd_score_list")

    if coverage_list:
        cvg_df = pd.DataFrame(coverage_list)
        output_dir = os.path.join(args.dump_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_outcome_cvg_comparaison(cvg_df, output_dir)
    else:
        print(f"No data found in coverage_list")

    if scs_coverage_list:
        scs_cvg_df = pd.DataFrame(scs_coverage_list)
        output_dir = os.path.join(args.dump_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_scs_cvg_comparaison(scs_cvg_df, output_dir)
    else:
        print(f"No data found in scs_coverage_list")

    if scs_cells_list:
        scs_cells_df = pd.DataFrame(scs_cells_list)
        output_dir = os.path.join(args.dump_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_scs_cells_comparaison(scs_cells_df, output_dir)
    else:
        print(f"No data found in scs_cells_list")


    if final_poses_variances_list:
        final_poses_variances_list_df = pd.DataFrame(final_poses_variances_list)

        output_dir = os.path.join(args.dump_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_final_poses_variance_comparaison(final_poses_variances_list_df, output_dir)
    else:
        print(f"No data found in scs_cells_list")

    if init_poses_variances_list:
        init_poses_variances_list_df = pd.DataFrame(init_poses_variances_list)

        output_dir = os.path.join(args.dump_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_init_poses_variance_comparaison(init_poses_variances_list_df, output_dir)
    else:
        print(f"No data found in scs_cells_list")


    if nbr_invalid_inds_list:
        nbr_invalid_inds_list_df = pd.DataFrame(nbr_invalid_inds_list)

        output_dir = os.path.join(args.dump_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_nbr_ivalid_inds_comparaison(nbr_invalid_inds_list_df, output_dir)
    else:
        print(f"No data found in scs_cells_list")

if __name__ == "__main__":
    main()
