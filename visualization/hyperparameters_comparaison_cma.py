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

def plot_fitness_comparison(df, export_path=None):

    plt.figure(figsize=(12, 12))
    palette = sns.color_palette("tab10")
    sns.lineplot(data=df, x='individual', y='fitness', hue='learning_rate', palette=palette, estimator=None) 

    plt.title("Comparaison des distributions de fitness")
    plt.xlabel("Individu (rang)")
    plt.ylabel("Fitness")
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / (f'fitness_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def plot_qd_score_comparaison(df, export_path=None):
       
    plt.figure(figsize=(12, 12))
    palette = sns.color_palette("tab10")
    sns.lineplot(data=df, x='Evaluation', y='QD Score', hue='learning_rate', palette=palette) 

    plt.title("Comparaison des distributions de qd_sore")
    plt.xlabel("Evaluation")
    plt.ylabel("QD score")
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / (f'qd_score_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def plot_outcome_cvg_comparaison(df, export_path=None):
        
    plt.figure(figsize=(12, 12))
    palette = sns.color_palette("tab10")
    sns.lineplot(data=df, x='Evaluation', y='outcome_cvg', hue='learning_rate', palette=palette) 

    plt.title("Comparaison des distributions de outcome_cvg")
    plt.xlabel("Individu (rang)")
    plt.ylabel("outcome_cvg")
    plt.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / (f'outcome_cvg_comparison.png')
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()

    plt.close('all')

def get_fitness_records(data_dict,learning_rate):
    fitness_records = []
    fitness_hist = data_dict.get('fitness_hist')
    fitnes_list = sorted(fitness_hist[-1])

    for idx, fitness_value in enumerate(fitnes_list, start=1):
        fitness_records.append({
            'learning_rate': learning_rate,
            'fitness': fitness_value,
            'individual': idx,
        })
    return fitness_records

def get_coverage_records(data_dict,learning_rate):
    coverage_records = []

    outcome_archive_cvg_hist = data_dict['outcome_archive_cvg_hist']
    evals_hist = data_dict['n_evals_hist']

    for i, cvg in enumerate(outcome_archive_cvg_hist):
        coverage_records.append({
            'learning_rate': learning_rate,
            'Evaluation': evals_hist[i],
            'outcome_cvg': cvg,
        })

    return coverage_records

def get_fitness_hist_records(data_dict,learning_rate):
    fitness_hist_records = []

    fitness_hist = data_dict.get('fitness_hist')
    evals_hist = data_dict['n_evals_hist']

    fitness_hist_records.append({
        'learning_rate': learning_rate,
        'fitness hist':fitness_hist,
        'Evaluations': evals_hist,
    })

    return fitness_hist_records

def get_qd_score_df(fitness_hist_list):
    
    min_vals = []

    for fitness_hist in fitness_hist_list :
        for fitness_list in fitness_hist['fitness hist'] :
            min_fitness = min(val for val in fitness_list )
            min_vals.append(min_fitness)
        
    min_val = min(min_vals)
    shift_value = -min_val if min_val < 0 else 0

    qd_score_records = []

    for fitness_hist in fitness_hist_list :
        for i,fitness_list in enumerate(fitness_hist['fitness hist']):
            shifted_fitness = [val + shift_value for val in fitness_list]
            qd_score = np.sum(shifted_fitness)
            qd_score_records.append({
                'learning_rate': fitness_hist['learning_rate'],
                'Evaluation': fitness_hist['Evaluations'][i],
                'QD Score': qd_score,
            })   
    return pd.DataFrame(qd_score_records)

def get_shifted_fitness_df(fitness_lists):

    fitness_values = [f['fitness'] for f in fitness_lists]
    min_val = min(fitness_values)
    print(min_val)
    shift_value = -min_val if min_val < 0 else 0

    fitness_records = []

    for fitness_dict in fitness_lists :
        shifted_fitness = fitness_dict['fitness'] + shift_value 
        fitness_records.append({
            'learning_rate': fitness_dict['learning_rate'],
            'fitness': shifted_fitness,
            'individual': fitness_dict['individual'],
        })
        
    return pd.DataFrame(fitness_records)

def load_data(path, learning_rate):

    data_path = os.path.join(path, "data_export.pkl")
 
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:

            data_dict = pickle.load(f)
            fitness_records = get_fitness_records(data_dict, learning_rate)
            fitness_hist_records = get_fitness_hist_records(data_dict,learning_rate)
            coverage_records = get_coverage_records(data_dict,learning_rate)

    else:
        print(f"Fichier non trouvé : {data_path}")

    return fitness_records, fitness_hist_records, coverage_records

def group_configs(args):
    target_object_name = args.object
    runs_dir = args.runs_dir

    grouped_paths = defaultdict(list)
    i = 0
    for root, _, files in os.walk(runs_dir):
        if 'config.pkl' in files:
            config_path = os.path.join(root, 'config.pkl')
            try:
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)
                object_name = config.get('env', {}).get('kwargs', {}).get('object_name', '')
                learning_rate = config.get('evo_proc', {}).get('learning_rate')
                if object_name == target_object_name:
                    grouped_paths[learning_rate].append(os.path.dirname(config_path))

            except Exception as e:
                print(f"Error reading {config_path}: {e}")
        i+=1

    if not grouped_paths:
        raise ValueError(f"No config.pkl files found for object '{target_object_name}' in {runs_dir}.")

    return grouped_paths

def main():
    args = parse_args()
    grouped_paths = group_configs(args)

    fitness_list = []
    fitnss_hist_list = []
    coverage_list = []

    for learning_rate, paths in grouped_paths.items():

        for path in paths:
            fitness_records,fitness_hist_records, coverage_records = load_data(path, learning_rate)

            fitness_list.extend(fitness_records)
            fitnss_hist_list.extend(fitness_hist_records)
            coverage_list.extend(coverage_records)

    if fitness_list:
        df = get_shifted_fitness_df(fitness_list)
        output_dir = os.path.join(args.dump_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_fitness_comparison(df, output_dir)
    else:
        print(f"No data found in all_fitness_records")

    if fitnss_hist_list:
        qd_df = get_qd_score_df(fitnss_hist_list)
        output_dir = os.path.join(args.dump_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_qd_score_comparaison(qd_df, output_dir)
    else:
        print(f"No data found in qd_score_list")

    if coverage_list:
        cvg_df = pd.DataFrame(coverage_list)
        output_dir = os.path.join(args.dump_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_outcome_cvg_comparaison(cvg_df, output_dir)
    else:
        print(f"No data found in coverage_list")

if __name__ == "__main__":
    main()
