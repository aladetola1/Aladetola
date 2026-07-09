
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import pickle
import yaml
from yaml.loader import SafeLoader
import pdb
import utils.common_tools as uct
from pathlib import Path
import numpy as np

def _safe_1d(x):
    return np.asarray(x).squeeze()

TIME_LABEL2VIS_STR = {
        'n_evals_hist': 'number of rollouts',
        'run_time_hist': 'run time in sec',
    }


def plotting_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-path",
                        type=str, 
                        default=None,
                        help="Path to the dump folder which contains runnning data to plot.")
    args = parser.parse_args()
    return args


def arg_parse_dump_path():
    args = plotting_arg_parser()
    dump_path = args.run_path

    if dump_path is None:
        raise AttributeError('Unset dump_path. (Call as : plotting.py --dump-path [path]')

    # Makes sure the dump path follows the right pattern
    dump_path = dump_path if dump_path[-1] != '/' else dump_path[:-1]

    return dump_path


def load_output_data(dump_path):

    run_name2load = dump_path
    details_export_pkl = run_name2load + '/details_export.pkl'
    data_export_pkl = run_name2load + '/data_export.pkl'

    with open(details_export_pkl, 'rb') as f:
        details_dict = pickle.load(f)

    with open(data_export_pkl, 'rb') as f:
        data_dict = pickle.load(f)

    return details_dict, data_dict


def load_run_info(dump_path):
    run_name2load = dump_path
    run_info_path = run_name2load + '/run_infos.yaml'

    # Open the file and load the file
    with open(run_info_path) as f:
        run_info_data = yaml.load(f, Loader=SafeLoader)

    return run_info_data


def path_safe_check(path):
    if not os.path.exists(path):
        raise AttributeError(f'Path does not exists : {path}')


def export_path_safe_create(path):
    if not os.path.exists(path):
        print(f'path={path} does not exists : creating it.')
        os.makedirs(path)
        print(f'path={path} successfully created.')
    else:
        print(f'path={path} already exists. Exporting path might overwrite previously generated plots.')


def plot_outcome_archive_coverage(data_dict, time_label='n_evals_hist', title='',export_path=None):

    assert time_label in ['n_evals_hist', 'run_time_hist']
    time_label_vis_str = TIME_LABEL2VIS_STR[time_label]

    outcome_archive_cvg_hist = data_dict['outcome_archive_cvg_hist']
    time_hist = data_dict[time_label]

    assert len(outcome_archive_cvg_hist) == len(time_hist)

    df = pd.DataFrame({time_label_vis_str: time_hist,
                       'output archive coverage': outcome_archive_cvg_hist})

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.plot(
    	np.asarray(time_hist).ravel(),
    	np.asarray(outcome_archive_cvg_hist).ravel(),
    	linewidth=2
    )

    title_str = title
    plt.suptitle(title_str)
    fig.subplots_adjust(bottom=0.15, wspace=0.4, hspace=0.3, right=0.94, left=0.06, top=0.90)

    if export_path is not None:
        export_path = export_path + '/' if export_path[-1] != '/' else export_path
        fig_path = export_path + title_str.replace(' ', '_').replace('|', '_') + '.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()
        plt.close()

def plot_success_archive_coverage(data_dict, time_label='n_evals_hist', title='', export_path=None):

    assert time_label in ['n_evals_hist', 'run_time_hist']
    time_label_vis_str = TIME_LABEL2VIS_STR[time_label]

    success_archive_cvg_hist = data_dict['success_archive_cvg_hist']
    time_hist = data_dict[time_label]

    assert len(success_archive_cvg_hist) == len(time_hist)

    df = pd.DataFrame({time_label_vis_str: time_hist,
                       'success archive coverage': success_archive_cvg_hist})

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.plot(
    	df[time_label_vis_str].to_numpy().ravel(),
    	df['success archive coverage'].to_numpy().ravel(),
    	linewidth=2
    )

    title_str = title
    plt.suptitle(title_str)
    fig.subplots_adjust(bottom=0.15, wspace=0.4, hspace=0.3, right=0.94, left=0.06, top=0.90)

    if export_path is not None:
        export_path = export_path + '/' if export_path[-1] != '/' else export_path
        fig_path = export_path + title_str.replace(' ', '_').replace('|', '_') + '.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()
        plt.close()


def plot_qd_score_over_evals(data_dict, title='', export_path=None):
    fitness_hist = data_dict['fitness_hist']
    evals_hist = data_dict['n_evals_hist']

    # shift fitness to positive values
    all_fitness_values = [val for fitness_list in fitness_hist for val in fitness_list]
    min_val = min(all_fitness_values)
    shift_value = -min_val if min_val < 0 else 0

    qd_score_per_gen = []
    for i,fitness_list in enumerate(fitness_hist):
        shifted_fitness = [val + shift_value for val in fitness_list]
        qd_score = np.sum(shifted_fitness)
        qd_score_per_gen.append({'Evaluation': evals_hist[i], 'QD Score': qd_score})

    # create data frame for the plot
    df = pd.DataFrame(qd_score_per_gen)
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.plot(
    	df['Evaluation'].to_numpy(dtype=np.float64).ravel(),
    	df['QD Score'].to_numpy(dtype=np.float64).ravel(),
    	linewidth=2
    )

    plt.xlabel("Evaluation")
    plt.ylabel("QD Score")
    plt.title(title)
    
    fig.subplots_adjust(bottom=0.15, wspace=0.4, hspace=0.3, right=0.94, left=0.06, top=0.90)

    if export_path is not None:
        export_path = export_path + '/' if export_path[-1] != '/' else export_path
        fig_path = export_path + title.replace(' ', '_').replace('|', '_') + '.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()
        plt.close()


def plot_fitness_mean_over_evals(data_dict, title='', export_path=None):
    fitness_hist = data_dict['fitness_hist']
    evals_hist = data_dict['n_evals_hist']
    
    mean_fitness_per_gen = []
    for i,fitness_list in enumerate(fitness_hist):
        mean_fitness = np.mean(fitness_list)
        mean_fitness_per_gen.append({'Evaluation': evals_hist[i], 'Mean Fitness': mean_fitness})

    df = pd.DataFrame(mean_fitness_per_gen)

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.plot(
    	df['Evaluation'].to_numpy(dtype=np.float64).ravel(),
    	df['Mean Fitness'].to_numpy(dtype=np.float64).ravel(),
    	linewidth=2
    )

    plt.xlabel("Evaluation")
    plt.ylabel("mean Fitness")
    plt.title(title)
    
    fig.subplots_adjust(bottom=0.15, wspace=0.4, hspace=0.3, right=0.94, left=0.06, top=0.90)

    if export_path is not None:
        export_path = export_path + '/' if export_path[-1] != '/' else export_path
        fig_path = export_path + title.replace(' ', '_').replace('|', '_') + '.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()
        plt.close()

def plot_max_fitness_over_evals(data_dict, title='', export_path=None):
    fitness_hist = data_dict['fitness_hist']
    evals_hist = data_dict['n_evals_hist']
    max_fitness_per_gen = []

    for i, fitness_list in enumerate(fitness_hist):
        max_fitness = max(fitness_list)
        max_fitness_per_gen.append({'Evaluation': evals_hist[i], 'Max Fitness': max_fitness})

    df = pd.DataFrame(max_fitness_per_gen)

    # Création du plot
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.plot(
    	df['Evaluation'].to_numpy().ravel(),
    	df['Max Fitness'].to_numpy().ravel(),
    	linewidth=2
    )

    plt.xlabel("Evaluation")
    plt.ylabel("Max Fitness value")
    plt.title(title)
    
    fig.subplots_adjust(bottom=0.15, wspace=0.4, hspace=0.3, right=0.94, left=0.06, top=0.90)

    if export_path is not None:
        export_path = export_path + '/' if export_path[-1] != '/' else export_path
        fig_path = export_path + title.replace(' ', '_').replace('|', '_') + '.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()
        plt.close()

def plot_fitness(data_dict, title='', export_path=None, shift=False):
    fitness_hist = data_dict['fitness_hist']
    last_fitness = sorted(fitness_hist[-1])
    df = pd.DataFrame({'Fitness (sorted)': last_fitness, 'Index': range(len(last_fitness))})

    if shift :
        min_val = min(last_fitness)
        shift_value = -min_val if min_val < 0 else 0
        shifted_fitness = [val + shift_value for val in last_fitness]

    df = pd.DataFrame({'Fitness (sorted)': last_fitness, 'Index': range(len(last_fitness))})

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(
    	df['Index'].to_numpy().ravel(),
    	df['Fitness (sorted)'].to_numpy().ravel(),
    	linewidth=2
    )

    plt.xlabel("Individuals ranked")
    plt.ylabel("Fitness value")
    plt.title(title)

    fig.subplots_adjust(bottom=0.15, wspace=0.4, hspace=0.3, right=0.94, left=0.06, top=0.90)

    if export_path is not None:
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        fig_path = export_path / f"{title.replace(' ', '_').replace('|', '_')}.png"
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()
        plt.close()

def plot_success_archive_qd_score(details_dict, data_dict, time_label='n_evals_hist', title='', export_path=None):

    assert time_label in ['n_evals_hist', 'run_time_hist']
    time_label_vis_str = TIME_LABEL2VIS_STR[time_label]

    success_archive_qd_score_hist = data_dict['success_archive_qd_score_hist']
    time_hist = data_dict[time_label]

    assert len(success_archive_qd_score_hist) == len(time_hist)

    df = pd.DataFrame({time_label_vis_str: time_hist,
                       'success archive qd score': success_archive_qd_score_hist})

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.plot(
    	df[time_label_vis_str].to_numpy().ravel(),
    	df['success archive qd score'].to_numpy().ravel(),
    	linewidth=2
    )

    title_str = title
    plt.suptitle(title_str)
    fig.subplots_adjust(bottom=0.15, wspace=0.4, hspace=0.3, right=0.94, left=0.06, top=0.90)

    if export_path is not None:
        export_path = export_path + '/' if export_path[-1] != '/' else export_path
        fig_path = export_path + title_str.replace(' ', '_').replace('|', '_') + '.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()
        plt.close()

def plot_n_successful_cells(details_dict, data_dict, time_label='n_evals_hist', title='', export_path=None):

    assert time_label in ['n_evals_hist', 'run_time_hist']
    time_label_vis_str = TIME_LABEL2VIS_STR[time_label]

    n_successful_cells_list = data_dict['n_successful_cells_list']
    time_hist = data_dict[time_label]

    assert len(n_successful_cells_list) == len(time_hist)

    df = pd.DataFrame({time_label_vis_str: time_hist,
                       'number of successful cells': n_successful_cells_list})

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.plot(
    	df[time_label_vis_str].to_numpy().ravel(),
    	df['number of successful cells'].to_numpy().ravel(),
    	linewidth=2
    )

    title_str = title
    plt.suptitle(title_str)
    fig.subplots_adjust(bottom=0.15, wspace=0.4, hspace=0.3, right=0.94, left=0.06, top=0.90)

    if export_path is not None:
        export_path = export_path + '/' if export_path[-1] != '/' else export_path
        fig_path = export_path + title_str.replace(' ', '_').replace('|', '_') + '.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()
        plt.close()


def plot_export_routine(dump_path, details_dict, data_dict):

    # Dump path sanity checks
    if not uct.is_export_path_type_valid(dump_path, attempted_export_str='plots'):
        return

    export_path_root = uct.get_export_path_root(dump_path=dump_path)
    export_path = export_path_root + '/plots'
    export_path_safe_create(export_path)


    # Number of evaluations
    plot_outcome_archive_coverage(data_dict, time_label='n_evals_hist', export_path=export_path,
                                  title='outcome archive coverage over n evals')

    # Running time (in sec)
    plot_success_archive_coverage(data_dict, time_label='n_evals_hist', export_path=export_path,
                                  title='success archive coverage over n evals')
    
    # Fitness
    plot_fitness(data_dict, export_path=export_path,
                                  title='fitness of successeful individulas', shift=False)
    
    plot_success_archive_qd_score(details_dict, data_dict,  time_label='n_evals_hist', export_path=export_path,
                                  title='success archive qd score over n evals')

    plot_n_successful_cells(details_dict, data_dict,  time_label='n_evals_hist', export_path=export_path,
                                  title='number of successful cells over n evals')

    # plot_max_fitness_over_evals(data_dict, export_path=export_path,
    #                               title='max scs fitness over evals')
    
    # plot_fitness_mean_over_evals(data_dict, export_path=export_path,
    #                              title='scs fitness mean over n evals')
    
    plt.close('all')

def local_plotting(dump_path):
    """Run the script locally (not used as an external module)."""
    path_safe_check(dump_path)

    details_dict, data_dict = load_output_data(dump_path)
    plot_export_routine(dump_path=dump_path, details_dict=details_dict, data_dict=data_dict)

    print('local_plotting() : running over.')


def main_local_plot():
    dump_path = arg_parse_dump_path()
    local_plotting(dump_path)


if __name__ == "__main__":
    sys.exit(main_local_plot())


