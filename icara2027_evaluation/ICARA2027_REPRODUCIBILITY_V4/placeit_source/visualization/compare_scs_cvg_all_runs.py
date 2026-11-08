
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
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def parse_args():
    parser = argparse.ArgumentParser()
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

def plot_all_in_grid(all_dfs, dump_path):
    #pdb.set_trace()

    cols = 4
    rows = int(np.ceil(len(all_dfs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5), sharex=False, sharey=False)
    axes = axes.flatten()
    palette = get_color_palette()

    title_name_map = {
        "ycb_banana_on_table": "banana-on-table",
        "ycb_mug_on_table": "mug-on-table",
        "cylinder_on_cube_with_cylinder_hole": "peg-in-hole",
        "ycb_spoon_on_mug": "spoon-in-mug",
        "ycb_bowl_on_ycb_bowl": "stack-bowl",        
        "screwdriver_on_screwdriver_holder": "screwdriver-on-support",              
        "ycb_mug_on_mug_tree": "hanging-mug",
        "ycb_power_drill_on_table": "powerdrill-on-table"   
    }

    legend_map = {
        "PLACEIT_RAND_SAMPLE": ("contact_rand_sample", palette["PLACEIT_RAND_SAMPLE"]),
        "PLACEIT_ME_SCS": ("contact_ME_scs", palette["PLACEIT_ME_SCS"]),
        "PLACEIT_ME_RAND": ("contact_ME_rand", palette["PLACEIT_ME_RAND"]),
        "PLACEIT_CMA_MAE": ("contact_CMA_MAE", palette["PLACEIT_CMA_MAE"]),
        "RAND_SAMPLE": ("rand_sample", palette["RAND_SAMPLE"]),        
        "ME_SCS": ("ME_scs", palette["ME_SCS"]),              
        "CMA_MAE": ("CMA_MAE", palette["CMA_MAE"]),     
        "ME_RAND": ("ME_rand", palette["ME_RAND"]),         
        "FACE_ALIGNMENT": ("face_alignment", palette["FACE_ALIGNMENT"]),      
        "PCA_ALIGNMENT": ("pca_alignment", palette["PCA_ALIGNMENT"]),
    }

    desired_order = [
        "rand_sample",
        "contact_rand_sample",
        "ME_rand",
        "contact_ME_rand",
        "ME_scs",
        "contact_ME_scs",
        "CMA_MAE",
        "contact_CMA_MAE",
        "face_alignment",
        "pca_alignment",
    ]

    for i, (sim_name, df) in enumerate(all_dfs.items()):
        ax = axes[i]
        sns.lineplot(data=df, x='Evaluation', y='scs_cvg', hue='algorithm',
                     palette=palette, ax=ax, legend=False)

        ax.set_title(title_name_map.get(sim_name, sim_name))
        ax.set_xlabel("Evaluations")
        #ax.set_ylabel("scs cvg")
        ax.set_ylabel(r"$\mathit{cvg}(\Phi^s)$")

        # Format scientifique (10^x) pour X et Y
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # toujours scientifique
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        # afficher les axes sur tous les côtés
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        # rendre l’axe Y un peu plus étroit
        ax.margins(y=0.05)

    # légende globale dans l'ordre voulu
    handles = []
    for key in desired_order:
        # retrouver la clé correspondante dans legend_map
        for algo, (label, color) in legend_map.items():
            if label == key:
                handles.append(mpatches.Patch(color=color, label=label))
                break

    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10, frameon=False)
    
    # Reduce vertical space between subplots and space to top edge
    plt.subplots_adjust(hspace=0.70, wspace=0.25, top=0.94, bottom=0.23, left=0.04, right=0.98)
    #plt.tight_layout(rect=[0, 0.1, 1, 0.93])

    os.makedirs(dump_path, exist_ok=True)
    fig_path = os.path.join(dump_path, "all_simulations_grid.png")
    plt.savefig(fig_path)
    fig_path = os.path.join(dump_path, "all_simulations_grid.pdf")
    plt.savefig(fig_path)
    print(f"Grid saved at {fig_path}")
    plt.close()

# def plot_all_in_grid(all_dfs, dump_path):
#     n = len(all_dfs)
#     cols = 4
#     rows = int(np.ceil(n / cols))

#     fig, axes = plt.subplots(rows, cols, figsize=(20, 8), sharex=True, sharey=False)
#     axes = axes.flatten()
#     palette = get_color_palette()

#     legend_name_map = {
#         "PLACEIT_RAND_SAMPLE": "contact_rand_sample",
#         "PLACEIT_ME_SCS": "contact_ME_scs",
#         "PLACEIT_ME_RAND": "contact_ME_rand",
#         "PLACEIT_CMA_MAE": "contact_CMA_MAE",
#         "RAND_SAMPLE": "rand_sample",        
#         "ME_SCS": "ME_scs",              
#         "CMA_MAE": "CMA_MAE",     
#         "ME_RAND": "ME_rand",         
#         "FACE_ALIGNMENT": "face_alignment",      
#         "PCA_ALIGNMENT": "pca_alignment",
#     }

#     title_name_map = {
#         "ycb_banana_on_table": "banana-on-table",
#         "ycb_mug_on_table": "mug-on-table",
#         "cylinder_on_cube_with_cylinder_hole": "peg-in-hole",
#         "ycb_spoon_on_mug": "spoon-in-mug",
#         "ycb_bowl_on_ycb_bowl": "stack-bowl",        
#         "screwdriver_on_screwdriver_holder": "screwdriver-on-support",              
#         "ycb_mug_on_mug_tree": "hanging-mug",
#         "ycb_power_drill_on_table": "powerdrill-on-table"   
#     }

#     legend_color_map = {
#         "rand_sample": palette["RAND_SAMPLE"],
#         "contact_rand_sample": palette["PLACEIT_RAND_SAMPLE"],
#         "ME_rand": palette["ME_RAND"],
#         "contact_ME_rand": palette["PLACEIT_ME_RAND"],
#         "ME_scs": palette["ME_SCS"],
#         "contact_ME_scs": palette["PLACEIT_ME_SCS"],
#         "CMA_MAE": palette["CMA_MAE"],
#         "contact_CMA_MAE": palette["PLACEIT_CMA_MAE"],
#         "face_alignment": palette["FACE_ALIGNMENT"],
#         "pca_alignment": palette["PCA_ALIGNMENT"],
#     }

#     for i, (sim_name, df) in enumerate(all_dfs.items()):
#         sns.lineplot(data=df, x='Evaluation', y='scs_cvg', hue='algorithm',
#                      palette=palette, ax=axes[i], legend=False)
#         axes[i].set_title(title_name_map.get(sim_name, sim_name))
#         axes[i].set_xlabel("Evaluations")
#         axes[i].set_ylabel("scs cvg")
#         axes[i].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
#         axes[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#         # Forcer format scientifique pour l'axe x
#         axes[i].xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
#         axes[i].ticklabel_format(axis='x', style='sci', scilimits=(0,0))

#     # Créer légende globale en bas avec noms modifiés
#     desired_order = [
#         "rand_sample",
#         "contact_rand_sample",
#         "ME_rand",
#         "contact_ME_rand",
#         "ME_scs",
#         "contact_ME_scs",
#         "CMA_MAE",
#         "contact_CMA_MAE",
#         "face_alignment",
#         "pca_alignment",
#     ]
#     handles = [
#         mpatches.Patch(color=legend_color_map[name], label=legend_name_map.get(name, name))
#         for name in desired_order
#     ]

#     fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10, frameon=False)
#     plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # laisse 10% en bas pour la légende


#     os.makedirs(dump_path, exist_ok=True)
#     fig_path = os.path.join(dump_path, "all_simulations_grid.png")
#     plt.savefig(fig_path)
#     print(f"Grid saved at {fig_path}")
#     plt.close()

def plot_scs_cvg_comparaison(df, export_path=None):
        
    plt.figure(figsize=(12, 12))
    palette = get_color_palette()
    sns.lineplot(data=df, x='Evaluation', y='scs_cvg', hue='algorithm', palette=palette) # marker='o'

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

def get_scs_coverage_records(data_dict,algo_name):
    coverage_records = []

    outcome_archive_cvg_hist = data_dict['success_archive_cvg_hist']
    evals_hist = data_dict['n_evals_hist']

    #pdb.set_trace()

    coverage_records = [
        {
            'algorithm': algo_name, 
            'Evaluation': evals_hist[i], 
            'scs_cvg': cvg 
        } 
        for i, cvg in enumerate(outcome_archive_cvg_hist)
    ]

    #for i, cvg in enumerate(outcome_archive_cvg_hist):
    #    coverage_records.append({
    #        'algorithm': algo_name,
    #        'Evaluation': evals_hist[i],
    #        'scs_cvg': cvg,
    #    })

    return coverage_records

def load_data(path, algo_name):

    data_path = os.path.join(path, "data_export.pkl")
 
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
            scs_cvg_records = get_scs_coverage_records(data_dict,algo_name)
    else:
        print(f"Fichier non trouvé : {data_path}")
        scs_cvg_records = []

    return scs_cvg_records

def group_configs(args):
    runs_dir = args.runs_dir

    grouped_paths = defaultdict(lambda: defaultdict(list))


    for root, _, files in os.walk(runs_dir):
        if 'config.pkl' in files:
            config_path = os.path.join(root, 'config.pkl')
            try:
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)

                object_name = config.get('env', {}).get('kwargs', {}).get('object_name', '')
                target_name = config.get('env', {}).get('kwargs', {}).get('target_name', '')
                sim_name = f"{object_name}_on_{target_name}"
                algorithm = str(config.get('algorithm')).split('.')[-1].split(':')[0]
                
                grouped_paths[sim_name][algorithm].append(os.path.dirname(config_path))
            except Exception as e:
                print(f"Error reading {config_path}: {e}")

    return grouped_paths

def main():
    args = parse_args()
    grouped_paths = group_configs(args)
    
    #pdb.set_trace()

    all_dfs = {}
    for sim_name, algo_dict in grouped_paths.items():
        scs_cvg_list = []
        
        #pdb.set_trace()
        DEBUG = False #True
        if DEBUG:
            algo_dict = {k: algo_dict[k] for k in list(algo_dict)[:1]}
            
        for algo, paths in algo_dict.items():
            for path in paths:
                scs_cvg_records= load_data(path, algo)
                scs_cvg_list.extend(scs_cvg_records)
        
        if scs_cvg_list:
            qd_df = pd.DataFrame(scs_cvg_list)
            all_dfs[sim_name] = qd_df
            #output_dir = os.path.join(args.dump_path, "QD score plots", f"{sim_name}.png")
            #plot_qd_scs_score_comparaison(qd_df, output_dir, title=f"{sim_name} QD Score")
        else:
            print(f"No data for {sim_name}")

    if all_dfs:
            plot_all_in_grid(all_dfs, os.path.join(args.dump_path, "plots"))

if __name__ == "__main__":
    main()
