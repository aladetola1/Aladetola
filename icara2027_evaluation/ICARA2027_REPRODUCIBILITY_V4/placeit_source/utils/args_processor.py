
import argparse
from functools import partial
from pathlib import Path

from utils.common_tools import arg_clean_str, wrapped_partial

from algorithms.evaluate import evaluate_placement_routine
import pdb
import configs.qd_config as qd_cfg
import configs.obj_config as obj_cfg


def arg_handler_greater(name, min, value):
    v = int(value)
    if v <= min:
        raise argparse.ArgumentTypeError(f"The {name.strip()} must be greater than {min}")
    return v


def parse_input_args():
    parser = argparse.ArgumentParser()

    # ---------------------------------------------------------------------------------------------------------------- #
    # Execution mode
    # ---------------------------------------------------------------------------------------------------------------- #
    parser.add_argument("-d", "--display",
                        action="store_true",
                        help="Display trajectory generation. Not supported with parallelization.")
    parser.add_argument("-ll", "--parallelize",
                        action="store_true",
                        help="Trigger multiprocessing parallelization.")
    parser.add_argument("-db", "--debug",
                        action="store_true",
                        help="Debug mode. Not supported with parallelization")

    # ---------------------------------------------------------------------------------------------------------------- #
    # Evolutionary algorithm args
    # ---------------------------------------------------------------------------------------------------------------- #
    # --) Tier 1
    parser.add_argument("-a", "--algorithm",
                        type=arg_clean_str,
                        default="me_scs",
                        choices=qd_cfg.ALGO_ARG_TO_METHOD,
                        help=f"Algorithm variant. Supported variants: {qd_cfg.ALGO_ARG_TO_METHOD}")
    
    parser.add_argument("-o", "--object",
                        type=str,
                        default=None,
                        help="The object to grasp")
    
    parser.add_argument("-t", "--target",
                        type=str,
                        default=None,
                        help="The target where to place")
    
    # --) Tier 2
    parser.add_argument("-sm", "--sigma-mut",
                        type=float,
                        default=0.2,
                        help="Variance of the gaussian mutation perturbation.")

    parser.add_argument("-p", "--population",
                        type=partial(arg_handler_greater, "population size", 1),
                        default=500,

                        help="The population size")
    parser.add_argument("-ii", "--include-invalids",
                        action="store_true",
                        help="Include invalid individual in the search. By default, new individuals are sampled until"
                             "the population (or offspring) is full of valid individuals.")
    # --) Tier 3
    parser.add_argument("-x", "--prob-cx",
                       type=float,
                       default=0.,
                       help="Probability to apply crossover.")

    parser.add_argument("-c", "--concave-mesh",
                       action="store_true",
                       help="load target object using concave mesh")
    
    # --) Ending conditions
    parser.add_argument("-nbr", "--n-budget-rollout",
                        type=int,
                        default=None,
                        help="Maximum number of evaluation (= rollout) before ending the evolutionary process.")
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # I/O args
    # ---------------------------------------------------------------------------------------------------------------- #
    parser.add_argument("-f", "--folder-name",
                        type=str,
                        default="run",
                        help="Run folder name suffix")
    parser.add_argument("-l", "--log-path",
                        type=str,
                        default=str(Path(__file__).parent.parent / "runs"),
                        help="Run folder path")

    # ---------------------------------------------------------------------------------------------------------------- #
    # Transferability Map-Elites (TR-ME) args
    # ---------------------------------------------------------------------------------------------------------------- #

    parser.add_argument("-drf", "--domain-randomization-fitness",
                        action="store_true",
                        help="Trigger DR-mixture fitness.")
    
    
    return parser.parse_args()


def init_run_details(ns_raw_args):
    evaluation_function = ns_raw_args['evaluation_function']
    genotype_len = ns_raw_args['genotype_len']
    qd_method = ns_raw_args['qd_method']
    pop_size = ns_raw_args['pop_size']
    archive_limit_size = qd_cfg.ARCHIVE_LIMIT_SIZE

    archive_limit_strat = ns_raw_args['archive_limit_strat']
    mut_strat = ns_raw_args['mut_strat']

    nb_offsprings_to_generate = ns_raw_args['nb_offsprings_to_generate']
    is_novelty_required = ns_raw_args['is_novelty_required']

    targeted_object = ns_raw_args['scene_details']['targeted_object']

    run_details = {
        'Evaluation function': evaluation_function.__name__,
        'Genotype length (genotype_len)': genotype_len,
        'Algorithm variant (qd_method)': qd_method.name,
        'Population size (pop_size)': pop_size,
        'Max number of individuals in the archive (archive_limit_size)': archive_limit_size,
        'Archive removal strategy when max size is reached (archive_limit_strat)': archive_limit_strat.name,
        'Mutation flag (mut_strat)': mut_strat.name,
        'Number of generated offspring per generation (nb_offsprings_to_generate)': nb_offsprings_to_generate,
        'Novelty required flag (is_novelty_required)': is_novelty_required,
        'Targeted object': targeted_object,
    }

    return run_details


def init_archive_kwargs(ns_raw_args):
    if ns_raw_args['archive_type'] == qd_cfg.ArchiveType.NOVELTY:
        archive_kwargs = {
            'archive_limit_size': qd_cfg.ARCHIVE_LIMIT_SIZE,
            'archive_limit_strat': ns_raw_args['archive_limit_size'],
            'pop_size': ns_raw_args['pop_size'],
        }
    elif ns_raw_args['archive_type'] == qd_cfg.ArchiveType.ELITE_STRUCTURED:
        archive_kwargs = {}
    elif ns_raw_args['archive_type'] == qd_cfg.ArchiveType.NONE:
        archive_kwargs = {}
    else:
        raise NotImplementedError()

    qd_method_cfg = qd_cfg.QD_METHODS_CONFIGS[ns_raw_args['qd_method']]

    archive_kwargs['fill_archive_strat'] = qd_method_cfg['fill_archive_strat']

    archive_kwargs['search_space_bb'] = ns_raw_args['search_space_bb']

    return archive_kwargs


def init_outcome_archive_kwargs(ns_raw_args):
    outcome_archive_kwargs = {
        'search_space_bb': ns_raw_args['search_space_bb'],
    }
    return outcome_archive_kwargs


def get_qd_algo_args(cfg, env, dump_folder_name):

    qd_method = cfg['algorithm']
    pop_size = cfg['evo_proc']['pop_size']
    n_budget_rollouts = cfg['evo_proc']['n_budget_rollouts']
    mut_strat = cfg['evo_proc']['mut_strat']
    prob_cx = cfg['evo_proc']['prob_cx']

    targeted_object = cfg['env']['kwargs']['object_name']
    target_name = cfg['env']['kwargs']['target_name']

    scene_details = {
        'targeted_object': targeted_object,
        'target_object': target_name,
        'use_concave_mesh' : cfg['env']['kwargs']['use_concave_mesh']
    }

    genotype_len = cfg['evo_proc']['qd_method_genotype_len']
    search_space_bb = cfg['evaluate']['search_space_bb']

    parallelize = cfg['parallelize']
    sigma_mut = cfg['evo_proc']['sigma_mut']
    select_off_strat = cfg['evo_proc']['select_off_strat']
    replace_pop_strat = cfg['evo_proc']['replace_pop_strat']
    archive_limit_strat = cfg['evo_proc']['archive_limit_strat']
    is_novelty_required = cfg['evo_proc']['is_novelty_required']
    archive_type = cfg['evo_proc']['archive_type']
    is_pop_based = cfg['evo_proc']['is_pop_based']
    include_invalid_inds = cfg['evo_proc']['include_invalid_inds']
    nb_offsprings_to_generate = int(pop_size * qd_cfg.OFFSPRING_NB_COEFF)

    eval_func = wrapped_partial(
        evaluate_placement_routine,
        env=env,
        eval_kwargs=cfg['evaluate']['kwargs'],
    )

    args = {
        'qd_method': qd_method,
        'pop_size': pop_size,
        'n_budget_rollouts': n_budget_rollouts,
        'mut_strat': mut_strat,
        'genotype_len': genotype_len,
        'scene_details': scene_details,
        'prob_cx': prob_cx,
        'search_space_bb': search_space_bb,
        'parallelize': parallelize,
        'sigma_mut': sigma_mut,
        'select_off_strat': select_off_strat,
        'replace_pop_strat': replace_pop_strat,
        'is_novelty_required': is_novelty_required,
        'is_pop_based': is_pop_based,
        'multiproc_pool': None,
        'nb_offsprings_to_generate': nb_offsprings_to_generate,
        'evaluation_function': eval_func,
        'archive_type': archive_type,
        'archive_limit_strat': archive_limit_strat,
        'include_invalid_inds':include_invalid_inds,
        'env':env,
    }

    args['run_details'] = init_run_details(args)
    args['archive_kwargs'] = init_archive_kwargs(args)
    args['outcome_archive_kwargs'] = init_outcome_archive_kwargs(args)
    args['run_name'] = dump_folder_name

    return args


def get_eval_kwargs(
        search_representation,
        domain_randomization_fitness
):
    eval_kwargs = {
        'search_representation': search_representation,
        'domain_randomization_fitness': domain_randomization_fitness,
    }
    return eval_kwargs


def get_env_kwargs(object_name, target_name, display, debug,use_concave_mesh):
    env_kwargs = {
        'object_name': object_name,
        'target_name': target_name,
        'display': display,
        'debug': debug,
        'use_concave_mesh': use_concave_mesh,
    }
    return env_kwargs


def get_parsed_input_arguments():
    args = parse_input_args()

    parsed_input_args = {
        'algorithm_str': args.algorithm,
        'display': args.display,
        'parallelize': args.parallelize,
        'sigma_mut': args.sigma_mut,
        'pop_size': args.population,
        'object': args.object,
        'target': args.target,
        'prob_cx': args.prob_cx,
        'log_path': args.log_path,
        'folder_name': args.folder_name,
        'n_budget_rollouts': args.n_budget_rollout,
        'domain_randomization_fitness': args.domain_randomization_fitness,
        'debug': args.debug,
        'include_invalid_inds': args.include_invalids,
        'use_concave_mesh': args.concave_mesh,
    }

    return parsed_input_args


def input_arguments_sanity_checks(processed_input_args):
    assert isinstance(processed_input_args['prob_cx'], float)
    assert 0. <= processed_input_args['prob_cx'] < 1.
    assert 0. <= processed_input_args['sigma_mut'] <= 1.

    if processed_input_args['display']:
        assert not processed_input_args['parallelize'], \
            'Cannot both display and parallelize computation. (choose either -ll or -d).'

def get_log_path(parsed_input_args) :

    log_path = Path(parsed_input_args["log_path"])

    # Créer le dossier si nécessaire
    if not log_path.exists():
        print(f"[INFO] Creating new log directory at {log_path}")
        log_path.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[INFO] Log directory already exists: {log_path}")

    # Remplace le chemin original par celui spécifique à l'objet
    return str(log_path)


def process_input_arguments(parsed_input_args):

    processed_input_args = parsed_input_args

    if processed_input_args['debug']:
        processed_input_args['display'] = True

    if not isinstance(parsed_input_args['sigma_mut'], float):
        processed_input_args['sigma_mut'] = float(parsed_input_args['sigma_mut'])

    processed_input_args['qd_method'] = qd_cfg.ALGO_ARG_TO_METHOD[parsed_input_args['algorithm_str']]
    del processed_input_args['algorithm_str']

    qd_meth_cfg = qd_cfg.QD_METHODS_CONFIGS[processed_input_args['qd_method']]
    processed_input_args['mut_strat'] = qd_meth_cfg['mutation_strat']
    processed_input_args['select_off_strat'] = qd_meth_cfg['select_off_strat']
    processed_input_args['replace_pop_strat'] = qd_meth_cfg['replace_pop_strat']
    processed_input_args['archive_limit_strat'] = qd_meth_cfg['archive_limit_strat']
    processed_input_args['is_novelty_required'] = qd_meth_cfg['is_novelty_required']
    processed_input_args['archive_type'] = qd_meth_cfg['archive_type']
    processed_input_args['is_pop_based'] = qd_meth_cfg['is_pop_based']
    processed_input_args['qd_method_genotype_len'] = qd_meth_cfg['genotype_len']
    processed_input_args['search_representation'] = qd_meth_cfg['search_representation']

    processed_input_args['env_kwargs'] = get_env_kwargs(
        object_name=processed_input_args['object'],
        target_name=processed_input_args['target'],
        display=processed_input_args['display'],
        debug=processed_input_args['debug'],
        use_concave_mesh=processed_input_args['use_concave_mesh']
    )
    processed_input_args['env_class'] = obj_cfg.OBJ_ENV_CLASS

    processed_input_args['eval_kwargs'] = get_eval_kwargs(
        search_representation=processed_input_args['search_representation'],
        domain_randomization_fitness=processed_input_args['domain_randomization_fitness']
    )
    processed_input_args['log_path'] = get_log_path(parsed_input_args)

    return processed_input_args


def get_input_arguments():

    parsed_input_args = get_parsed_input_arguments()

    processed_input_args = process_input_arguments(parsed_input_args)

    input_arguments_sanity_checks(processed_input_args)

    return processed_input_args

