import os.path

from sklearn.neighbors import NearestNeighbors as Nearest
import numpy as np
from pathlib import Path
import pdb
import algorithms.stats as stats
from utils.io_run_data import load_dict_pickle
import configs.eval_config as eval_cfg

import environments.src.env_constants as env_consts


N_DECIMAL_PRECISION = 2
OPTIMAL_FITNESS = 2.0
SHIFT_VALUE = 10

class StatsTracker:
    def __init__(self, qd_method):

        self.outcome_archive_len = None
        self._outcome_archive_cvg_hist = []  # coverage of As w.r.t. the number of evaluations
        self._success_archive_cvg_hist = []
        self._success_archive_qd_score_hist = []
        self._outcome_ratio_hist = []
        self._success_ratio_hist = []
        self._fitnesses_hist = []
        self._fitness_list = []
        self._scs_fits_hist = []
        self._scs_fits_list = []
        self._final_6dof_pos_list = []
        self._init_6dof_pos_list = []
        self.n_successful_cells_list = []
        self.n_invalid_inds = []

        self._n_evals = []  # number of evaluations corresponding to each export (each generation)
        self._run_time = []  # running time corresponding to each export (each generation)

        self.qd_method = None
        self._init_attributes(qd_method=qd_method)

        self._rolling_n_valid = 0  # number of evaluation considered in update() in which the object is touched
        self._rolling_n_success = 0  # number of evaluation considered in update() in which the object is grasped
        self._rolling_n_rollout = 0  # number of rollouts considered in update()

    def _init_attributes(self, qd_method):
        self.qd_method = qd_method

    def update(self, pop, outcome_archive, curr_n_evals, timer, timer_label):

        outcome_archive_cvg = self._get_outcome_archive_cvg(outcome_archive)
        success_archive_cvg = self._get_success_archive_cvg(outcome_archive)
        success_archive_qd_score = self._get_success_archive_qd_score(outcome_archive)
        outcome_ratio, success_ratio = self._get_outcome_and_success_ratios(pop, curr_n_evals)
        self.outcome_archive_len = len(outcome_archive)
        fitness = self._get_fitness_from_outvome_archive(outcome_archive)
        scs_fits = self._get_fitness_from_success_archive(outcome_archive)
        curr_run_time = timer.get_on_the_fly_time(label=timer_label)

        n_successful_cells = outcome_archive.get_n_successful_cells()
        self.n_successful_cells_list.append(n_successful_cells)
        self._outcome_archive_cvg_hist.append(outcome_archive_cvg)
        self._success_archive_cvg_hist.append(success_archive_cvg)
        self._success_archive_qd_score_hist.append(success_archive_qd_score)
        self._outcome_ratio_hist.append(outcome_ratio)
        self._success_ratio_hist.append(success_ratio)
        self._n_evals.append(curr_n_evals)
        self._run_time.append(curr_run_time)
        self._fitnesses_hist.append(fitness)
        self._scs_fits_hist.append(scs_fits)
        self._scs_fits_list = self._scs_fits_hist[-1]
        self._fitness_list = self._fitnesses_hist[-1]
        self._final_6dof_pos_list = self.get_final_6dof_pos(outcome_archive)
        self._init_6dof_pos_list = self.get_init_6dof_pos(outcome_archive)

        nbr_invalid_inds = outcome_archive.get_n_invalid_cells(pop)
        self.n_invalid_inds.append(nbr_invalid_inds)

    def _get_outcome_archive_cvg(self, outcome_archive):
        return len(outcome_archive) / outcome_archive.max_size    

    def _get_success_archive_cvg(self, outcome_archive):
        return outcome_archive.get_n_successful_cells() / outcome_archive.max_size

    def _get_success_archive_qd_score(self, outcome_archive):
        scs_fits = outcome_archive.get_successful_inds_fitnesses()
        scs_fits_shifted = [fit + SHIFT_VALUE for fit in scs_fits]
        return np.sum(fit for fit in scs_fits_shifted)

    def _get_fitness_from_outvome_archive(self, outcome_archive):
        fitness = outcome_archive.fits
        return fitness

    def _get_fitness_from_success_archive(self, outcome_archive):
        fitness = outcome_archive.get_successful_inds_fitnesses()
        return fitness
        
    def _get_outcome_and_success_ratios(self, pop, curr_n_evals):

        pop_n_valid = sum(info['is_valid'] for info in pop.infos)
        pop_n_success = sum(info['is_success'] for info in pop.infos)
        pop_n_rollout = len(pop)

        self._rolling_n_valid += pop_n_valid
        self._rolling_n_success += pop_n_success
        self._rolling_n_rollout += pop_n_rollout
        assert curr_n_evals == self._rolling_n_rollout  # might cause issues if called twice for a single gen ?

        outcome_ratio = self._rolling_n_valid / self._rolling_n_rollout
        success_ratio = self._rolling_n_success / self._rolling_n_rollout

        return outcome_ratio, success_ratio

    def get_fitness_list(self):
        return self._fitnesses_hist[-1]
    
    def get_final_6dof_pos(self, outcome_archive):
        infos = outcome_archive.infos
        return [info['final_6dof_pose'] for info in infos]

    def get_init_6dof_pos(self, outcome_archive):
        infos = outcome_archive.infos
        return [info['init_6dof_pose'] for info in infos]
    
    def ending_analysis(self, outcome_archive):
        pass
    
    def get_output_data(self):

        output_data_kwargs = {
            'qd_method': self.qd_method,
            'outcome_archive_cvg_hist': np.array(self._outcome_archive_cvg_hist),
            'success_archive_cvg_hist': np.array(self._success_archive_cvg_hist),
            'success_archive_qd_score_hist': np.array(self._success_archive_qd_score_hist),
            'outcome_ratio_hist': np.array(self._outcome_ratio_hist),
            'success_ratio_hist': np.array(self._success_ratio_hist),
            'n_evals_hist': np.array(self._n_evals),
            'run_time_hist': np.array(self._run_time),
            'fitness_hist' : np.array(self._scs_fits_hist, dtype=object),
            'fitness_list' : np.array(self._scs_fits_list),
            'final_6dof_pos' : np.array(self._final_6dof_pos_list),
            'init_6dof_pos' : np.array(self._init_6dof_pos_list),
            'n_successful_cells_list': np.array(self.n_successful_cells_list),
            'n_invalid_cells':np.array(self.n_invalid_inds),
        }
        return output_data_kwargs



