import pdb

import numpy as np
import random

import configs.eval_config as eval_cfg
from environments.src.search_space_bb_processor import get_bounding_box_diagonal_length
from .archive import Archive

SA_UNIT_VOXEL_SIDE_SIZE = 0.01
N_BINS = 10

class StructuredArchive(Archive):
    def __init__(self, search_space_bb, fill_archive_strat):
        super().__init__(fill_archive_strat=fill_archive_strat)

        self.max_roll, self.max_pitch, self.max_yaw = None, None, None
        self.min_roll, self.min_pitch, self.min_yaw = None, None, None   
        self.n_bins_roll, self.n_bins_pitch, self.n_bins_yaw = None, None, None  # sampling per component
        self.len_bin_roll, self.len_bin_pitch, self.len_bin_yaw = None, None, None  # bin len per component


        self.max_x, self.max_y, self.max_z = None, None, None
        self.min_x, self.min_y, self.min_z = None, None, None
        self.n_bins_x, self.n_bins_y, self.n_bins_z = None, None, None  # sampling per component
        self.len_bin_x, self.len_bin_y, self.len_bin_z = None, None, None  # bin len per component

        self.max_size = None

        self._all_bins_centroid_xyz_poses = None

        self._archive_type = 'structured_archive'  # used in fill_archive() for genericity

        self._map_inds = {}
        self._map_bds = {}
        self._map_fits = {}
        self._map_infos = {}

        self._init_hyperparameters(search_space_bb)


    @property
    def archive_type(self):
        return self._archive_type

    @property
    def inds(self):
        return [self._map_inds[key_map_ids] for key_map_ids in self._map_inds]

    @property
    def all_bins_centroid_xyz_poses(self):
        return self._all_bins_centroid_xyz_poses

    @property
    def fits(self):
        return [self._map_fits[key_map_ids] for key_map_ids in self._map_inds]
    
    @property    
    def infos(self):
        return [self._map_infos[key_map_ids] for key_map_ids in self._map_infos]

    def __len__(self):
        return len(self._map_inds)

    def _set_ind(self, key, ind, bds, fit, info):
        self._map_inds[key] = ind
        self._map_bds[key] = bds
        self._map_fits[key] = fit
        self._map_infos[key] = info


    def _init_hyperparameters(self, search_space_bb):

        self.max_x = search_space_bb.aabb_max[0]
        self.max_y = search_space_bb.aabb_max[1]
        self.max_z = search_space_bb.aabb_max[2]
        self.min_x = search_space_bb.aabb_min[0]
        self.min_y = search_space_bb.aabb_min[1]
        self.min_z = search_space_bb.aabb_min[2]

        self.n_bins_x = int((self.max_x - self.min_x) / SA_UNIT_VOXEL_SIDE_SIZE)
        self.n_bins_y = int((self.max_y - self.min_y) / SA_UNIT_VOXEL_SIDE_SIZE)
        self.n_bins_z = int((self.max_z - self.min_z) / SA_UNIT_VOXEL_SIDE_SIZE)

        self.len_bin_x = (self.max_x - self.min_x) / self.n_bins_x
        self.len_bin_y = (self.max_y - self.min_y) / self.n_bins_y
        self.len_bin_z = (self.max_z - self.min_z) / self.n_bins_z

        self.min_roll,self.max_roll = eval_cfg.FIXED_INTERVAL_EULER
        self.min_pitch, self.max_pitch = eval_cfg.FIXED_INTERVAL_EULER
        self.min_yaw, self.max_yaw = eval_cfg.FIXED_INTERVAL_EULER

        self.n_bins_roll = self.n_bins_pitch = self.n_bins_yaw = N_BINS

        self.len_bin_roll = (self.max_roll - self.min_roll) / self.n_bins_roll
        self.len_bin_pitch = (self.max_pitch - self.min_pitch) / self.n_bins_pitch
        self.len_bin_yaw = (self.max_yaw - self.min_yaw) / self.n_bins_yaw

        self.max_size = self.n_bins_x * self.n_bins_y * self.n_bins_z * self.n_bins_roll * self.n_bins_pitch * self.n_bins_yaw

        self._all_bins_centroid_xyz_poses = self.get_all_bins_centroid_xyz_poses()

    def _get_i_bin_bd(self, ind_bds):

        pos_x, pos_y, pos_z, roll, pitch, yaw = ind_bds

        i_bin_x = np.floor((pos_x - self.min_x) / self.len_bin_x)
        i_bin_y = np.floor((pos_y - self.min_y) / self.len_bin_y)
        i_bin_z = np.floor((pos_z - self.min_z) / self.len_bin_z)
        i_bin_roll= np.floor((roll - self.min_roll) / self.len_bin_roll)
        i_bin_pitch = np.floor((pitch - self.min_pitch) / self.len_bin_pitch)
        i_bin_yaw = np.floor((yaw - self.min_yaw) / self.len_bin_yaw)

        # corner cases in which the value is above the specified max -> set to last bin
        # i_bin_x = self.n_bins_x - 1 if i_bin_x > self.n_bins_x else i_bin_x
        # i_bin_y = self.n_bins_y - 1 if i_bin_y > self.n_bins_y else i_bin_y
        # i_bin_z = self.n_bins_z - 1 if i_bin_z > self.n_bins_z else i_bin_z
        # i_bin_roll = self.n_bins_roll - 1 if i_bin_roll >= self.n_bins_roll else i_bin_roll
        # i_bin_pitch = self.n_bins_pitch - 1 if i_bin_pitch >= self.n_bins_pitch else i_bin_pitch
        # i_bin_yaw = self.n_bins_yaw - 1 if i_bin_yaw >= self.n_bins_yaw else i_bin_yaw

        i_bin_x = int(np.clip(i_bin_x, 0, self.n_bins_x - 1))
        i_bin_y = int(np.clip(i_bin_y, 0, self.n_bins_y - 1))
        i_bin_z = int(np.clip(i_bin_z, 0, self.n_bins_z - 1))
        i_bin_roll = int(np.clip(i_bin_roll, 0, self.n_bins_roll - 1))
        i_bin_pitch = int(np.clip(i_bin_pitch, 0, self.n_bins_pitch - 1))
        i_bin_yaw = int(np.clip(i_bin_yaw, 0, self.n_bins_yaw - 1))

        i_bin_bd = (
            i_bin_x 
            + i_bin_y * self.n_bins_x 
            + i_bin_z * self.n_bins_x * self.n_bins_y 
            + i_bin_roll * self.n_bins_x * self.n_bins_y * self.n_bins_z 
            + i_bin_pitch * self.n_bins_x * self.n_bins_y * self.n_bins_z * self.n_bins_roll
            + i_bin_yaw * self.n_bins_x * self.n_bins_y * self.n_bins_z * self.n_bins_roll * self.n_bins_pitch
        )
        return int(i_bin_bd)


    def get_all_bins_centroid_xyz_poses(self):
        all_bins_centroid_xyz_poses = [
            [self.min_x + self.len_bin_x / 2 + i_bin_x * self.len_bin_x,
             self.min_y + self.len_bin_y / 2 + i_bin_y * self.len_bin_y,
             self.min_z + self.len_bin_z / 2 + i_bin_z * self.len_bin_z]
            for i_bin_x in range(self.n_bins_x)
            for i_bin_y in range(self.n_bins_y)
            for i_bin_z in range(self.n_bins_z)
        ]
        return all_bins_centroid_xyz_poses

    def select_success_based(self, n_sample, duplicate_scs_inds=False):

        if n_sample > len(self._map_inds):
            # pick all elements of Ao once
            all_inds_in_archive = [self._map_inds[key_ind_bin] for key_ind_bin in self._map_inds]
            n_missing_inds = n_sample - len(all_inds_in_archive)
            # fill the pop with random element of Ao
            additional_inds = [self._map_inds[random.sample(list(self._map_inds), 1)[0]] for _ in range(n_missing_inds)]
            selected_inds = all_inds_in_archive + additional_inds
            return selected_inds

        scs_inds = self.get_successful_inds()

        if len(scs_inds) >= n_sample:
            # randomly sample from successes
            selected_inds = random.sample(scs_inds, n_sample)

        elif len(scs_inds) > 0 and duplicate_scs_inds:
            # fill the population with duplicates of successful inds
            selected_inds = scs_inds
            n_scs = len(scs_inds)
            n_missing_inds = n_sample - n_scs

            n_repeat_inds = int(n_missing_inds / n_scs)
            repeated_inds = scs_inds * n_repeat_inds
            selected_inds += repeated_inds

            n_filling_inds = n_sample - len(selected_inds)
            filling_inds = scs_inds[:n_filling_inds]
            selected_inds += filling_inds

            if len(selected_inds) != n_sample:
                pdb.set_trace()

        else:
            # not success : fill with randomly sampled non-successful inds
            rand_sel_inds_keys = random.sample(list(self._map_inds), n_sample)
            rand_sel_inds = [self._map_inds[map_ind_key] for map_ind_key in rand_sel_inds_keys]
            selected_inds = rand_sel_inds

        assert len(selected_inds) == n_sample
        return selected_inds

    def select_fitness_based(self, n_sample):

        if n_sample > len(self._map_inds):
            # concatenation of a list of single ind
            selected_inds = [self._map_inds[random.sample(list(self._map_inds), 1)[0]] for _ in range(n_sample)]
            return selected_inds

        scs_inds, scs_bds, scs_fits, scs_infos = self.get_successful_inds_data()

        if len(scs_inds) >= n_sample:
            # select inds with higher fitness
            sorted_scs_ids = np.argsort(scs_fits)[::-1][:n_sample]
            selected_inds = scs_inds[sorted_scs_ids].tolist()

        else:
            # not enough success : fill with randomly sampled non-successful inds
            selected_inds = scs_inds.tolist()
            n_missing_inds = n_sample - len(selected_inds)

            rand_sel_inds_keys = random.sample(list(self._map_inds), n_missing_inds)
            rand_sel_inds = [self._map_inds[map_ind_key] for map_ind_key in rand_sel_inds_keys]

            selected_inds += rand_sel_inds

        assert len(selected_inds) == n_sample
        return selected_inds

    def manage_archive_size(self):
        pass  # empty function for compatibility: structured archive size is bounded by definition

    def get_successful_inds(self):
        return [
            self._map_inds[key_map_ids] for key_map_ids in self._map_inds
            if self._map_infos[key_map_ids]['is_success']
        ]


    def get_successful_inds_6dof_poses(self):
        return np.array(
            [
                [self._map_infos[key_map_ids][eval_cfg.XYZ_POSE_X_KEY_ID],
                 self._map_infos[key_map_ids][eval_cfg.XYZ_POSE_Y_KEY_ID],
                 self._map_infos[key_map_ids][eval_cfg.XYZ_POSE_Z_KEY_ID],

                 self._map_infos[key_map_ids][eval_cfg.XYZ_QUATERNION_1_KEY_ID],
                 self._map_infos[key_map_ids][eval_cfg.XYZ_QUATERNION_2_KEY_ID],
                 self._map_infos[key_map_ids][eval_cfg.XYZ_QUATERNION_3_KEY_ID],
                 self._map_infos[key_map_ids][eval_cfg.XYZ_QUATERNION_4_KEY_ID]]
                for key_map_ids in self._map_inds
                if self._map_infos[key_map_ids][eval_cfg.IS_SUCCESS_KEY_ID]
            ]
        )

    def get_successful_inds_xyz_poses(self):
        return np.array(
            [
                [self._map_infos[key_map_ids][eval_cfg.XYZ_POSE_X_KEY_ID],
                 self._map_infos[key_map_ids][eval_cfg.XYZ_POSE_Y_KEY_ID],
                 self._map_infos[key_map_ids][eval_cfg.XYZ_POSE_Z_KEY_ID]]
                for key_map_ids in self._map_inds
                if self._map_infos[key_map_ids][eval_cfg.IS_SUCCESS_KEY_ID]
            ]
        )

    def get_successful_inds_fitnesses(self):
        return np.array(
            [
                self._map_fits[key_map_ids]
                for key_map_ids in self._map_inds
                if self._map_infos[key_map_ids]['is_success']
            ]
        )

    def get_successful_inds_data(self):
        scs_data = [
            [self._map_inds[key_map_ids],
             self._map_bds[key_map_ids],
             self._map_fits[key_map_ids],
             self._map_infos[key_map_ids]]
            for key_map_ids in self._map_inds
            if self._map_infos[key_map_ids]['is_success']
        ]
        if len(scs_data) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # returns as inds, bds, fits, infos
        return map(np.array, zip(*scs_data))

    def get_inds_data(self):
        scs_data = [
            [self._map_inds[key_map_ids],
             self._map_bds[key_map_ids],
             self._map_fits[key_map_ids],
             self._map_infos[key_map_ids]]
            for key_map_ids in self._map_inds
        ]

        if len(scs_data) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # returns as inds, bds, fits, infos
        return map(np.array, zip(*scs_data))
    
    def random_sample(self, n_sample):
        """(Selection function) Randomly sample n_sample individuals from the current population."""

        if n_sample > len(self.inds):
            # concatenation of a list of single ind => sample_ind = [ind] --> sample_ind[0] --> ind
            selected_inds = [random.sample(self.inds, 1)[0] for i_miss in range(n_sample)]
            return selected_inds

        selected_inds = random.sample(self.inds, n_sample)  # already a list of inds

        return selected_inds

    def get_dims(self):
        return {
            'max_z':float(self.max_z),
            'max_y': float(self.max_y),
            'max_x': float(self.max_x),
            'min_z': float(self.min_z),
            'min_y': float(self.min_y),
            'min_x': float(self.min_x),
            'n_bins_z': float(self.n_bins_z),
            'n_bins_y': float(self.n_bins_y),
            'n_bins_x': float(self.n_bins_x),
            'len_bin_z': float(self.len_bin_z),
            'len_bin_y': float(self.len_bin_y),
            'len_bin_x': float(self.len_bin_x),
            'max_z':float(self.max_z),
            'max_roll': float(self.max_roll),
            'max_pitch': float(self.max_pitch),
            'max_yaw': float(self.max_yaw),
            'min_z': float(self.min_z),
            'min_roll': float(self.min_roll),
            'min_pitch': float(self.min_pitch),
            'min_yaw': float(self.min_yaw),
            'n_bins_z': float(self.n_bins_z),
            'n_bins_roll': float(self.n_bins_roll),
            'n_bins_pitch': float(self.n_bins_pitch),
            'n_bins_yaw': float(self.n_bins_yaw),
            'len_bin_z': float(self.len_bin_z),
            'len_bin_roll': float(self.len_bin_roll),
            'len_bin_pitch': float(self.len_bin_pitch),
            'len_bin_yaw': float(self.len_bin_yaw),
        }