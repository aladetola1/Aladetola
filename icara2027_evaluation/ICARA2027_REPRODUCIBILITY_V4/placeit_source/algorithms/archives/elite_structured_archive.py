import pdb
from .structured_archive import StructuredArchive
import configs.eval_config as eval_cfg
import random

class EliteStructuredArchive(StructuredArchive):
    def __init__(self, search_space_bb, fill_archive_strat):
        super().__init__(search_space_bb=search_space_bb, fill_archive_strat=fill_archive_strat)

    def fill_elites(self, pop):
        """Fill the archive based on the fitnesses of individuals."""

        zipped_ind_data = zip(pop.inds, pop.bds, pop.fits, pop.infos)

        for id_ind, (ind_candidate, ind_bds, ind_fit, ind_infos) in enumerate(zipped_ind_data):
 
            candidate_is_valid = ind_infos['is_valid']

            if not candidate_is_valid :
                continue
            
            all_i_bins = [self._get_i_bin_bd(ind_bds)]
            key_map_ids = tuple(all_i_bins)

            is_cell_empty = key_map_ids not in self._map_inds
            if is_cell_empty:
                self._set_ind(key=key_map_ids, ind=ind_candidate, bds=ind_bds, fit=ind_fit, info=ind_infos)

            else:
                ind_niche_infos = self._map_infos[key_map_ids]

                candidate_is_scs = ind_infos['is_success']
                niche_is_scs = ind_niche_infos['is_success']

                if not candidate_is_scs and not niche_is_scs:
                    if random.random() > 0.5:
                        self._set_ind(key=key_map_ids, ind=ind_candidate, bds=ind_bds, fit=ind_fit, info=ind_infos)

                elif not candidate_is_scs and niche_is_scs:
                    pass

                elif candidate_is_scs and not niche_is_scs:
                    self._set_ind(key=key_map_ids, ind=ind_candidate, bds=ind_bds, fit=ind_fit, info=ind_infos)

                else :
                    assert candidate_is_scs and niche_is_scs
                    candidate_fitness = ind_fit
                    niche_fitness = self._map_fits[key_map_ids]
                    if candidate_fitness > niche_fitness:
                        self._set_ind(key=key_map_ids, ind=ind_candidate, bds=ind_bds, fit=ind_fit, info=ind_infos)

        return