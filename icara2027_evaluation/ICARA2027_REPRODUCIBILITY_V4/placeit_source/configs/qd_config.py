
from enum import Enum


########################################################################################################################
#  CONSTANTS
########################################################################################################################


# -------------------------------------------------------------------------------------------------------------------- #
# GENOME
# -------------------------------------------------------------------------------------------------------------------- #

GENOTYPE_MAX_VAL = 1.
FIXED_INTERVAL_GENOME = [-GENOTYPE_MAX_VAL, GENOTYPE_MAX_VAL]


# -------------------------------------------------------------------------------------------------------------------- #
# GENOME LENGTH
# -------------------------------------------------------------------------------------------------------------------- #

GENOTYPE_LEN_6DOF_POSE = 6

# Approach variants list-based
GENOTYPE_LEN_CONTACT_POINT_FINDER_LIST = 1

GENOTYPE_PCA_ALIGNMENT_LIST = 2

GENOTYPE_FACE_ALIGNMENT_LIST = 2


# -------------------------------------------------------------------------------------------------------------------- #
# OFFSPRING
# -------------------------------------------------------------------------------------------------------------------- #

OFFSPRING_NB_COEFF = 1.  # number of offsprings generated (coeff of pop length)


# -------------------------------------------------------------------------------------------------------------------- #
# ARCHIVE
# -------------------------------------------------------------------------------------------------------------------- #

ARCHIVE_LIMIT_SIZE = 25000
ARCHIVE_DECREMENTAL_RATIO = 0.9  # if archive size is bigger than thresh, cut down archive by this ratio


# -------------------------------------------------------------------------------------------------------------------- #
# RUNNING DATA SAVING
# -------------------------------------------------------------------------------------------------------------------- #

DUMP_SCS_ARCHIVE_ON_THE_FLY = False  # Useful for backup if the run is interrupted ; not crucial for fast exec
N_GEN_FREQ_DUMP_SCS_ARCHIVE = 10


# -------------------------------------------------------------------------------------------------------------------- #
# TIMER
# -------------------------------------------------------------------------------------------------------------------- #

QD_RUN_TIME_LABEL = 'run_qd'


########################################################################################################################
#  STRATEGIES
########################################################################################################################


# -------------------------------------------------------------------------------------------------------------------- #
#  ARCHIVE MANAGEMENT STRATEGIES
# -------------------------------------------------------------------------------------------------------------------- #

class ArchiveType(Enum):
    ELITE_STRUCTURED = 1
    NOVELTY = 2
    NONE = 3


# -------------------------------------------------------------------------------------------------------------------- #
#  ARCHIVE MANAGEMENT STRATEGIES
# -------------------------------------------------------------------------------------------------------------------- #


class FillArchiveStrategy(Enum):
    NOVELTY_BASED = 1
    STRUCTURED_ELITES = 2
    NONE = 3


# -------------------------------------------------------------------------------------------------------------------- #
#  POPULATION MANAGEMENT STRATEGIES
# -------------------------------------------------------------------------------------------------------------------- #

class ReplacePopulationStrategy(Enum):
    RANDOM = 1
    NOVELTY_BASED = 2
    RESET_FROM_SCRATCH = 3


# -------------------------------------------------------------------------------------------------------------------- #
#  MUTATION STRATEGIES
# -------------------------------------------------------------------------------------------------------------------- #


class MutationStrategy(Enum):
    GAUSS = 1
    NONE = 2

# -------------------------------------------------------------------------------------------------------------------- #
#  SELECT OFFSPRING STRATEGIES
# -------------------------------------------------------------------------------------------------------------------- #


class SelectOffspringStrategy(Enum):
    RANDOM_FROM_POP = 1
    RANDOM_FROM_ARCHIVE = 2
    FITNESS_FROM_ARCHIVE = 3
    SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE = 4
    SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE_WITH_DUPLICATES = 5


# -------------------------------------------------------------------------------------------------------------------- #
# METHODS
# -------------------------------------------------------------------------------------------------------------------- #

class SupportedMethod(Enum):
    RAND_SAMPLE = 1
    ME_RAND = 2
    ME_SCS = 3
    CMA_MAE = 4
    PLACEIT_ME_RAND = 5
    PLACEIT_ME_SCS = 6
    PLACEIT_CMA_MAE = 7
    RAND_PLACEIT = 8
    PCA_ALIGNMENT = 9
    FACE_ALIGNMENT = 10
    PLACEIT_RAND_SAMPLE = 11




# -------------------------------------------------------------------------------------------------------------------- #
# ARCHIVE LIMIT STRATEGY
# -------------------------------------------------------------------------------------------------------------------- #


class ArchiveLimitStrategy(Enum):
    RANDOM = 1


# -------------------------------------------------------------------------------------------------------------------- #
# SEARCH SPACE REPRESENTATION
# -------------------------------------------------------------------------------------------------------------------- #

class SearchSpaceRepresentation(Enum):
    SPHERICAL = 1
    CONTACT_STRATEGY_CONTACT_POINT_FINDER = 2
    APPROACH_STRATEGY_CONTACT_POINT_FINDER = 3
    PCA_ALIGNMENT_STRATEGY  = 4 
    FACE_ALIGNMENT_STRATEGY = 5
    CARTESIAN = 6


########################################################################################################################
#  METHOD SETUPS
########################################################################################################################

# Input argument strings
ALGO_ARG_TO_METHOD = {
    'rand_sample': SupportedMethod.RAND_SAMPLE,

    'me_scs': SupportedMethod.ME_SCS,

    'me_rand': SupportedMethod.ME_RAND,

    'cma_mae': SupportedMethod.CMA_MAE,

    'placeit_me_rand' : SupportedMethod.PLACEIT_ME_RAND,

    'placeit_me_scs' : SupportedMethod.PLACEIT_ME_SCS,

    'placeit_cma_mae': SupportedMethod.PLACEIT_CMA_MAE,

    'rand_placeit' : SupportedMethod.RAND_PLACEIT,

    'pca_alignment' : SupportedMethod.PCA_ALIGNMENT,

    'face_alignment' : SupportedMethod.FACE_ALIGNMENT,

    'placeit_rand_sample' : SupportedMethod.PLACEIT_RAND_SAMPLE,
}

# Configs associated with each method

QD_METHODS_CONFIGS = {
    SupportedMethod.RAND_SAMPLE:{
        'archive_type': ArchiveType.NONE,
        'fill_archive_strat': FillArchiveStrategy.NONE,
        'mutation_strat': MutationStrategy.NONE,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_POP,
        'replace_pop_strat': ReplacePopulationStrategy.RESET_FROM_SCRATCH,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': True,
        'genotype_len': GENOTYPE_LEN_6DOF_POSE,
        'search_representation': SearchSpaceRepresentation.SPHERICAL,
    },
    SupportedMethod.PCA_ALIGNMENT:{
        'archive_type': ArchiveType.NONE,
        'fill_archive_strat': FillArchiveStrategy.NONE,
        'mutation_strat': MutationStrategy.NONE,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_POP,
        'replace_pop_strat': ReplacePopulationStrategy.RESET_FROM_SCRATCH,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': True,
        'genotype_len': GENOTYPE_PCA_ALIGNMENT_LIST,
        'search_representation': SearchSpaceRepresentation.PCA_ALIGNMENT_STRATEGY,
    },
    SupportedMethod.FACE_ALIGNMENT:{
        'archive_type': ArchiveType.NONE,
        'fill_archive_strat': FillArchiveStrategy.NONE,
        'mutation_strat': MutationStrategy.NONE,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_POP,
        'replace_pop_strat': ReplacePopulationStrategy.RESET_FROM_SCRATCH,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': True,
        'genotype_len': GENOTYPE_FACE_ALIGNMENT_LIST,
        'search_representation': SearchSpaceRepresentation.FACE_ALIGNMENT_STRATEGY,
    },
    SupportedMethod.ME_RAND: {
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'genotype_len': GENOTYPE_LEN_6DOF_POSE,
        'search_representation': SearchSpaceRepresentation.SPHERICAL,
    },
    SupportedMethod.ME_SCS:{
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'genotype_len': GENOTYPE_LEN_6DOF_POSE,
        'search_representation': SearchSpaceRepresentation.SPHERICAL,
    },
    SupportedMethod.CMA_MAE: {
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'genotype_len': GENOTYPE_LEN_6DOF_POSE,
        'search_representation': SearchSpaceRepresentation.SPHERICAL,
    },
    SupportedMethod.PLACEIT_RAND_SAMPLE:{
        'archive_type': ArchiveType.NONE,
        'fill_archive_strat': FillArchiveStrategy.NONE,
        'mutation_strat': MutationStrategy.NONE,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_POP,
        'replace_pop_strat': ReplacePopulationStrategy.RESET_FROM_SCRATCH,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': True,
        'genotype_len': GENOTYPE_LEN_6DOF_POSE + GENOTYPE_LEN_CONTACT_POINT_FINDER_LIST,
        'search_representation': SearchSpaceRepresentation.APPROACH_STRATEGY_CONTACT_POINT_FINDER,
    },
    SupportedMethod.PLACEIT_ME_RAND: {
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'genotype_len': GENOTYPE_LEN_6DOF_POSE + GENOTYPE_LEN_CONTACT_POINT_FINDER_LIST,
        'search_representation': SearchSpaceRepresentation.APPROACH_STRATEGY_CONTACT_POINT_FINDER,
    },
    SupportedMethod.PLACEIT_ME_SCS:{
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'genotype_len': GENOTYPE_LEN_6DOF_POSE + GENOTYPE_LEN_CONTACT_POINT_FINDER_LIST,
        'search_representation': SearchSpaceRepresentation.APPROACH_STRATEGY_CONTACT_POINT_FINDER,
    },
    SupportedMethod.PLACEIT_CMA_MAE: {
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'genotype_len': GENOTYPE_LEN_6DOF_POSE + GENOTYPE_LEN_CONTACT_POINT_FINDER_LIST,
        'search_representation': SearchSpaceRepresentation.APPROACH_STRATEGY_CONTACT_POINT_FINDER,
    },
    SupportedMethod.RAND_PLACEIT:{
        'archive_type': ArchiveType.NONE,
        'fill_archive_strat': FillArchiveStrategy.NONE,
        'mutation_strat': MutationStrategy.NONE,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_POP,
        'replace_pop_strat': ReplacePopulationStrategy.RESET_FROM_SCRATCH,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': True,
        'genotype_len': GENOTYPE_LEN_6DOF_POSE + GENOTYPE_LEN_CONTACT_POINT_FINDER_LIST,
        'search_representation': SearchSpaceRepresentation.APPROACH_STRATEGY_CONTACT_POINT_FINDER,
    },
}

# -------------------------------------------------------------------------------------------------------------------- #
# PYRIBS INTERFACE
# -------------------------------------------------------------------------------------------------------------------- #

# Pyribs qd methods
CMA_ME_QD_METHODS = []
CMA_ES_QD_METHODS = []
CMA_MAE_QD_METHODS = [SupportedMethod.CMA_MAE]

PYRIBS_QD_METHODS = CMA_ME_QD_METHODS + CMA_ES_QD_METHODS + CMA_MAE_QD_METHODS

# Pyribs learning rates
CMA_MAE_PREDEFINED_ALPHA = 0.5
CMA_ME_PREDEFINED_ALPHA = 1.0
CMA_ES_PREDEFINED_ALPHA = 0.0

# Pyribs hyperparameters (similar to cma_mae paper)
CMA_MAE_EMITTER_BATCH_SIZE = 36
CMA_MAE_N_EMITTERS = 15
CMA_MAE_POP_SIZE = CMA_MAE_EMITTER_BATCH_SIZE * CMA_MAE_N_EMITTERS














