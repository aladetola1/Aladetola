
from pathlib import Path
import glob
import yaml
import numpy as np
import pdb

from utils.io_run_data import load_dict_pickle

import configs.obj_config as obj_cfg

N_SCS_RUN_DATA_KEY = 'Number of successful individuals (archive_success len)'
CONFIG_PKL_FILE_NAME = 'config'
SCS_ARCHIVE_INDS_KEY = 'inds'
IS_SCS_ARCHIVE_INDS_KEY = 'is_scs_inds'
SCS_ARCHIVE_BEHAVIOR_KEY = 'behavior_descriptors'
SCS_ARCHIVE_FITNESS_KEY = 'fitnesses'
SCS_ARCHIVE_IS_ROBUST_POSE_FLAG_KEY = 'is_robust_pose_flag'
SCS_ARCHIVE_INFOS_KEY = 'infos'
SCS_ARCHIVE_INFO_KEYS_KEY = 'infos_keys'
SCS_ARCHIVE_ROBUSTNESS_INFOS_KEY = 'robustness_fitness'

def get_last_dump_ind_file(folder):
    ind_root_path = folder/"success_archives"
    inds_folders = [path for path in glob.glob(f'{ind_root_path}/individuals_*.npz')]
    np_ext_str = '.npz'
    n_char2skip = len(np_ext_str)
    all_ids = [int(ind_f.split("/")[-1].split("_")[-1][:-n_char2skip]) for ind_f in inds_folders]
    id_last_dump = np.argmax(all_ids)
    last_dump_ind_file = inds_folders[id_last_dump]
    return last_dump_ind_file


def get_specific_dump_ind_file(folder, id):
    ind_root_path = folder/"success_archives"
    inds_folders = [path for path in glob.glob(f'{ind_root_path}/individuals_*.npz')]
    np_ext_str = '.npz'
    n_char2skip = len(np_ext_str)
    all_ids = [int(ind_f.split("/")[-1].split("_")[-1][:-n_char2skip]) for ind_f in inds_folders]
    if id not in all_ids:
        id_last_dump = np.argmax(all_ids)
        last_dump_ind_file = inds_folders[id_last_dump]
        queried_dump_ind_file = last_dump_ind_file
        print(f'id={id} not in inds for folder={folder}, taking last value = {id_last_dump}')
    else:
        id_in_ind_list = all_ids.index(id)
        queried_dump_ind_file = inds_folders[id_in_ind_list]
        print('found')
    return queried_dump_ind_file


def get_last_dump_ind_file_qd(folder):
    ind_root_path = folder/"success_archives_qd"
    inds_folders = [path for path in glob.glob(f'{ind_root_path}/individuals_*.npz')]
    np_ext_str = '.npz'
    n_char2skip = len(np_ext_str)
    all_ids = [int(ind_f.split("/")[-1].split("_")[-1][:-n_char2skip]) for ind_f in inds_folders]
    id_last_dump = np.argmax(all_ids)
    last_dump_ind_file = inds_folders[id_last_dump]
    return last_dump_ind_file


def get_folder_path_from_str(run_folder_path):
    folder_path = Path(run_folder_path)
    if len(list(folder_path.glob("**/run_details*.yaml"))) == 0:
        raise FileNotFoundError("No run found")
    return folder_path


def load_run_output_files(folder):

    run_details_path = next(folder.glob("**/run_details*.yaml"))
    run_infos_path = next(folder.glob("**/run_infos*.yaml"))

    with open(run_details_path, "r") as f:
        run_details = yaml.safe_load(f)

    with open(run_infos_path, "r") as f:
        run_infos = yaml.safe_load(f)

    with open(folder / "config.pkl", "r") as f:
        path2file = folder / f'{CONFIG_PKL_FILE_NAME}.pkl'
        cfg = load_dict_pickle(file_path=path2file)

    return run_details, run_infos, cfg


def load_all_inds(folder, individuals_id=None):
    if individuals_id is not None:
        ind_file = get_specific_dump_ind_file(folder, id=individuals_id)
    else:
        ind_file = get_last_dump_ind_file(folder)
    

    return np.load(ind_file)[SCS_ARCHIVE_INDS_KEY]

def load_all_behavior_descriptors(folder):
    ind_file = get_last_dump_ind_file(folder)
    return np.load(ind_file, allow_pickle=True)[SCS_ARCHIVE_BEHAVIOR_KEY]


def load_all_fitnesses(folder, individuals_id=None):

    if individuals_id is not None:
        ind_file = get_specific_dump_ind_file(folder, id=individuals_id)
    else:
        ind_file = get_last_dump_ind_file(folder)

    return np.load(ind_file, allow_pickle=True)[SCS_ARCHIVE_FITNESS_KEY]


def load_all_infos(folder, individuals_id=None):

    if individuals_id is not None:
        ind_file = get_specific_dump_ind_file(folder, id=individuals_id)
    else:
        ind_file = get_last_dump_ind_file(folder)

    info = np.load(ind_file, allow_pickle=True)[SCS_ARCHIVE_INFOS_KEY]
    return info

def load_all_flags(folder, individuals_id=None):

    if individuals_id is not None:
        ind_file = get_specific_dump_ind_file(folder, id=individuals_id)
    else:
        ind_file = get_last_dump_ind_file(folder)

    data = np.load(ind_file, allow_pickle=True)

    assert SCS_ARCHIVE_IS_ROBUST_POSE_FLAG_KEY in data, f"Key '{SCS_ARCHIVE_IS_ROBUST_POSE_FLAG_KEY}' non found in : {ind_file}"
    
    is_robust_pose_flag = np.load(ind_file, allow_pickle=True)[SCS_ARCHIVE_IS_ROBUST_POSE_FLAG_KEY]
    
    return is_robust_pose_flag

def load_all_robustness_infos(folder, individuals_id=None):

    if individuals_id is not None:
        ind_file = get_specific_dump_ind_file(folder, id=individuals_id)
    else:
        ind_file = get_last_dump_ind_file(folder)

    data = np.load(ind_file, allow_pickle=True)

    assert SCS_ARCHIVE_ROBUSTNESS_INFOS_KEY in data, f"Key '{SCS_ARCHIVE_ROBUSTNESS_INFOS_KEY}' non found in : {ind_file}"
    
    robustness_infos = np.load(ind_file, allow_pickle=True)[SCS_ARCHIVE_ROBUSTNESS_INFOS_KEY]
    
    return robustness_infos

def get_info_from_key(infos, info_keys, key):
    id_key = np.where(info_keys == key)[0][0]
    return infos[id_key]


def init_env(cfg, display=False):
    env_class = obj_cfg.OBJ_ENV_CLASS
    env_kwargs = cfg['env']['kwargs']
    env_kwargs['display'] = display

    return env_class(**env_kwargs)

def get_all_6dof_poses_from_infos(infos):
    poses = []
    for row in infos:
        init_pose =row['init_6dof_pose']
        pos = init_pose['xyz']
        rpy = init_pose['euler_rpy']
        full_pose = pos + rpy  
        poses.append(full_pose)

    all_6dof_poses = np.array(poses)

    return all_6dof_poses

def get_final_6dof_pose_from_infos(infos):
    poses = []
    for row in infos:
        init_pose =row['final_6dof_pose']
        pos = init_pose['xyz']
        rpy = init_pose['euler_rpy']
        full_pose = pos + rpy  
        poses.append(full_pose)

    all_6dof_poses = np.array(poses)

    return all_6dof_poses

def get_all_gripper_6dof_pose_from_infos(infos):
    targeted_keys = ['xyz_pose_x', 'xyz_pose_y', 'xyz_pose_z', 'quat_1', 'quat_2', 'quat_3', 'quat_4']
    info_keys = [0,0]
    assert len(set(targeted_keys).intersection(set(info_keys))) == len(targeted_keys), \
        'Infos keys are not valid for extracting 6dof pose.'

    assert isinstance(info_keys, np.ndarray)
    info_keys_list = info_keys.tolist()

    id_xyz_pose_x = info_keys_list.index("xyz_pose_x")
    id_xyz_pose_y = info_keys_list.index("xyz_pose_y")
    id_xyz_pose_z = info_keys_list.index("xyz_pose_z")
    id_quat_1 = info_keys_list.index("quat_1")
    id_quat_2 = info_keys_list.index("quat_2")
    id_quat_3 = info_keys_list.index("quat_3")
    id_quat_4 = info_keys_list.index("quat_4")

    xyz_pose_x = infos[:, id_xyz_pose_x]
    xyz_pose_y = infos[:, id_xyz_pose_y]
    xyz_pose_z = infos[:, id_xyz_pose_z]
    quat_1 = infos[:, id_quat_1]
    quat_2 = infos[:, id_quat_2]
    quat_3 = infos[:, id_quat_3]
    quat_4 = infos[:, id_quat_4]

    n_inds = infos.shape[0]
    all_gripper_6dof_pose = [
        {
            'xyz': [xyz_pose_x[i_ind], xyz_pose_y[i_ind], xyz_pose_z[i_ind]],
            'euler_rpy': None,
            'quaternions': [quat_1[i_ind], quat_2[i_ind], quat_3[i_ind], quat_4[i_ind]]
        }
        for i_ind in range(n_inds)
    ]

    return all_gripper_6dof_pose
