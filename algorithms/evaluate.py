import numpy as np
import time

import environments.src.genome_to_6dof_poses as cvt_gen_6dof
import configs.qd_config as qd_cfg
import configs.eval_config as eval_cfg
import pdb

INVALID_POSE_FITNESS = -1000000.0
INVALID_POSE_BEHAVIOR = [0., 0., 0., 0., 0., 0.]
INVALID_6DOF_INIT_POSE =((0., 0., 0.),( 0., 0., 0.))
INVALID_6DOF_FINAL_POSE = ((0., 0., 0.),( 0., 0., 0.))

def evaluate_placement_routine(individual, env, eval_kwargs):

    raise_exceptions_flg = True
    try:
        behavior, fitness, infos = evaluate_placement(
            individual=individual,
            env=env,
            eval_kwargs=eval_kwargs,
        )
    except Exception as e:
        # Might be raised in some pathological cases due to simulator issues
        if raise_exceptions_flg:
            raise e
        behavior, fitness = exception_handler_evaluate_placement_ind(
            individual=individual, eval_kwargs=eval_kwargs,
        )

    return behavior, fitness, infos


def evaluate_placement(individual, env, eval_kwargs):

    obj_6dof_init_pose = cvt_genome_to_6dof_obj_pose(individual=individual, env=env, eval_kwargs=eval_kwargs)

    if obj_6dof_init_pose is None:
        return invalid_6dof_pose_evaluation_outcome()
    
    env.reset()
    eval_6dof_pose_kwargs = {
        'env': env,
        'obj_6dof_pose': obj_6dof_init_pose,
    }

    object_output_data = evaluate_6dof_obj_pose(**eval_6dof_pose_kwargs)

    if env.is_debug_mode():
        time.sleep(2)
        env.delete_debug_bodies()

    is_stable = object_output_data['is_stable']
    is_valid = object_output_data['is_valid']
    obj_trajectory_data = object_output_data['trajectory_data']

    is_success = is_stable

    if not is_valid or obj_trajectory_data is None :
        return invalid_6dof_pose_evaluation_outcome()


    fitness = get_fitness(obj_trajectory_data)

    obj_6dof_final_pose = obj_trajectory_data[-1]
    grasp_proxy = get_grasp_proxy(obj_6dof_final_pose)

    final_pose_dict = {
        'xyz': list(obj_6dof_final_pose[0]),
        'euler_rpy': list(obj_6dof_final_pose[1]),
        'quaternions': None
    }


    infos = {
        'init_6dof_pose': obj_6dof_init_pose,
        'final_6dof_pose': final_pose_dict,
        'is_success': float(is_success),
        'is_valid': float(is_valid),
        'grasp_proxy': float(grasp_proxy),
    }

    xyz_final_pos = final_pose_dict['xyz']
    euler_final_rpy = final_pose_dict['euler_rpy']
    behavior = xyz_final_pos + euler_final_rpy
    return behavior, fitness, infos


def get_fitness(obj_trajectory_data):

    if obj_trajectory_data is None :
        return INVALID_POSE_FITNESS

    positions = [pos for pos, _ in obj_trajectory_data]
    orientations = [orient for _, orient in obj_trajectory_data]

    positions = np.array(positions)
    orientations = np.array(orientations)

    # Calcul de la variance
    pos_variance = np.var(positions)
    orient_variance = np.var(orientations)

    fitness = - (pos_variance + orient_variance)

    return fitness


def get_grasp_proxy(obj_6dof_final_pose):
    """
    Grasp accessibility proxy.
    Higher = easier top-down grasp.
    """

    xyz = np.array(obj_6dof_final_pose[0])
    rpy = np.array(obj_6dof_final_pose[1])

    roll, pitch, yaw = rpy

    height_score = xyz[2]

    # Distance from table center
    center_distance = np.linalg.norm(xyz[:2])

    # Uprightness 
    tilt = np.sqrt(roll**2 + pitch**2)

    # Bounded scores
    center_score = np.exp(-center_distance / 0.1)
    upright_score = np.exp(-tilt)

    grasp_proxy = (
        0.5 * height_score +
        0.3 * center_score +
        0.2 * upright_score
    )

    return grasp_proxy

def cvt_genome_to_6dof_obj_pose(individual, env, eval_kwargs):
    search_representation = eval_kwargs['search_representation']

    if search_representation == qd_cfg.SearchSpaceRepresentation.SPHERICAL:
        object_6dof_pose = cvt_gen_6dof.cvt_genome_to_6dof_pose_spherical(genome=individual,env=env)

    elif search_representation == qd_cfg.SearchSpaceRepresentation.APPROACH_STRATEGY_CONTACT_POINT_FINDER:
        object_6dof_pose = cvt_gen_6dof.cvt_genome_to_6dof_pose_approach_strat_contact_point_finder_search(genome=individual,env=env)

    elif search_representation == qd_cfg.SearchSpaceRepresentation.PCA_ALIGNMENT_STRATEGY:
        object_6dof_pose = cvt_gen_6dof.cvt_genome_to_6dof_pose_strat_pca_alignment(genome=individual,env=env)

    elif search_representation == qd_cfg.SearchSpaceRepresentation.FACE_ALIGNMENT_STRATEGY:
        object_6dof_pose = cvt_gen_6dof.cvt_genome_to_6dof_pose_strat_face_alignment(genome=individual,env=env)   

    elif search_representation == qd_cfg.SearchSpaceRepresentation.CARTESIAN:
        object_6dof_pose = cvt_gen_6dof.cvt_genome_to_6dof_pose_cartesian(genome=individual,env=env)   

    else:
        raise NotImplementedError('error in search representation type')

    return object_6dof_pose

def evaluate_6dof_obj_pose(env, obj_6dof_pose, domain_randomization_args=None):
    
    env.set_6dof_obj_pose(obj_6dof_pose)   

    is_there_overlap = env.is_there_overlapping()

    if is_there_overlap:
        return get_overlap_output_dict()
    
    trajectory_data = env.drop_object()

    is_stable = is_object_stable(env, trajectory_data)

    object_output_dict = {
        'is_stable': is_stable,
        'is_valid': not is_there_overlap,
        'is_overlap': is_there_overlap,
        'trajectory_data': trajectory_data
    }
    return object_output_dict

def get_overlap_output_dict():
    return {
        'is_stable': False,
        'is_valid': False,
        'is_overlap': True,
        'trajectory_data': None,
    }

def init_output_dict():
    object_output_data = {
        'is_stable': None,
        'is_valid': None,
        'is_overlap': None,
        'trajectory_data': None,
    }
    return object_output_data

def is_object_stable(env, trajectory_data):
    is_stable = True

    if not trajectory_data :
        return not is_stable

    is_contact = env.is_there_contacts()

    if not is_contact :
        return not is_stable
    
    last_poses = trajectory_data[-10:]

    fitness = - get_fitness(last_poses)

    if fitness < 1 :
        is_stable = True

    return is_stable        

def invalid_6dof_pose_evaluation_outcome():
    invalid_pose_behavior = INVALID_POSE_BEHAVIOR
    invalid_pose_fitness = INVALID_POSE_FITNESS

    invalid_pose_info = {
        'init_6dof_pose': INVALID_6DOF_INIT_POSE,
        'final_6dof_pose': INVALID_6DOF_FINAL_POSE,
        'is_success': float(False),
        'is_valid': float(False),
    }

    return invalid_pose_behavior, invalid_pose_fitness, invalid_pose_info


def exception_handler_evaluate_placement_ind(individual, eval_kwargs):
    raise NotImplementedError()
