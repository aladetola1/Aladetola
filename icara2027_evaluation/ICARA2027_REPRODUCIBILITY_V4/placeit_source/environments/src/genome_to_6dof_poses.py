
import numpy as np
import random
from utils.common_tools import project_from_to_value
from environments.src.search_space_bb_processor import get_bounding_box_diagonal_length
from algorithms.evaluation.pose_strategies_routines import get_normal_surface_point, get_obj_pose_relatively_to_contact_point, align_objects_by_axis
import pdb
from scipy.spatial.transform import Rotation
import configs.qd_config as qd_cfg
import configs.eval_config as eval_cfg
import warnings

EXTREMITY_WIRST = 0.112

def cvt_genome_to_6dof_pose_cartesian(genome, env):
    object_pos_xyz_genome = genome[:3]
    object_orient_rpy_genome = genome[3:6]

    aabb_max = env.sim_engine.search_space_bb.aabb_max
    aabb_min = env.sim_engine.search_space_bb.aabb_min

    object_pos_xyz = [
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[aabb_min[i_val], aabb_max[i_val]],
            x_start=val
        ) for i_val, val in enumerate(object_pos_xyz_genome)
    ]
    object_orient_rpy = [
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=eval_cfg.FIXED_INTERVAL_EULER,
            x_start=val
        ) for val in object_orient_rpy_genome
    ]

    object_orient_quat = list(env.bullet_client.getQuaternionFromEuler(object_orient_rpy))

    object_6dof_pose = {
        'xyz': object_pos_xyz,
        'euler_rpy': object_orient_rpy,
        'quaternions': object_orient_quat,
    }
    return object_6dof_pose

def cvt_genome_to_6dof_pose_spherical(genome, env):

    dist2obj_centroid_genome = genome[0]
    longitude_angle_genome = genome[1]
    colatitude_angle_genome = genome[2]

    dist2obj_centroid = project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[0, env.sim_engine.search_space_bb_side],
            x_start=dist2obj_centroid_genome
        )
    longitude_angle = project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=eval_cfg.FIXED_INTERVAL_EULER,
            x_start=longitude_angle_genome
        )
    colatitude_angle = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, np.pi],
        x_start=colatitude_angle_genome
    )

    x = dist2obj_centroid * np.sin(colatitude_angle) * np.cos(longitude_angle)
    y = dist2obj_centroid * np.sin(colatitude_angle) * np.sin(longitude_angle) 
    z = dist2obj_centroid * np.cos(colatitude_angle)

    obj_pos_xyz = [x, y, z]
    
    object_orient_rpy_genome = genome[3:6]

    obj_orient_rpy = [
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=eval_cfg.FIXED_INTERVAL_EULER,
            x_start=val
        ) for val in object_orient_rpy_genome
    ]

    obj_6dof_pose = {
        'xyz': obj_pos_xyz,
        'euler_rpy': obj_orient_rpy,
        'quaternions': None,
    }
    return obj_6dof_pose

def cvt_genome_to_preset_6dof_pose_approach_strat_contact_point_finder(genome, env):

    # Extract values from the genome
    contact_point_finder_xyz_pose_genome_values = genome[:3]
    nu_genome = genome[3]
    d_genome = genome[4]
    ksi_genome = genome[5]
    omega_genome = genome[6]

    # Get contact point finder voxel 3D pose
    contact_point_finder_xyz_pose = [
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[ss_min, ss_max],
            x_start=gen_val
        )
        for gen_val, ss_min, ss_max in zip(
            contact_point_finder_xyz_pose_genome_values,
            env.sim_engine.search_space_bb_target.aabb_min,
            env.sim_engine.search_space_bb_target.aabb_max
        )
    ]

    # Compute the object position and orientation from the contact point
    nu_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, np.pi],
        x_start=nu_genome
    )

    d_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[env.pose_relative_to_contact_point_d_min, env.pose_relative_to_contact_point_d_max],
        x_start=d_genome
    )

    ksi_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=eval_cfg.FIXED_INTERVAL_EULER,
        x_start=ksi_genome
    )
    omega_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=eval_cfg.FIXED_INTERVAL_EULER,
        x_start=omega_genome
    )

    object_pose_from_contact_params = {
        'nu': nu_projected,
        'd': d_projected,
        'ksi': ksi_projected,
        'omega': omega_projected,
    }

    return contact_point_finder_xyz_pose, object_pose_from_contact_params


def cvt_genome_to_preset_6dof_pose_contact_strat_contact_point_finder(genome, env):

    # Extract values from the genome
    contact_point_finder_xyz_pose_genome_values = genome[:3]
    nu_genome = genome[3]
    d_genome = genome[4]
    ksi_genome = genome[5]
    omega_genome = genome[6]

    # Get contact point finder voxel 3D pose
    contact_point_finder_xyz_pose = [
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[ss_min, ss_max],
            x_start=gen_val
        )
        for gen_val, ss_min, ss_max in zip(
            contact_point_finder_xyz_pose_genome_values,
            env.sim_engine.search_space_bb_target.aabb_min,
            env.sim_engine.search_space_bb_target.aabb_max
        )
    ]

    # Compute the object position and orientation from the contact point
    nu_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, np.pi],
        x_start=nu_genome
    )

    d_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[env.pose_relative_to_contact_point_d_min, env.pose_relative_to_contact_point_d_max],
        x_start=d_genome
    )

    ksi_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=eval_cfg.FIXED_INTERVAL_EULER,
        x_start=ksi_genome
    )
    omega_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=eval_cfg.FIXED_INTERVAL_EULER,
        x_start=omega_genome
    )

    object_pose_from_contact_params = {
        'nu': nu_projected,
        'd': d_projected,
        'ksi': ksi_projected,
        'omega': omega_projected,
    }

    return contact_point_finder_xyz_pose, object_pose_from_contact_params

def cvt_genome_to_preset_6dof_pose_strat_pca_alignment(genome,env):

    rotation_genome = genome[0]
    translation_genome = genome[1]

    rotation_angle = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=eval_cfg.FIXED_INTERVAL_EULER,
        x_start=rotation_genome
    )

    translation_distance = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, env.sim_engine.search_space_bb_side],
        x_start=translation_genome
    )

    axis1_genome = np.random.randint(0, 3)
    axis2_genome = np.random.randint(0, 3)

    obj_pca_axes = env.sim_engine.bullet_sim_obj.pca_axes
    target_pca_axes = env.sim_engine.bullet_sim_target.pca_axes

    obj_pca_axe_genome = obj_pca_axes[axis1_genome]
    target_pca_axes_genome = target_pca_axes[axis2_genome]

    return obj_pca_axe_genome, target_pca_axes_genome, rotation_angle, translation_distance

def cvt_genome_to_preset_6dof_pose_strat_face_alignment(genome,env):

    rotation_genome = genome[0]
    translation_genome = genome[1]

    rotation_angle = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=eval_cfg.FIXED_INTERVAL_EULER,
        x_start=rotation_genome
    )

    translation_distance = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, env.sim_engine.search_space_bb_side],
        x_start=translation_genome
    )

    normal_to_triangles_obj = env.sim_engine.bullet_sim_obj.object_normals_to_triangles
    normal_to_triangles_traget = env.sim_engine.bullet_sim_target.object_normals_to_triangles

    non_zero_mask_obj = ~np.all(normal_to_triangles_obj == 0, axis=1)
    non_zero_mask_target = ~np.all(normal_to_triangles_traget == 0, axis=1)

    # Filtres les vecteurs non nuls
    filtered_normals_obj = normal_to_triangles_obj[non_zero_mask_obj]
    filtered_normals_target = normal_to_triangles_traget[non_zero_mask_target]

    normal_to_triangle_obj_genome = random.choice(filtered_normals_obj)
    normal_to_triangle_target_genome = random.choice(filtered_normals_target)

    return normal_to_triangle_obj_genome, normal_to_triangle_target_genome, rotation_angle, translation_distance

def cvt_genome_to_6dof_pose_approach_strat_contact_point_finder_search(genome, env):
    contact_point_finder_xyz_pose, object_pose_from_contact_params = \
        cvt_genome_to_preset_6dof_pose_approach_strat_contact_point_finder(genome=genome, env=env)

    # Get closer point on the target surface
    query = [contact_point_finder_xyz_pose]
    closest_contact_point_id = env.sim_engine.bullet_sim_target.k_tree_uniform_contact_points.kneighbors(X=query)[1][0][0]
    closest_contact_point = env.sim_engine.bullet_sim_target.uniform_obj_contact_points[closest_contact_point_id]

    normal_at_contact_point = get_normal_surface_point(
        bullet_client=env.bullet_client,
        list_of_points_for_each_triangle_object_mesh=env.sim_engine.bullet_sim_target.list_of_points_for_each_triangle_obj_mesh,
        object_normals_to_triangles=env.sim_engine.bullet_sim_target.object_normals_to_triangles,
        contact_point=closest_contact_point,
        debug=False
    )

    obj_6dof_pose = get_obj_pose_relatively_to_contact_point(
        bullet_client=env.bullet_client,
        object_pose_from_contact_params=object_pose_from_contact_params,
        normal_at_contact_point=normal_at_contact_point,
        contact_point=closest_contact_point,
        debug=False,
    )

    return obj_6dof_pose

def cvt_genome_to_6dof_pose_strat_pca_alignment(genome,env):

    obj_pca_axe_genome, target_pca_axes_genome, rotation_angle, translation_distance = cvt_genome_to_preset_6dof_pose_strat_pca_alignment(genome, env)

    R, T = align_objects_by_axis(
        object_axis=obj_pca_axe_genome,
        target_axis=target_pca_axes_genome,
        rotation_angle=rotation_angle,
        translation_distance=translation_distance)

    if is_gimbal_lock(R) :
        return None
        
    obj_orient_rpy = Rotation.from_matrix(R).as_euler('xyz')

    obj_pos_xyz = T

    obj_6dof_pose = {
        'xyz': obj_pos_xyz,
        'euler_rpy': obj_orient_rpy,
        'quaternions': None,
    }

    return obj_6dof_pose

def cvt_genome_to_6dof_pose_strat_face_alignment(genome,env):
    normal_to_triangle_obj_genome, normal_to_triangle_target_genome, rotation_angle, translation_distance = cvt_genome_to_preset_6dof_pose_strat_face_alignment(genome, env)

    R, T = align_objects_by_axis(
        object_axis=normal_to_triangle_obj_genome,
        target_axis=normal_to_triangle_target_genome,
        rotation_angle=rotation_angle,
        translation_distance=translation_distance)
    
    max_attempts = 10
    attempt = 0

    while is_gimbal_lock(R) :
        attempt += 1
        if attempt >= max_attempts:
            return None
        
        normal_to_triangle_obj_genome, normal_to_triangle_target_genome, rotation_angle, translation_distance = cvt_genome_to_preset_6dof_pose_strat_face_alignment(genome, env)

        R, T = align_objects_by_axis(
            object_axis=normal_to_triangle_obj_genome,
            target_axis=normal_to_triangle_target_genome,
            rotation_angle=rotation_angle,
            translation_distance=translation_distance)
        
    obj_orient_rpy = Rotation.from_matrix(R).as_euler('xyz')

    obj_pos_xyz = T

    obj_6dof_pose = {
        'xyz': obj_pos_xyz,
        'euler_rpy': obj_orient_rpy,
        'quaternions': None,
    }

    return obj_6dof_pose

def is_gimbal_lock(R):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = Rotation.from_matrix(R).as_euler('xyz')
        return any("Gimbal lock" in str(warn.message) for warn in w)
