
import os
import numpy as np
from pathlib import Path
import pybullet_data

from environments.src.search_space_bb_processor import get_search_space_bb, get_search_space_bb_side
from environments.src.bullet_simulation.entities.bullet_sim_object import BulletSimObject
from environments.src.bullet_simulation.entities.bullet_sim_target import BulletSimTarget
from algorithms.evaluation.pose_strategies_routines import display_mesh_point_from_array_debug, display_normal_to_contact_point_debug

import pdb
class SimulationEngine:
    def __init__(
            self,
            object_name,
            bullet_client,
            target_name,
            use_concave_mesh,
    ):
        self.init_state_p_file_root = None  # local save of bullet sim config for quick reinitialization
        self.bullet_sim_obj = None  # manage bullet simulation object to grasp
        self.bullet_sim_target = None 
        self._search_space_bb = None
        self._search_space_bb_side = None

        self._init_attributes(
            object_name=object_name,
            target_name=target_name,
            bullet_client=bullet_client,
            use_concave_mesh=use_concave_mesh,
        )

    @property
    def obj_id(self):
        return self.bullet_sim_obj.obj_id
    
    @property
    def target_id(self):
        return self.bullet_sim_target.target_id

    @property
    def search_space_bb(self):
        return self._search_space_bb

    @property
    def search_space_bb_side(self):
        return self._search_space_bb_side

    @property
    def search_space_bb_object(self):
        return self.bullet_sim_obj.search_space_bb_object

    @property
    def search_space_bb_target(self):
        return self.bullet_sim_target.search_space_bb_target
    
    @property
    def path2obj_point_cloud(self):
        return self.bullet_sim_obj.path2obj_point_cloud

    @property
    def list_of_points_for_each_triangle_obj_mesh(self):
        return self.bullet_sim_obj.list_of_points_for_each_triangle_obj_mesh

    @property
    def object_normals_to_triangles(self):
        return self.bullet_sim_obj.object_normals_to_triangles

    @property
    def obj_mesh_vertice_points(self):
        return self.bullet_sim_obj.obj_mesh_vertice_points

    @property
    def obj_point_cloud(self):
        return self.bullet_sim_obj.obj_point_cloud
        
    @property
    def uniform_obj_contact_points(self):
        return self.bullet_sim_obj.uniform_obj_contact_points

    @property
    def k_tree_uniform_contact_points(self):
        return self.bullet_sim_obj.k_tree_uniform_contact_points

    @property
    def pose_relative_to_contact_point_d_min(self):
        return self.bullet_sim_obj.pose_relative_to_contact_point_d_min

    @property
    def pose_relative_to_contact_point_d_max(self):
        return self.bullet_sim_obj.pose_relative_to_contact_point_d_max


    def _init_attributes(
            self,
            object_name,
            target_name,
            bullet_client,
            use_concave_mesh,
    ):

        bullet_client.resetSimulation()
        bullet_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        bullet_client.setGravity(0,0,-9.81)

        object_kwargs = {
            'bullet_client': bullet_client,
            'name': object_name,
        }
        
        self.bullet_sim_obj = BulletSimObject(**object_kwargs)

        env_kwargs = {
            'bullet_client': bullet_client,
            'name': target_name,
            'use_concave_mesh': use_concave_mesh,
        }

        self.bullet_sim_target = BulletSimTarget(**env_kwargs)

        self._search_space_bb = get_search_space_bb(obj_id=self.obj_id, target_id=self.target_id)     # object + env
        self._search_space_bb_side = get_search_space_bb_side(self._search_space_bb)
        

    def load_state_from_local_save(self, bullet_client):
        assert self.init_state_p_file_root, 'bullet tmp file not properly set'
        file2load = self.init_state_p_file_root
        bullet_client.restoreState(fileName=file2load)

    def init_local_sim_save(self, bullet_client):

        save_folder_root = os.getcwd() + '/tmp'

        self.init_state_p_file_root = save_folder_root + "/init_state.bullet"

        Path(save_folder_root).mkdir(exist_ok=True)

        # Make sure each worker (cpu core) has its own local save
        init_state_p_local_pid_file = self.init_state_p_file_root
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(f'dir_path = {dir_path}')
        print('trying to save = ', init_state_p_local_pid_file)
        bullet_client.saveBullet(init_state_p_local_pid_file)
        print(f'init_state_p_local_pid_file={init_state_p_local_pid_file} successfully saved.')


    def set_6dof_pose_object(
            self, bullet_client, start_pos_object_xyz, start_orient_object_rpy, start_orient_object_quat
    ):
        if start_orient_object_rpy is not None:
            start_orient_object_quaternion = bullet_client.getQuaternionFromEuler(start_orient_object_rpy)
        elif start_orient_object_quat is not None:
            start_orient_object_quaternion = start_orient_object_quat
        else:
            raise AttributeError('start_orient_robot_rpy or start_orient_robot_quat must be != None to set pose.')

        bullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=self.obj_id,
            posObj=start_pos_object_xyz,
            ornObj=start_orient_object_quaternion
        )

    def reset(self, bullet_client):
        self.bullet_sim_obj.reset_object_pose(bullet_client=bullet_client)

    def is_there_contacts(self, contacts):
        return len(contacts) != 0

    def is_there_overlapping(self, bullet_client,threshold):
        contacts_obj = bullet_client.getContactPoints(bodyA=self.obj_id)
        id_arg_contact_dist = 8
        if self.is_there_contacts(contacts_obj):
            for i in range(len(contacts_obj)):
                contact_distance = contacts_obj[i][id_arg_contact_dist]
                if contact_distance < threshold:
                    is_there_overlap = True
                    return is_there_overlap
        is_there_overlap = False
        return is_there_overlap

    def get_contact_points(self, bullet_client):
        contact_points = bullet_client.getContactPoints(bodyA=self.obj_id)
        return contact_points