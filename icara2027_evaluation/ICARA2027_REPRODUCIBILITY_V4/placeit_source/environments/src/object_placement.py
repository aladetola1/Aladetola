

import time
import numpy as np

from pybullet_utils.bullet_client import BulletClient
import pybullet as p

import environments.src.env_constants as env_consts
import configs.eval_config as eval_cfg
import configs.exec_config as exec_cfg

from environments.src.bullet_simulation.simulation_rendering import SimulationRendering
from environments.src.bullet_simulation.object_simulation_engine import SimulationEngine
import configs.exec_config as exec_cfg
import pdb

class ObjectPlacement:
    def __init__(
            self,
            object_name,
            target_name = None,
            display=eval_cfg.BULLET_DEFAULT_DISPLAY_FLG,
            debug=False,
            use_concave_mesh=False,
            **kwargs
            ):

        self._bullet_client = None  # bullet physics client
        self.physics_client_id = None  # bullet physics client id
        self.sim_render = None  # manage simulation rendering
        self.sim_engine = None  # manage simulation engine

        self._debug = debug
        self.debug_i_debug_bodies = []  # for debugging purpose
        self.target_name = target_name

        self._init_attributes(
            display=display,
            object_name=object_name,
            use_concave_mesh=use_concave_mesh,
            **kwargs
        )

    @property
    def obj_id(self):
        return self.sim_engine.obj_id
    
    @property
    def bullet_client(self):
        return self._bullet_client

    @property
    def debug(self):
        return self._debug
    
    @property
    def search_space_bb(self):
        return self.sim_engine.search_space_bb

    @property
    def search_space_bb_side(self):
        return self.sim_engine.search_space_bb_side
    
    @property
    def search_space_bb_object(self):
        return self.sim_engine.search_space_bb_object

    @property
    def search_space_bb_target(self):
        return self.sim_engine.search_space_bb_target
    
    @property
    def pose_relative_to_contact_point_d_min(self):
        return self.sim_engine.pose_relative_to_contact_point_d_min

    @property
    def pose_relative_to_contact_point_d_max(self):
        return self.sim_engine.pose_relative_to_contact_point_d_max

    def _init_attributes(
            self,
            display,
            object_name,
            use_concave_mesh,
    ):

        self._init_bullet_physics_client(display=display)

        sim_engine_kwargs = {
            'object_name': object_name,
            'bullet_client': self._bullet_client,
            'target_name': self.target_name,
            'use_concave_mesh': use_concave_mesh,
        }
        
        self.sim_engine = SimulationEngine(**sim_engine_kwargs)

        self.sim_engine.reset(bullet_client=self._bullet_client)

        self.sim_engine.init_local_sim_save(bullet_client=self._bullet_client)
        
        self._init_rendering(
            display=display,
        )

    def _init_rendering(self, display):
        self.sim_render = SimulationRendering(bullet_client=self._bullet_client, display=display)

    def _init_bullet_physics_client(self, display):
        self._bullet_client = BulletClient(connection_mode=p.GUI if display else p.DIRECT)
        self.physics_client_id = self._bullet_client._client
    
    def reset(
            self,
    ):
        # load from local save the initialized scene for faster computation
        self.sim_engine.load_state_from_local_save(
            bullet_client=self._bullet_client,
        )

    def close(self):
        is_bullet_client_on = self.physics_client_id >= 0
        if is_bullet_client_on:
            self._bullet_client.disconnect()
            self.physics_client_id = -1

    def is_debug_mode(self):
        return self._debug

    def delete_debug_bodies(self):
        if len(self.debug_i_debug_bodies) == 0:
            return

        for body_id in self.debug_i_debug_bodies:
            self.bullet_client.removeBody(body_id)

        self.debug_i_debug_bodies = []

    def set_6dof_obj_pose(self, obj_6dof_pose):

        self.sim_engine.set_6dof_pose_object(
            bullet_client=self._bullet_client,
            start_pos_object_xyz=obj_6dof_pose['xyz'],
            start_orient_object_rpy=obj_6dof_pose['euler_rpy'],
            start_orient_object_quat=obj_6dof_pose['quaternions'],
        )
        
        self._bullet_client.stepSimulation()
    
    def get_obj_mesh_vertice_points_in_world_frame(self):

        pc_obj_frame = self.sim_engine.obj_mesh_vertice_points

        pos, orn = self.bullet_client.getBasePositionAndOrientation(self.obj_id)    
        rot_matrix = np.array(self.bullet_client.getMatrixFromQuaternion(orn)).reshape(3, 3)

        pc_world_frame = (rot_matrix @ pc_obj_frame.T).T + pos

        T_obj_world = np.eye(4)
        T_obj_world[:3, :3] = rot_matrix
        T_obj_world[:3, 3] = pos

        # Matrice inverse : world → object
        T_world_obj = np.linalg.inv(T_obj_world)

        return pc_world_frame , T_world_obj
    
    def get_obj_mesh_vertice_points_in_obj_frame(self):
        return self.sim_engine.obj_mesh_vertice_points

    def get_obj_point_cloud_in_world_frame(self):

        pc_obj_frame = self.sim_engine.obj_mesh_vertice_points

        pos, orn = self.bullet_client.getBasePositionAndOrientation(self.obj_id)    
        rot_matrix = np.array(self.bullet_client.getMatrixFromQuaternion(orn)).reshape(3, 3)

        pc_world_frame = (rot_matrix @ pc_obj_frame.T).T + pos

        T_obj_world = np.eye(4)
        T_obj_world[:3, :3] = rot_matrix
        T_obj_world[:3, 3] = pos

        # Matrice inverse : world → object
        T_world_obj = np.linalg.inv(T_obj_world)

        return pc_world_frame , T_world_obj
    
    def get_obj_point_cloud_in_obj_frame(self):
        return self.sim_engine.obj_point_cloud

    def drop_object(self) :

        # Initialize simulation args
        simulation_time = exec_cfg.SIMULATION_TIME
        time_step = exec_cfg.TIME_STEP
        sampling_rate = exec_cfg.SAMPLING_RATE

        total_steps = int(simulation_time / time_step)
        sample_interval = max(1, int(1.0 / (sampling_rate * time_step)))
       
        trajectory_data = []

        for step in range(total_steps):
            self._bullet_client.stepSimulation()
            
            if step % sample_interval == 0:
                pos, quat = self._bullet_client.getBasePositionAndOrientation(self.obj_id)
                euler = self._bullet_client.getEulerFromQuaternion(quat)
                trajectory_data.append((pos, euler))
                
            if self.sim_render.display:
                time.sleep(time_step)

            if self.is_there_overlapping(threshold=-5e-3) :
                return None

        return trajectory_data

    def add_noise_to_object_state(self):

        obj_pose_xyz, obj_orient_quat = self.bullet_client.getBasePositionAndOrientation(self.obj_id)

        noise2add_obj_pose_z = np.random.normal(
            loc=0.0, scale=eval_cfg.DOMAIN_RANDOMIZATION_OBJECT_POS_VARIANCE_IN_M
        )
        noise2add_obj_orient_euler_rpy = np.random.normal(
            loc=0.0, scale=eval_cfg.DOMAIN_RANDOMIZATION_OBJECT_ORIENT_EULER_VARIANCE_IN_RAD, size=3
        )
        noise2add_obj_orient_quat = self.bullet_client.getQuaternionFromEuler(noise2add_obj_orient_euler_rpy)

        #noisy_obj_pose_xyz = np.array(obj_pose_xyz) + noise2add_obj_pose_xyz
        noisy_obj_pose_xyz = np.array(obj_pose_xyz).copy()
        noisy_obj_pose_xyz[2] += noise2add_obj_pose_z
        noisy_obj_orient_quat = np.array(obj_orient_quat) + noise2add_obj_orient_quat

        self.bullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=self.obj_id,
            posObj= noisy_obj_pose_xyz,
            ornObj=noisy_obj_orient_quat
        )

    def add_noise_to_friction_coefficients(self):

        noisy_rolling_friction = np.random.uniform(
            low=eval_cfg.DOMAIN_RANDOMIZATION_ROLLING_FRICTION_MIN_VALUE,
            high=eval_cfg.DOMAIN_RANDOMIZATION_ROLLING_FRICTION_MAX_VALUE
        )
        noisy_spinning_friction = np.random.uniform(
            low=eval_cfg.DOMAIN_RANDOMIZATION_SPINNING_FRICTION_MIN_VALUE,
            high=eval_cfg.DOMAIN_RANDOMIZATION_SPINNING_FRICTION_MAX_VALUE
        )

        self.bullet_client.changeDynamics(
            bodyUniqueId=self.obj_id, linkIndex=-1,
            rollingFriction=noisy_rolling_friction,
            spinningFriction=noisy_spinning_friction
        )

    def is_there_contacts(self):
        contacts_obj = self._bullet_client.getContactPoints(bodyA=self.obj_id)
        return self.sim_engine.is_there_contacts(contacts_obj)
    
    def is_there_overlapping(self, threshold=0):
        return self.sim_engine.is_there_overlapping(bullet_client=self._bullet_client,threshold=threshold)
    
    def get_contact_points(self) :
        return self.sim_engine.get_contact_points(bullet_client=self._bullet_client)

    def add_noise_to_object_final_state(self):

        obj_pose_xyz, obj_orient_quat = self.bullet_client.getBasePositionAndOrientation(self.obj_id)
        obj_orient_euler = np.array(self.bullet_client.getEulerFromQuaternion(obj_orient_quat))
        obj_pose_xyz = np.array(obj_pose_xyz)
        pose_rpy = np.concatenate([obj_pose_xyz, obj_orient_euler])

        idx = np.random.randint(0, 6)

        # Appliquer le bruit
        if idx < 3:
            # Bruit positionnel
            pose_rpy[idx] += np.random.normal(loc=0.0, scale=eval_cfg.DOMAIN_RANDOMIZATION_OBJECT_POS_VARIANCE_IN_M)
        else:
            # Bruit rotationnel
            pose_rpy[idx] += np.random.normal(loc=0.0, scale=eval_cfg.DOMAIN_RANDOMIZATION_OBJECT_ORIENT_EULER_VARIANCE_IN_RAD)

        noisy_xyz = pose_rpy[:3]
        noisy_euler = pose_rpy[3:]
        noisy_quat = self.bullet_client.getQuaternionFromEuler(noisy_euler)

        self.bullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=self.obj_id,
            posObj= noisy_xyz,
            ornObj=noisy_quat
        )

    def apply_external_force_to_object_final_state(self) :

        obj_pose_xyz, obj_orient_quat = self.bullet_client.getBasePositionAndOrientation(self.obj_id)

        position = obj_pose_xyz
        #force = np.random.uniform(-10, 10, size=3).tolist()
        force = self.generate_random_force()

        # Appliquer la force
        p.applyExternalForce(
            objectUniqueId=self.obj_id,
            linkIndex=-1,
            forceObj=force,
            posObj=position,
            flags=self.bullet_client.WORLD_FRAME
        )

        self._bullet_client.stepSimulation()

    def generate_random_force(self, min_acc=5.0, max_acc=10.0):
        link_index=-1

        mass, *_ = self._bullet_client.getDynamicsInfo(self.obj_id,link_index)
        acc = np.random.uniform(min_acc, max_acc, size=3) * np.random.choice([-1, 1], size=3)

        force = 10 * mass * acc

        return force

    def visualize_contacts_with_normals(self):
        contact_points = self._bullet_client.getContactPoints(bodyA=self.obj_id)
        
        for contact in contact_points:
            contact_pos = contact[5]  # Position du contact
            normal = contact[7]       # Vecteur normal
            normal_end = [contact_pos[i] + 0.1 * normal[i] for i in range(3)]
            
            # Point de contact
            p.addUserDebugPoints(
                [contact_pos],
                [[1, 0, 0]],
                pointSize=8,
                lifeTime=1.0
            )
            
            # Ligne normale
            p.addUserDebugLine(
                contact_pos,
                normal_end,
                lineColorRGB=[0, 1, 0],  # Vert
                lineWidth=2,
                lifeTime=1.0
            )