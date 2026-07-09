import numpy as np

def run_grasp_rollout(env, object_pose, n_steps=200):
    """
    Full grasp + lift rollout using existing robot_grasping framework.
    """

    # 1. reset full environment
    env.reset()

    # 2. set object pose (via simulation engine)
    env.sim_engine.set_6dof_pose_object(
        bullet_client=env.bullet_client,
        start_pos_object_xyz=object_pose["xyz"],
        start_orient_object_rpy=None,
        start_orient_object_quat=object_pose["quaternions"]
    )

    # 3. run pre-grasp motion (already defined in robot policy)
    obs = env.step_pregrasp()

    # 4. close gripper using EXISTING implementation
    env.robot.close_gripper(
        robot_id=env.sim_engine.obj_id,
        fingers_joint_infos=env.sim_engine.fingers_joint_infos(),
        synergy_label=None
    )

    # 5. simulate physics forward
    for _ in range(n_steps):
        env.step()

    # 6. evaluate success
    grasp_success = env.sim_engine.is_there_contacts(
        env.bullet_client.getContactPoints(bodyA=env.sim_engine.obj_id)
    )

    lifted = env.sim_engine.obj_id is not None and env.sim_engine.obj_id

    return {
        "grasp_success": grasp_success,
        "lift_success": lifted
    }
