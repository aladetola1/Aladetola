import os
import numpy as np
import pybullet as p
import pybullet_data


# ---------------------------
# INIT SIMULATION
# ---------------------------
def init_sim():
    if p.getConnectionInfo()['isConnected']:
        p.disconnect()

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)


def reset_sim():
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")


# ---------------------------
# OBJECTS
# ---------------------------
URDF_MAP = {
    "mug": "environments/3d_models/objects/mug/mug.urdf",
    "bowl": "environments/3d_models/objects/ycb_bowl/ycb_bowl.urdf",
    "banana": "environments/3d_models/objects/ycb_banana/ycb_banana.urdf",
    "drill": "environments/3d_models/objects/ycb_power_drill/ycb_power_drill.urdf",
    "plate": "environments/3d_models/objects/ycb_plate/ycb_plate.urdf",
    "spoon": "environments/3d_models/objects/ycb_spoon/ycb_spoon.urdf",
}


def load_object(name):
    return p.loadURDF(URDF_MAP[name], [0, 0, 0.2])


# ---------------------------
# PHYSICS SETTLING (CRITICAL FIX)
# ---------------------------
def wait_until_settled(obj_id, max_steps=1000):
    for _ in range(max_steps):
        p.stepSimulation()

        lin_vel, ang_vel = p.getBaseVelocity(obj_id)

        if (np.linalg.norm(lin_vel) < 0.02 and
            np.linalg.norm(ang_vel) < 0.02):
            break


# ---------------------------
# STABILITY CHECK (RELAXED)
# ---------------------------
def is_stable(obj_id):
    lin_vel, ang_vel = p.getBaseVelocity(obj_id)

    return (
        np.linalg.norm(lin_vel) < 0.1 and
        np.linalg.norm(ang_vel) < 0.1
    )


# ---------------------------
# UPRIGHT CHECK (REALISTIC)
# ---------------------------
def is_upright(obj_id):
    _, orn = p.getBasePositionAndOrientation(obj_id)
    rpy = p.getEulerFromQuaternion(orn)

    return abs(rpy[0]) < 0.9 and abs(rpy[1]) < 0.9


# ---------------------------
# SINGLE EPISODE
# ---------------------------
def run_episode(x, obj_name):
    reset_sim()

    obj_id = load_object(obj_name)

    pos = x[:3]
    orn = p.getQuaternionFromEuler(x[3:])

    p.resetBasePositionAndOrientation(obj_id, pos, orn)

    # IMPORTANT: let physics settle fully
    wait_until_settled(obj_id)

    return is_stable(obj_id) and is_upright(obj_id)


# ---------------------------
# LOAD BEST SOLUTION
# ---------------------------
def load_best(run_path):
    path = os.path.join(run_path, "success_archives/individuals_0.npz")
    d = np.load(path, allow_pickle=True)

    fitness = d["fitnesses"]
    inds = d["inds"]

    best_idx = np.argmax(fitness)
    return inds[best_idx], fitness[best_idx]


# ---------------------------
# RUN EVALUATION
# ---------------------------
RUNS = {
    "mug": "runs/run22",
    "bowl": "runs/run22",
    "banana": "runs/run22",
    "drill": "runs/run22",
    "plate": "runs/run22",
    "spoon": "runs/run22",
}

init_sim()

results = []

for obj, path in RUNS.items():
    print(f"\n=== Evaluating {obj} ===")

    base, base_fit = load_best(path)

    obj_id = load_object(obj)

    # apply best pose
    p.resetBasePositionAndOrientation(
        obj_id,
        base[:3],
        p.getQuaternionFromEuler(base[3:])
    )

    wait_until_settled(obj_id)

    stable = is_stable(obj_id)
    upright = is_upright(obj_id)

    score = float(stable and upright)

    print("best fitness:", base_fit)
    print("upright+stable score:", score)

    results.append((obj, base_fit, score))


np.savez(
    "upright_grasp_results.npz",
    results=np.array(results, dtype=object)
)

print("\nSaved → upright_grasp_results.npz")
