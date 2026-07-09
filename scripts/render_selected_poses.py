import os
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

OBJECTS = {
    "mug": {
        "run": "results/mug/run87",
        "urdf": "environments/3d_models/objects/ycb_mug/ycb_mug.urdf",
    },
    "bowl": {
        "run": "results/bowl/run89",
        "urdf": "environments/3d_models/objects/ycb_bowl/ycb_bowl.urdf",
    },
    "spoon": {
        "run": "results/spoon/run94",
        "urdf": "environments/3d_models/objects/ycb_spoon/ycb_spoon.urdf",
    },
    "power_drill": {
        "run": "results/power_drill/run97",
        "urdf": "environments/3d_models/objects/ycb_power_drill/ycb_power_drill.urdf",
    },
    "plate": {
        "run": "results/plate/run100",
        "urdf": "environments/3d_models/objects/ycb_plate/ycb_plate.urdf",
    },
    "banana": {
        "run": "results/banana/run103",
        "urdf": "environments/3d_models/objects/ycb_banana/ycb_banana.urdf",
    },
}

OUT_DIR = "results/paper_pose_figures"
os.makedirs(OUT_DIR, exist_ok=True)

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

for name, cfg in OBJECTS.items():
    print("Rendering:", name)

    top_path = os.path.join(cfg["run"], "top10_grasp_candidates.npy")

    if not os.path.exists(top_path):
        print("Missing:", top_path)
        continue

    if not os.path.exists(cfg["urdf"]):
        print("Missing URDF:", cfg["urdf"])
        continue

    candidates = np.load(top_path, allow_pickle=True)

    best = candidates[0]
    if hasattr(best, "item"):
        best = best.item()

    final_pose = best["final_pose"]

    xyz = final_pose["xyz"]
    rpy = final_pose["euler_rpy"]
    quat = p.getQuaternionFromEuler(rpy)

    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf")

    table_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.35, 0.35, 0.02]
    )

    table_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.35, 0.35, 0.02],
        rgbaColor=[0.65, 0.45, 0.25, 1.0]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_col,
        baseVisualShapeIndex=table_vis,
        basePosition=[0, 0, 0.20]
    )

    p.loadURDF(
        cfg["urdf"],
        basePosition=xyz,
        baseOrientation=quat,
        useFixedBase=True
    )

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.55, -0.75, 0.55],
        cameraTargetPosition=[0, 0, 0.25],
        cameraUpVector=[0, 0, 1]
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=45,
        aspect=1.3,
        nearVal=0.01,
        farVal=3.0
    )

    width, height = 1400, 1000

    img = p.getCameraImage(
        width,
        height,
        view_matrix,
        proj_matrix,
        renderer=p.ER_TINY_RENDERER
    )

    rgb = np.reshape(img[2], (height, width, 4))[:, :, :3]

    out_path = os.path.join(OUT_DIR, f"{name}_best_grasp_pose.png")
    Image.fromarray(rgb.astype(np.uint8)).save(out_path)

    print("Saved:", out_path)

p.disconnect()
