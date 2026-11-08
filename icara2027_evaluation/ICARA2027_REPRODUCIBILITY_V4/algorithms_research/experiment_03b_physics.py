import csv
import pickle
from pathlib import Path

import numpy as np
import pybullet as p


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "icara2027_evaluation" / "E3b_ycb_mug_4seconds.csv"

ARCHIVE = ROOT / "paper_runs" / "mug" / "data_export.pkl"
TABLE = ROOT / "environments/3d_models/objects/table/table.urdf"
MUG = ROOT / "environments/3d_models/objects/ycb_mug/ycb_mug.urdf"

SIMULATION_STEPS = 960
TIME_STEP = 1 / 240


def gfi(pose):
    x, y, z = pose["xyz"]
    roll, pitch, yaw = pose["euler_rpy"]

    return (
        0.5 * z
        + 0.3 * np.exp(-np.hypot(x, y) / 0.1)
        + 0.2 * np.exp(-np.hypot(roll, pitch))
    )


with open(ARCHIVE, "rb") as f:
    data = pickle.load(f)

poses = [
    pose for pose in data["final_6dof_pos"]
    if pose["xyz"][2] > 0
]

ranked_indices = sorted(
    range(len(poses)),
    key=lambda i: gfi(poses[i]),
    reverse=True
)

client = p.connect(p.DIRECT)
results = []

try:
    for rank, index in enumerate(ranked_indices, start=1):
        pose = poses[index]

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(TIME_STEP)

        table = p.loadURDF(str(TABLE), useFixedBase=True)
        p.changeDynamics(table, -1, lateralFriction=0.3)

        initial_xyz = np.asarray(pose["xyz"], dtype=float)
        initial_quat = p.getQuaternionFromEuler(
            list(pose["euler_rpy"])
        )

        mug = p.loadURDF(
            str(MUG),
            basePosition=initial_xyz.tolist(),
            baseOrientation=initial_quat,
            useFixedBase=False
        )

        p.changeDynamics(
            mug, -1,
            lateralFriction=0.5,
            rollingFriction=0.0001,
            spinningFriction=0.0001
        )

        for _ in range(SIMULATION_STEPS):
            p.stepSimulation()

        final_xyz, final_quat = p.getBasePositionAndOrientation(mug)
        linear_velocity, angular_velocity = p.getBaseVelocity(mug)

        drift_mm = (
            np.linalg.norm(np.asarray(final_xyz) - initial_xyz)
            * 1000
        )

        q_diff = p.getDifferenceQuaternion(
            initial_quat, final_quat
        )

        rotation_deg = np.degrees(
            2 * np.arctan2(
                np.linalg.norm(q_diff[:3]),
                abs(q_diff[3])
            )
        )

        contacts = len(p.getContactPoints(mug, table))

        results.append({
            "rank": rank,
            "archive_index": index,
            "gfi": float(gfi(pose)),
            "drift_mm": float(drift_mm),
            "rotation_deg": float(rotation_deg),
            "contacts": contacts,
            "linear_speed": float(np.linalg.norm(linear_velocity)),
            "angular_speed": float(np.linalg.norm(angular_velocity)),
        })

        if rank % 50 == 0 or rank == len(ranked_indices):
            print(f"Completed {rank}/{len(ranked_indices)} poses", flush=True)

finally:
    p.disconnect()

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

top10 = results[:10]
remaining = results[10:]

print("\n========== E3b MUG RESULTS ==========")
print("Total poses:", len(results))
print("Top-10 mean drift (mm):",
      round(np.mean([r["drift_mm"] for r in top10]), 4))
print("Remaining poses mean drift (mm):",
      round(np.mean([r["drift_mm"] for r in remaining]), 4))
print("Top-10 mean rotation (deg):",
      round(np.mean([r["rotation_deg"] for r in top10]), 4))
print("Remaining poses mean rotation (deg):",
      round(np.mean([r["rotation_deg"] for r in remaining]), 4))
print("Saved:", OUTPUT)
