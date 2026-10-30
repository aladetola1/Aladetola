from pathlib import Path
import pickle
import csv
import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "icara2027_evaluation/E3c_disturbance_all_objects.csv"

OBJECTS = [
    "mug", "bowl", "spoon",
    "power_drill", "plate", "banana"
]

FORCE_LEVELS = [0.10, 0.20, 0.40]
DIRECTIONS = {
    "+X": (1, 0, 0),
    "-X": (-1, 0, 0),
    "+Y": (0, 1, 0),
    "-Y": (0, -1, 0)
}

DT = 1 / 240
SETTLING_STEPS = 960
FORCE_STEPS = 24
RECOVERY_STEPS = 960

rng = np.random.default_rng(2027)


def gfi(pose):
    x, y, z = pose["xyz"]
    roll, pitch, yaw = pose["euler_rpy"]

    return (
        0.5 * z
        + 0.3 * np.exp(-np.hypot(x, y) / 0.1)
        + 0.2 * np.exp(-np.hypot(roll, pitch))
    )


def evaluate_pose(name, pose, force_ratio, direction):
    cid = p.connect(p.DIRECT)

    try:
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT)

        table = p.loadURDF(
            str(ROOT / "environments/3d_models/objects/table/table.urdf"),
            useFixedBase=True
        )
        p.changeDynamics(table, -1, lateralFriction=0.3)

        model_name = "ycb_" + name
        urdf = (
            ROOT / "environments/3d_models/objects"
            / model_name / (model_name + ".urdf")
        )

        obj = p.loadURDF(
            str(urdf),
            basePosition=pose["xyz"],
            baseOrientation=p.getQuaternionFromEuler(
                pose["euler_rpy"]
            ),
            useFixedBase=False
        )

        p.changeDynamics(
            obj, -1,
            lateralFriction=0.5,
            rollingFriction=0.0001,
            spinningFriction=0.0001
        )

        mass = p.getDynamicsInfo(obj, -1)[0]
        force_magnitude = force_ratio * mass * 9.81
        force_vector = [
            force_magnitude * d for d in direction
        ]

        for _ in range(SETTLING_STEPS):
            p.stepSimulation()

        before_pos, before_quat = (
            p.getBasePositionAndOrientation(obj)
        )

        for _ in range(FORCE_STEPS):
            com_pos, _ = p.getBasePositionAndOrientation(obj)

            p.applyExternalForce(
                obj, -1,
                forceObj=force_vector,
                posObj=com_pos,
                flags=p.WORLD_FRAME
            )
            p.stepSimulation()

        for _ in range(RECOVERY_STEPS):
            p.stepSimulation()

        after_pos, after_quat = (
            p.getBasePositionAndOrientation(obj)
        )

        before = np.array(before_pos)
        after = np.array(after_pos)

        horizontal_drift = (
            np.linalg.norm(after[:2] - before[:2]) * 1000
        )

        total_drift = (
            np.linalg.norm(after - before) * 1000
        )

        q_diff = p.getDifferenceQuaternion(
            before_quat, after_quat
        )

        rotation = np.degrees(
            2 * np.arctan2(
                np.linalg.norm(q_diff[:3]),
                abs(q_diff[3])
            )
        )

        contacts = len(p.getContactPoints(obj, table))

        return {
            "force_N": force_magnitude,
            "horizontal_drift_mm": horizontal_drift,
            "total_drift_mm": total_drift,
            "rotation_deg": rotation,
            "final_contacts": contacts,
            "contact_retained": int(contacts > 0)
        }

    finally:
        p.disconnect(cid)


fields = [
    "object", "group", "rank", "archive_index",
    "gfi", "force_ratio", "direction",
    "force_N", "horizontal_drift_mm",
    "total_drift_mm", "rotation_deg",
    "final_contacts", "contact_retained"
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for name in OBJECTS:
        archive = ROOT / "paper_runs" / name / "data_export.pkl"

        with archive.open("rb") as source:
            data = pickle.load(source)

        valid = [
            (i, pose)
            for i, pose in enumerate(data["final_6dof_pos"])
            if pose["xyz"][2] > 0
        ]

        ranked = sorted(
            valid,
            key=lambda item: gfi(item[1]),
            reverse=True
        )

        top10 = list(range(10))
        random10 = rng.choice(
            np.arange(10, len(ranked)),
            size=10,
            replace=False
        )

        selections = (
            [("GFI Top-10", i) for i in top10]
            + [("Random", int(i)) for i in random10]
        )

        completed = 0

        for group, rank_index in selections:
            archive_index, pose = ranked[rank_index]

            for ratio in FORCE_LEVELS:
                for direction_name, direction in DIRECTIONS.items():
                    result = evaluate_pose(
                        name, pose, ratio, direction
                    )

                    row = {
                        "object": name,
                        "group": group,
                        "rank": rank_index + 1,
                        "archive_index": archive_index,
                        "gfi": gfi(pose),
                        "force_ratio": ratio,
                        "direction": direction_name,
                        **result
                    }

                    writer.writerow(row)
                    completed += 1

        f.flush()
        print(
            f"Completed {name}: {completed} trials",
            flush=True
        )

print("\n========== E3c COMPLETE ==========")
print("Saved:", OUTPUT)
