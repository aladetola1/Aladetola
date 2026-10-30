import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environments.src.object_placement import ObjectPlacement

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "experiment_11_additional_115_manifest.csv"

OUTPUTS = [
    HERE / "experiment_11_additional_115_replay.csv",
    HERE / "experiment_12_additional_115_contact_force.csv",
    HERE / "experiment_13_additional_115_fullrate.csv",
]

def rotation_error_deg(q1, q2):
    return float(np.degrees(
        (Rotation.from_quat(q1) * Rotation.from_quat(q2).inv()).magnitude()
    ))

def run_placement(env, initial_pose, archived_final):
    env.reset()
    env.set_6dof_obj_pose(initial_pose)

    p = env.bullet_client
    object_id = env.obj_id
    table_id = env.sim_engine.target_id

    mass = float(p.getDynamicsInfo(object_id, -1)[0])
    weight = mass * 9.81
    assert weight > 0

    positions = []
    orientations = []
    linear_speeds = []
    angular_speeds = []
    sampled_linear = []
    sampled_angular = []

    force_ratios = []
    horizontal_forces = []
    positive_contacts = []

    archived_samples = []
    final_table_contacts = 0

    for step in range(960):
        p.stepSimulation()

        if step % 12 == 0:
            xyz, quat = p.getBasePositionAndOrientation(object_id)
            archived_samples.append((
                np.asarray(xyz, dtype=float),
                np.asarray(quat, dtype=float)
            ))

        if step < 840:
            continue

        xyz, quat = p.getBasePositionAndOrientation(object_id)
        velocity, angular_velocity = p.getBaseVelocity(object_id)

        positions.append(np.asarray(xyz, dtype=float))
        orientations.append(np.asarray(quat, dtype=float))

        linear_speed = float(np.linalg.norm(velocity))
        angular_speed = float(np.linalg.norm(angular_velocity))

        linear_speeds.append(linear_speed)
        angular_speeds.append(angular_speed)

        if step % 12 != 0:
            continue

        sampled_linear.append(linear_speed)
        sampled_angular.append(angular_speed)

        contacts = p.getContactPoints(
            bodyA=object_id,
            bodyB=table_id
        )

        total_force = np.zeros(3, dtype=float)
        positive = 0

        for c in contacts:
            if c[9] <= 0:
                continue

            positive += 1
            total_force += (
                c[9] * np.asarray(c[7], dtype=float)
                + c[10] * np.asarray(c[11], dtype=float)
                + c[12] * np.asarray(c[13], dtype=float)
            )

        positive_contacts.append(positive)
        force_ratios.append(float(total_force[2] / weight))
        horizontal_forces.append(
            float(np.linalg.norm(total_force[:2]))
        )

    final_table_contacts = len(p.getContactPoints(
        bodyA=object_id,
        bodyB=table_id
    ))

    assert len(archived_samples) == 80
    assert len(positions) == 120
    assert len(sampled_linear) == 10
    assert len(force_ratios) == 10

    archive_xyz = np.asarray(
        archived_final["xyz"], dtype=float
    )
    archive_quat = Rotation.from_euler(
        "xyz", archived_final["euler_rpy"]
    ).as_quat()

    last_recorded_xyz, last_recorded_quat = archived_samples[-1]

    replay = {
        "replay_valid": True,
        "error": "",
        "trajectory_samples": len(archived_samples),
        "position_error_mm": 1000 * float(
            np.linalg.norm(last_recorded_xyz - archive_xyz)
        ),
        "rotation_error_deg": rotation_error_deg(
            last_recorded_quat, archive_quat
        ),
        "final_engine_position_error_mm": 1000 * float(
            np.linalg.norm(positions[-1] - archive_xyz)
        ),
        "final_engine_rotation_error_deg": rotation_error_deg(
            orientations[-1], archive_quat
        ),
        "final_table_contacts": final_table_contacts,
        "final_table_contact_present": final_table_contacts > 0,
    }

    ratios = np.asarray(force_ratios, dtype=float)
    residuals = np.abs(ratios - 1.0)

    force = {
        "valid": True,
        "error": "",
        "mass_kg": mass,
        "weight_N": weight,
        "observations": len(ratios),
        "mean_total_force_ratio": float(np.mean(ratios)),
        "min_total_force_ratio": float(np.min(ratios)),
        "max_total_force_ratio": float(np.max(ratios)),
        "mean_corrected_residual": float(np.mean(residuals)),
        "max_corrected_residual": float(np.max(residuals)),
        "max_horizontal_force_N": float(max(horizontal_forces)),
        "max_linear_speed_m_s": float(max(sampled_linear)),
        "max_angular_speed_rad_s": float(max(sampled_angular)),
        "min_positive_contacts": int(min(positive_contacts)),
    }

    P = np.asarray(positions)

    rotation_steps = [
        rotation_error_deg(orientations[i + 1], orientations[i])
        for i in range(119)
    ]

    peak_linear = max(linear_speeds)
    peak_angular = max(angular_speeds)
    peak_linear_20 = max(sampled_linear)
    peak_angular_20 = max(sampled_angular)

    motion = {
        "valid": True,
        "error": "",
        "samples_240Hz": len(positions),
        "samples_20Hz": len(sampled_linear),
        "peak_linear_240_m_s": peak_linear,
        "peak_linear_20_m_s": peak_linear_20,
        "peak_angular_240_rad_s": peak_angular,
        "peak_angular_20_rad_s": peak_angular_20,
        "linear_peak_ratio": (
            peak_linear / peak_linear_20
            if peak_linear_20 > 0 else np.nan
        ),
        "angular_peak_ratio": (
            peak_angular / peak_angular_20
            if peak_angular_20 > 0 else np.nan
        ),
        "mean_linear_m_s": float(np.mean(linear_speeds)),
        "mean_angular_rad_s": float(np.mean(angular_speeds)),
        "first_half_mean_linear_m_s": float(
            np.mean(linear_speeds[:60])
        ),
        "last_half_mean_linear_m_s": float(
            np.mean(linear_speeds[60:])
        ),
        "first_half_mean_angular_rad_s": float(
            np.mean(angular_speeds[:60])
        ),
        "last_half_mean_angular_rad_s": float(
            np.mean(angular_speeds[60:])
        ),
        "accumulated_translation_m": float(
            np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
        ),
        "net_translation_m": float(
            np.linalg.norm(P[-1] - P[0])
        ),
        "accumulated_rotation_deg": float(sum(rotation_steps)),
        "net_rotation_deg": rotation_error_deg(
            orientations[-1], orientations[0]
        ),
    }

    return replay, force, motion


def main():
    manifest = pd.read_csv(MANIFEST)

    assert len(manifest) == 115
    assert not manifest.duplicated(
        ["object", "archive_index"]
    ).any()

    for path in OUTPUTS:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    replay_rows, force_rows, motion_rows = [], [], []

    for obj, subset in manifest.groupby("object", sort=True):
        archive_path = (
            ROOT / "paper_runs" / obj / "data_export.pkl"
        )

        with open(archive_path, "rb") as f:
            archive = pickle.load(f)

        env = None

        try:
            env = ObjectPlacement(
                object_name="ycb_" + obj,
                target_name="table",
                display=False,
                use_concave_mesh=False
            )

            for row in subset.itertuples(index=False):
                idx = int(row.archive_index)

                key = {
                    "object": obj,
                    "archive_index": idx,
                }

                try:
                    replay, force, motion = run_placement(
                        env,
                        archive["init_6dof_pos"][idx],
                        archive["final_6dof_pos"][idx]
                    )

                    print(
                        f"OK: {obj} index={idx} "
                        f"peak={motion['peak_linear_240_m_s']*1000:.6f} mm/s",
                        flush=True
                    )

                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    print(
                        f"FAILED: {obj} index={idx}: {message}",
                        flush=True
                    )

                    replay = {
                        "replay_valid": False, "error": message
                    }
                    force = {"valid": False, "error": message}
                    motion = {"valid": False, "error": message}

                replay_rows.append({**key, **replay})
                force_rows.append({**key, **force})
                motion_rows.append({**key, **motion})

        finally:
            if env is not None:
                env.close()

    assert len(replay_rows) == 115
    assert len(force_rows) == 115
    assert len(motion_rows) == 115

    datasets = [
        pd.DataFrame(replay_rows),
        pd.DataFrame(force_rows),
        pd.DataFrame(motion_rows),
    ]

    for df in datasets:
        assert not df.duplicated(
            ["object", "archive_index"]
        ).any()

    for path, df in zip(OUTPUTS, datasets):
        with path.open("x", newline="", encoding="utf-8") as f:
            df.to_csv(f, index=False)

        print(f"SAVED: {path.name}, rows={len(df)}")

    print("\nSuccessful replay records:",
          int(datasets[0]["replay_valid"].sum()))
    print("Successful contact-force records:",
          int(datasets[1]["valid"].sum()))
    print("Successful full-rate records:",
          int(datasets[2]["valid"].sum()))


if __name__ == "__main__":
    main()
