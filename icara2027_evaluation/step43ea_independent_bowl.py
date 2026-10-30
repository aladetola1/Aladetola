import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pybullet_data
from scipy.spatial.transform import Rotation

here = Path(__file__).resolve().parent
root = here.parent.resolve()
sys.path.insert(0, str(root))

from environments.src.object_placement import ObjectPlacement

PROTOCOL_PATH = here / "icara2027_bowl_grasp_protocol_frozen.json"
EXPECTED_SHA = "443abc88abf2b09a72cc2d71f1e39b2774acd00a2e9f611aa124c0041ce734b5"
RESULT_PATH = here / "step43ea_independent_bowl_results.csv"

HOME = np.array([0, -.55, 0, -2.15, 0, 1.65, .78])
LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
               -2.8973, -.0175, -2.8973])
HI = np.array([2.8973, 1.7628, 2.8973, -.0698,
               2.8973, 3.7525, 2.8973])

FIELDS = [
    "archive_index", "methods", "status", "success",
    "geometric_candidates", "static_candidates_checked",
    "static_configurations_checked",
    "chosen_candidate_rank", "chosen_angle_deg",
    "chosen_wrist_tilt_deg", "chosen_height_adjustment_mm",
    "chosen_width_mm", "chosen_grasp_x", "chosen_grasp_y",
    "chosen_grasp_z", "geometric_contact_opening_mm",
    "approach_physical_object_frames",
    "approach_table_frames", "approach_self_collision_frames",
    "approach_max_object_motion_mm",
    "approach_position_error_mm",
    "minimum_object_gap_mm", "minimum_table_gap_mm",
    "closure_bilateral_force_frames",
    "closure_self_collision_frames",
    "lift_bilateral_force_frames",
    "lift_table_frames", "lift_self_collision_frames",
    "hold_bilateral_force_frames",
    "hold_table_frames", "hold_self_collision_frames",
    "minimum_hold_rise_mm", "final_rise_mm",
    "maximum_cumulative_translation_slip_mm",
    "maximum_cumulative_rotation_slip_deg",
    "maximum_hold_only_translation_slip_mm",
    "maximum_hold_only_rotation_slip_deg",
    "failed_checks", "error"
]


def load_protocol():
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA:
        raise RuntimeError(
            "Frozen protocol SHA256 mismatch: " + digest
        )

    protocol = json.loads(PROTOCOL_PATH.read_text())

    assert protocol["status"] == "protocol_frozen_before_independent_trials"
    assert protocol["object"] == "bowl"
    assert protocol["development_indices"] == [130]
    assert len(protocol["held_out_indices"]) == 32
    assert len(set(protocol["held_out_indices"])) == 32
    assert 130 not in protocol["held_out_indices"]

    s = protocol["search"]
    assert s["sampled_heights"] == 21
    assert s["angles_deg"] == list(range(0, 360, 10))
    assert s["wrist_tilts_deg"] == [0, 45, -45]
    assert s["height_adjustments_mm"] == [0, -20]
    assert s["finger_axis_centering_offset_mm"] == -5
    assert s["max_collision_checked_geometric_candidates"] == 20
    assert s["candidate_selection"] == "first_static_feasible_in_fixed_order"
    assert s["dynamic_grasp_attempts_per_placement"] == 1
    assert s["dynamic_retry_after_failure"] is False

    assert protocol["physics"]["object_lateral_friction"] == .5
    assert protocol["physics"]["no_friction_modification"] is True
    assert protocol["physics"]["no_artificial_object_attachment"] is True
    assert protocol["success"]["slip_reference"] == "end_of_finger_closure"
    assert protocol["success"]["slip_evaluation_interval"] == "entire_lift_and_hold"

    all_selected = set()
    expected = {
        "selected_GFI": 10,
        "selected_Centrality-only": 10,
        "selected_Height-only": 10,
        "selected_Uprightness-only": 9,
        "selected_Height+Centrality": 10
    }
    for method, count in expected.items():
        selected = protocol["methods"][method]
        assert len(selected) == count
        assert len(set(selected)) == count
        assert set(selected).issubset(protocol["held_out_indices"])
        all_selected.update(selected)

    assert all_selected == set(protocol["held_out_indices"])

    source = here / protocol["source_csv"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol[
        "source_csv_sha256"
    ], "Consolidated source CSV changed"

    return protocol, digest


def evaluate(index, protocol, archive):
    row = {field: "" for field in FIELDS}
    row["archive_index"] = index
    row["success"] = False
    row["methods"] = ";".join(
        name.replace("selected_", "")
        for name, values in protocol["methods"].items()
        if index in values
    )

    cfg = protocol["search"]
    crit = protocol["success"]
    control = protocol["control"]
    hz = int(protocol["physics_hz"])
    open_joint = protocol["gripper_opening_mm"] / 2000.
    preheight = protocol["pregrasp_height_mm"] / 1000.
    liftheight = protocol["lift_height_mm"] / 1000.
    min_force = crit["bilateral_contact_force_N_min_exclusive"]

    env = None

    try:
        env = ObjectPlacement(
            object_name="ycb_bowl",
            target_name="table",
            display=False,
            use_concave_mesh=False
        )
        env.reset()
        env.set_6dof_obj_pose(
            archive["init_6dof_pos"][index]
        )

        p = env.bullet_client
        p.setTimeStep(1 / hz)
        p.setGravity(0, 0, -9.81)

        for _ in range(protocol["settling_steps"]):
            p.stepSimulation()

        obj = env.obj_id
        table = env.sim_engine.target_id

        if abs(p.getDynamicsInfo(obj, -1)[1] - .5) > 1e-8:
            raise RuntimeError("Baseline bowl friction mismatch")

        initial = np.asarray(
            p.getBasePositionAndOrientation(obj)[0]
        )

        mn, mx = map(np.asarray, p.getAABB(obj))
        table_top = float(p.getAABB(table)[1][2])
        center = (mn[:2] + mx[:2]) / 2

        zlo = max(
            float(mn[2]) + cfg["height_margin_from_object_aabb_m"],
            table_top + cfg["height_min_above_table_m"]
        )
        zhi = float(mx[2]) - cfg["height_margin_from_object_aabb_m"]

        if zlo >= zhi:
            row["status"] = "NO_HEIGHT_RANGE"
            row["failed_checks"] = "no_height_range"
            return row

        candidates = []

        for z in np.linspace(zlo, zhi, cfg["sampled_heights"]):
            for angle in cfg["angles_deg"]:
                theta = np.radians(angle)
                axis = np.array([
                    np.cos(theta), np.sin(theta), 0.
                ])
                inner = np.array([center[0], center[1], z])
                outer = inner + cfg["radial_ray_length_m"] * axis

                a = p.rayTest(inner.tolist(), outer.tolist())[0]
                b = p.rayTest(outer.tolist(), inner.tolist())[0]

                if a[0] != obj or b[0] != obj:
                    continue

                A = np.asarray(a[3])
                B = np.asarray(b[3])
                nA = np.asarray(a[4])
                nB = np.asarray(b[4])

                width = float((B - A) @ axis)
                if not (
                    cfg["min_contact_width_mm"] / 1000.
                    <= width
                    <= cfg["max_contact_width_mm"] / 1000.
                ):
                    continue

                if (
                    float(nA @ nB) > cfg["normal_dot_max"]
                    or float(-nA @ axis) < cfg["normal_alignment_min"]
                    or float(nB @ axis) < cfg["normal_alignment_min"]
                ):
                    continue

                candidates.append((
                    -float(z),
                    abs(width * 1000 - 20),
                    int(angle),
                    (A + B) / 2,
                    width
                ))

        candidates.sort(key=lambda c: c[:3])
        row["geometric_candidates"] = len(candidates)

        if not candidates:
            row["status"] = "NO_GEOMETRIC_CANDIDATE"
            row["failed_checks"] = "no_geometric_candidate"
            return row

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        robot = p.loadURDF(
            "franka_panda/panda.urdf",
            control["panda_base_xyz_m"],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )

        def set_arm(q, opening=80):
            for j, v in enumerate(q):
                p.resetJointState(robot, j, float(v))
            for j in (9, 10):
                p.resetJointState(robot, j, opening / 2000.)
            p.performCollisionDetection()

        def ik(target, orientation):
            return np.asarray(
                p.calculateInverseKinematics(
                    robot, 11, np.asarray(target).tolist(), orientation,
                    lowerLimits=LO.tolist(),
                    upperLimits=HI.tolist(),
                    jointRanges=(HI-LO).tolist(),
                    restPoses=HOME.tolist(),
                    maxNumIterations=250,
                    residualThreshold=1e-6
                )[:7]
            )

        def gap(body, link=None):
            kw = {} if link is None else {"linkIndexA": link}
            cs = p.getClosestPoints(robot, body, .15, **kw)
            return min(
                (float(c[8]) * 1000 for c in cs),
                default=float("inf")
            )

        def selfbad():
            return any(
                c[8] < -.0001
                and tuple(sorted((c[3], c[4]))) != (6, 8)
                for c in p.getContactPoints(robot, robot)
            )

        chosen = None
        configurations_checked = 0

        for rank, (_, _, angle, midpoint, width) in enumerate(
            candidates[:cfg["max_collision_checked_geometric_candidates"]],
            start=1
        ):
            if chosen is not None:
                break

            for tilt in cfg["wrist_tilts_deg"]:
                if chosen is not None:
                    break

                orientation = p.getQuaternionFromEuler([
                    np.pi,
                    np.radians(tilt),
                    np.radians(angle - 90)
                ])

                for adjustment in cfg["height_adjustments_mm"]:
                    configurations_checked += 1

                    target = (
                        np.asarray(midpoint)
                        + np.array([0., 0., adjustment / 1000.])
                    )

                    if target[2] < (
                        table_top
                        + cfg["minimum_adjusted_grasp_height_above_table_mm"] / 1000.
                    ):
                        continue

                    set_arm(HOME)
                    q = ik(target, orientation)
                    set_arm(q, 40)

                    f9 = np.asarray(p.getLinkState(robot, 9)[4])
                    f10 = np.asarray(p.getLinkState(robot, 10)[4])
                    finger_axis = f10 - f9
                    finger_axis[2] = 0.

                    norm = np.linalg.norm(finger_axis)
                    if norm < 1e-8:
                        continue

                    finger_axis /= norm
                    target = (
                        target
                        + finger_axis
                        * cfg["finger_axis_centering_offset_mm"] / 1000.
                    )
                    pre = target + np.array([0., 0., preheight])

                    valid = True

                    for alpha in np.linspace(
                        0, 1, cfg["approach_static_waypoints"]
                    ):
                        waypoint = pre + alpha * (target - pre)
                        set_arm(HOME)
                        q = ik(waypoint, orientation)
                        set_arm(q)

                        state = p.getLinkState(
                            robot, 11, computeForwardKinematics=True
                        )

                        position_error = np.linalg.norm(
                            np.asarray(state[4]) - waypoint
                        ) * 1000.

                        rotation_error = np.degrees(
                            (
                                Rotation.from_quat(state[5]).inv()
                                * Rotation.from_quat(orientation)
                            ).magnitude()
                        )

                        if not (
                            np.isfinite(q).all()
                            and np.all(q >= LO - 1e-6)
                            and np.all(q <= HI + 1e-6)
                            and position_error
                            < cfg["static_position_error_mm_max_exclusive"]
                            and rotation_error
                            < cfg["static_rotation_error_deg_max_exclusive"]
                            and gap(obj)
                            >= cfg["static_penetration_tolerance_mm"]
                            and gap(table)
                            >= cfg["static_penetration_tolerance_mm"]
                            and not selfbad()
                        ):
                            valid = False
                            break

                    if not valid:
                        continue

                    set_arm(HOME)
                    qgrasp = ik(target, orientation)
                    contact_opening = None
                    closing_collision = False

                    for opening in cfg["closure_openings_screen_mm"]:
                        set_arm(qgrasp, opening)

                        if (
                            gap(table)
                            < cfg["static_penetration_tolerance_mm"]
                            or gap(obj, 8)
                            < cfg["static_penetration_tolerance_mm"]
                            or selfbad()
                        ):
                            closing_collision = True
                            break

                        if gap(obj, 9) <= 0 and gap(obj, 10) <= 0:
                            contact_opening = opening
                            break

                    if closing_collision or contact_opening is None:
                        continue

                    chosen = (
                        rank, angle, tilt, adjustment,
                        target, pre, orientation, width,
                        contact_opening
                    )
                    break

        row["static_candidates_checked"] = min(
            len(candidates),
            cfg["max_collision_checked_geometric_candidates"]
        ) if chosen is None else chosen[0]
        row["static_configurations_checked"] = configurations_checked

        if chosen is None:
            row["status"] = "NO_STATIC_FEASIBLE_CANDIDATE"
            row["failed_checks"] = "static_feasibility"
            return row

        (
            rank, angle, tilt, adjustment,
            grasp, pre, orientation, width, opening
        ) = chosen

        row.update({
            "chosen_candidate_rank": rank,
            "chosen_angle_deg": angle,
            "chosen_wrist_tilt_deg": tilt,
            "chosen_height_adjustment_mm": adjustment,
            "chosen_width_mm": round(width * 1000, 4),
            "chosen_grasp_x": round(float(grasp[0]), 7),
            "chosen_grasp_y": round(float(grasp[1]), 7),
            "chosen_grasp_z": round(float(grasp[2]), 7),
            "geometric_contact_opening_mm": opening
        })

        set_arm(HOME)

        qpre = ik(pre, orientation)
        qgrasp = ik(grasp, orientation)

        descent = [
            ik(pre + (grasp - pre) * i / control["descent_frames"],
               orientation)
            for i in range(control["descent_frames"] + 1)
        ]

        lift = [
            ik(grasp + np.array([
                0., 0., liftheight * i / control["lift_frames"]
            ]), orientation)
            for i in range(control["lift_frames"] + 1)
        ]

        def command(q, close=False):
            p.setJointMotorControlArray(
                robot,
                list(range(7)),
                p.POSITION_CONTROL,
                targetPositions=np.asarray(q).tolist(),
                forces=[control["arm_motor_force_Nm"]] * 7,
                positionGains=[control["arm_position_gain"]] * 7
            )
            for j in (9, 10):
                p.setJointMotorControl2(
                    robot, j, p.POSITION_CONTROL,
                    targetPosition=0. if close else open_joint,
                    force=control["finger_motor_force_N"]
                )

        def object_pose():
            xyz, quat = p.getBasePositionAndOrientation(obj)
            return np.asarray(xyz), Rotation.from_quat(quat)

        def ee_pose():
            state = p.getLinkState(
                robot, 11, computeForwardKinematics=True
            )
            return (
                np.asarray(state[4]),
                Rotation.from_quat(state[5])
            )

        def physical(cs):
            return any(
                c[8] < -.0001 or c[9] > min_force
                for c in cs
            )

        def contact_state():
            cs = p.getContactPoints(robot, obj)
            bilateral = (
                any(c[3] == 9 and c[9] > min_force for c in cs)
                and any(c[3] == 10 and c[9] > min_force for c in cs)
            )
            table_touch = physical(
                p.getContactPoints(obj, table)
            )
            return bilateral, table_touch, selfbad()

        approach = {
            "object": 0, "table": 0, "self": 0,
            "motion": 0.,
            "object_gap": float("inf"),
            "table_gap": float("inf")
        }

        def audit_approach():
            approach["object"] += int(
                physical(p.getContactPoints(robot, obj))
            )
            approach["table"] += int(
                physical(p.getContactPoints(robot, table))
            )
            approach["self"] += int(selfbad())
            approach["motion"] = max(
                approach["motion"],
                np.linalg.norm(object_pose()[0] - initial) * 1000
            )
            approach["object_gap"] = min(
                approach["object_gap"], gap(obj)
            )
            approach["table_gap"] = min(
                approach["table_gap"], gap(table)
            )

        set_arm(HOME)

        n = control["home_to_pregrasp_frames"]
        for i in range(n):
            alpha = (i + 1) / n
            smooth = 3 * alpha * alpha - 2 * alpha ** 3
            command(HOME + smooth * (qpre - HOME))
            p.stepSimulation()
            audit_approach()

        for _ in range(control["pregrasp_settle_frames"]):
            command(qpre)
            p.stepSimulation()
            audit_approach()

        for q in descent[1:]:
            command(q)
            p.stepSimulation()
            audit_approach()

        for _ in range(control["grasp_settle_frames"]):
            command(qgrasp)
            p.stepSimulation()
            audit_approach()

        position_error = np.linalg.norm(
            ee_pose()[0] - grasp
        ) * 1000

        row.update({
            "approach_physical_object_frames": approach["object"],
            "approach_table_frames": approach["table"],
            "approach_self_collision_frames": approach["self"],
            "approach_max_object_motion_mm": round(approach["motion"], 4),
            "approach_position_error_mm": round(position_error, 4),
            "minimum_object_gap_mm": round(approach["object_gap"], 4),
            "minimum_table_gap_mm": round(approach["table_gap"], 4)
        })

        approach_checks = {
            "object_contact": approach["object"] == 0,
            "table_contact": approach["table"] == 0,
            "self_collision": approach["self"] == 0,
            "object_clearance": approach["object_gap"]
                >= crit["approach_object_penetration_mm_limit"],
            "object_motion": approach["motion"]
                < crit["approach_displacement_mm_max_exclusive"],
            "position_error": position_error
                < crit["approach_position_error_mm_max_exclusive"]
        }

        failures = [
            key for key, ok in approach_checks.items() if not ok
        ]

        if failures:
            row["status"] = "APPROACH_FAILED"
            row["failed_checks"] = ";".join(failures)
            return row

        closure = 0
        closure_self = 0

        for _ in range(control["closure_frames"]):
            command(qgrasp, close=True)
            p.stepSimulation()
            both, _, sc = contact_state()
            closure += int(both)
            closure_self += int(sc)

        start_x, start_r = object_pose()
        start_e, start_er = ee_pose()

        relative_xyz0 = start_er.inv().apply(start_x - start_e)
        relative_rot0 = start_er.inv() * start_r

        def measurement():
            x, r = object_pose()
            e, er = ee_pose()
            relative_xyz = er.inv().apply(x - e)
            relative_rot = er.inv() * r

            return (
                (x[2] - initial[2]) * 1000,
                np.linalg.norm(relative_xyz - relative_xyz0) * 1000,
                np.degrees(
                    (relative_rot0.inv() * relative_rot).magnitude()
                )
            )

        lift_bilateral = 0
        lift_table = 0
        lift_self = 0
        max_slip = 0.
        max_rot = 0.

        for q in lift[1:]:
            command(q, close=True)
            p.stepSimulation()

            both, touch, sc = contact_state()
            lift_bilateral += int(both)
            lift_table += int(touch)
            lift_self += int(sc)

            rise, slip, rotation = measurement()
            max_slip = max(max_slip, slip)
            max_rot = max(max_rot, rotation)

        hold_q = np.array([
            p.getJointState(robot, j)[0]
            for j in range(7)
        ])

        lift_x, lift_r = object_pose()
        lift_e, lift_er = ee_pose()

        hold_xyz0 = lift_er.inv().apply(lift_x - lift_e)
        hold_rot0 = lift_er.inv() * lift_r

        hold_bilateral = 0
        hold_table = 0
        hold_self = 0
        min_hold_rise = float("inf")
        max_hold_only_slip = 0.
        max_hold_only_rot = 0.

        for _ in range(control["hold_frames"]):
            command(hold_q, close=True)
            p.stepSimulation()

            both, touch, sc = contact_state()
            hold_bilateral += int(both)
            hold_table += int(touch)
            hold_self += int(sc)

            rise, slip, rotation = measurement()
            min_hold_rise = min(min_hold_rise, rise)
            max_slip = max(max_slip, slip)
            max_rot = max(max_rot, rotation)

            x, r = object_pose()
            e, er = ee_pose()

            current_xyz = er.inv().apply(x - e)
            current_rot = er.inv() * r

            max_hold_only_slip = max(
                max_hold_only_slip,
                np.linalg.norm(current_xyz - hold_xyz0) * 1000
            )
            max_hold_only_rot = max(
                max_hold_only_rot,
                np.degrees(
                    (hold_rot0.inv() * current_rot).magnitude()
                )
            )

        final_rise = (
            object_pose()[0][2] - initial[2]
        ) * 1000

        row.update({
            "closure_bilateral_force_frames": closure,
            "closure_self_collision_frames": closure_self,
            "lift_bilateral_force_frames": lift_bilateral,
            "lift_table_frames": lift_table,
            "lift_self_collision_frames": lift_self,
            "hold_bilateral_force_frames": hold_bilateral,
            "hold_table_frames": hold_table,
            "hold_self_collision_frames": hold_self,
            "minimum_hold_rise_mm": round(min_hold_rise, 4),
            "final_rise_mm": round(final_rise, 4),
            "maximum_cumulative_translation_slip_mm":
                round(max_slip, 4),
            "maximum_cumulative_rotation_slip_deg":
                round(max_rot, 4),
            "maximum_hold_only_translation_slip_mm":
                round(max_hold_only_slip, 4),
            "maximum_hold_only_rotation_slip_deg":
                round(max_hold_only_rot, 4)
        })

        checks = {
            "closure": closure >= crit["closure_bilateral_frames_min"],
            "closure_self": closure_self == 0,
            "lift_contacts": lift_bilateral
                >= crit["lift_bilateral_frames_min"],
            "lift_self": lift_self == 0,
            "hold_contacts": hold_bilateral
                >= crit["hold_bilateral_frames_min"],
            "hold_table": hold_table
                <= crit["hold_table_contact_frames_max"],
            "hold_self": hold_self == 0,
            "lift_height": min_hold_rise
                > crit["minimum_retention_rise_mm_exclusive"],
            "translation_slip": max_slip
                < crit["relative_translation_slip_mm_max_exclusive"],
            "rotation_slip": max_rot
                < crit["relative_rotation_slip_deg_max_exclusive"]
        }

        failures = [name for name, ok in checks.items() if not ok]

        row["success"] = not failures
        row["status"] = "SUCCESS" if not failures else "GRASP_FAILED"
        row["failed_checks"] = ";".join(failures)
        return row

    except Exception as exc:
        row["status"] = "ERROR"
        row["success"] = False
        row["failed_checks"] = "execution_error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["audit", "development", "independent"],
        default="audit"
    )
    args = parser.parse_args()

    protocol, digest = load_protocol()

    print("\nSTEP 43EA — BOWL EVALUATOR", flush=True)
    print("=" * 85)
    print("Mode:", args.mode)
    print("Frozen protocol SHA256:", digest)
    print("Held-out placements:", len(protocol["held_out_indices"]))
    print("Development placement:", protocol["development_indices"])
    print("Dynamic retries:", protocol["search"]["dynamic_retry_after_failure"])
    print("Baseline friction:", protocol["physics"]["object_lateral_friction"])
    print("Cumulative slip reference:", protocol["success"]["slip_reference"])

    if args.mode == "audit":
        print("\nPROTOCOL AND SOURCE HASH AUDIT PASSED")
        print("No simulation executed.")
        print("No held-out placements accessed.")
        print("No results CSV created.")
        return

    with (root / protocol["archive_directory"] / "data_export.pkl").open(
        "rb"
    ) as f:
        archive = pickle.load(f)

    if args.mode == "development":
        indices = protocol["development_indices"]
    else:
        if RESULT_PATH.exists():
            raise FileExistsError(
                f"Refusing to overwrite experimental results: {RESULT_PATH}"
            )
        indices = protocol["held_out_indices"]

    rows = []
    for number, index in enumerate(indices, 1):
        result = evaluate(index, protocol, archive)
        rows.append(result)
        print(
            f"[{number}/{len(indices)}] "
            f"index={index} status={result['status']} "
            f"success={result['success']} "
            f"failed={result['failed_checks']}",
            flush=True
        )

    if args.mode == "development":
        print("\nDEVELOPMENT REPORT")
        print(json.dumps(rows, indent=2, default=str))
        print("No independent results written.")
        return

    if len(rows) != 32:
        raise RuntimeError("Incorrect number of held-out rows")

    with RESULT_PATH.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("\nINDEPENDENT BOWL EVALUATION COMPLETE")
    print("Results CSV:", RESULT_PATH.name)
    print("Successes:", sum(r["success"] is True for r in rows))
    print("Failures:", sum(r["success"] is False for r in rows))
    print("Errors:", sum(r["status"] == "ERROR" for r in rows))
    print("Frozen protocol unchanged.")


if __name__ == "__main__":
    main()
