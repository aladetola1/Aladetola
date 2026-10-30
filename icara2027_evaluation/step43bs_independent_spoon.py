import sys, json, pickle, csv
from pathlib import Path
import numpy as np
import pandas as pd
import pybullet_data

here = Path.cwd()
root = here.parent.resolve()
sys.path.insert(0, str(root))

from environments.src.object_placement import ObjectPlacement

manifest = json.loads(
    (here / "icara2027_spoon_grasp_protocol_frozen.json")
    .read_text(encoding="utf-8")
)

with (root / "paper_runs/spoon/data_export.pkl").open("rb") as f:
    archive = pickle.load(f)

indices = manifest["held_out_indices"]
development = set(manifest["development_indices"])
assert len(indices) == 24
assert not (set(indices) & development)

methods = manifest["methods"]
cfg = manifest["search"]
criteria = manifest["success"]

output = here / "step43bs_independent_spoon_results.csv"
if output.exists():
    raise FileExistsError(
        "Results already exist; refusing to overwrite: " + str(output)
    )

HOME = np.array([0, -.55, 0, -2.15, 0, 1.65, .78])
LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
               -2.8973, -.0175, -2.8973])
HI = np.array([2.8973, 1.7628, 2.8973, -.0698,
               2.8973, 3.7525, 2.8973])

fields = [
    "archive_index", "methods", "status", "success",
    "geometric_candidates", "static_candidates_checked",
    "chosen_angle_deg", "chosen_width_mm",
    "approach_collision_frames", "approach_spoon_motion_mm",
    "approach_ee_error_mm", "minimum_table_gap_mm",
    "closure_bilateral_frames", "lift_bilateral_frames",
    "hold_bilateral_frames", "hold_table_frames",
    "minimum_hold_rise_mm", "final_rise_mm",
    "maximum_slip_mm", "error"
]

def trial(index):
    row = {k: "" for k in fields}
    row["archive_index"] = index
    row["methods"] = ";".join(
        name.replace("selected_", "")
        for name, selected in methods.items()
        if index in selected
    )
    row["success"] = False
    env = None

    try:
        env = ObjectPlacement(
            object_name="ycb_spoon",
            target_name="table",
            display=False,
            use_concave_mesh=False
        )
        env.reset()
        env.set_6dof_obj_pose(
            archive["init_6dof_pos"][index]
        )
        p = env.bullet_client
        p.setTimeStep(1 / manifest["physics_hz"])
        p.setGravity(0, 0, -9.81)

        for _ in range(manifest["settling_steps"]):
            p.stepSimulation()

        obj = env.obj_id
        table = env.sim_engine.target_id
        initial = np.asarray(
            p.getBasePositionAndOrientation(obj)[0]
        )

        mn, mx = p.getAABB(obj)
        mn, mx = np.asarray(mn), np.asarray(mx)

        zlo = max(.225, float(mn[2] + .003))
        zhi = float(mx[2] - .003)
        if zlo >= zhi:
            row["status"] = "NO_HEIGHT_RANGE"
            return row

        center_xy = (mn[:2] + mx[:2]) / 2
        candidates = []

        for z in np.linspace(
            zlo, zhi, cfg["sampled_heights"]
        ):
            for angle in cfg["angles_deg"]:
                t = np.radians(angle)
                axis = np.array(
                    [np.cos(t), np.sin(t), 0.]
                )
                side = np.array(
                    [-np.sin(t), np.cos(t), 0.]
                )

                for offset in cfg["lateral_offsets_m"]:
                    center = (
                        np.array([*center_xy, z])
                        + offset * side
                    )
                    a = center - .25 * axis
                    b = center + .25 * axis
                    ha = p.rayTest(
                        a.tolist(), b.tolist()
                    )[0]
                    hb = p.rayTest(
                        b.tolist(), a.tolist()
                    )[0]

                    if ha[0] != obj or hb[0] != obj:
                        continue

                    A = np.asarray(ha[3])
                    B = np.asarray(hb[3])
                    width = np.linalg.norm(B - A)
                    if not (
                        cfg["min_contact_width_mm"] / 1000
                        <= width
                        <= cfg["max_contact_width_mm"] / 1000
                    ):
                        continue

                    direction = (B - A) / width
                    na = np.asarray(ha[4])
                    nb = np.asarray(hb[4])

                    align = min(
                        float(-na @ direction),
                        float(nb @ direction)
                    )
                    dot = float(na @ nb)

                    if (
                        align < cfg["normal_alignment_min"]
                        or dot > cfg["normal_dot_max"]
                    ):
                        continue

                    mid = (A + B) / 2
                    candidates.append((
                        (-align, abs(width - .020), angle,
                         float(z), float(offset)),
                        angle, mid, width
                    ))

        candidates.sort(key=lambda item: item[0])
        row["geometric_candidates"] = len(candidates)

        if not candidates:
            row["status"] = "NO_GEOMETRIC_CANDIDATE"
            return row

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )
        robot = p.loadURDF(
            "franka_panda/panda.urdf",
            [-.5, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )

        def set_configuration(q):
            for j, v in enumerate(q):
                p.resetJointState(robot, j, float(v))
            for j in (9, 10):
                p.resetJointState(robot, j, .015)
            p.performCollisionDetection()

        def solve(position, orientation):
            return np.asarray(
                p.calculateInverseKinematics(
                    robot, 11,
                    np.asarray(position).tolist(),
                    orientation,
                    lowerLimits=LO.tolist(),
                    upperLimits=HI.tolist(),
                    jointRanges=(HI - LO).tolist(),
                    restPoses=HOME.tolist(),
                    maxNumIterations=250,
                    residualThreshold=1e-6
                )[:7]
            )

        def self_collision():
            return any(
                c[8] < -.0001
                and tuple(sorted((c[3], c[4])))
                != (6, 8)
                for c in p.getContactPoints(robot, robot)
            )

        selected = None
        maximum = cfg["max_collision_checked_candidates"]

        for number, (_, angle, mid, width) in enumerate(
            candidates[:maximum], 1
        ):
            row["static_candidates_checked"] = number
            orn = p.getQuaternionFromEuler([
                np.pi, 0, np.radians(angle - 90)
            ])
            set_configuration(HOME)
            q = solve(mid, orn)
            set_configuration(q)

            actual = np.asarray(
                p.getLinkState(
                    robot, 11,
                    computeForwardKinematics=True
                )[4]
            )
            err = np.linalg.norm(actual - mid) * 1000

            sc = p.getClosestPoints(robot, obj, .10)
            tc = p.getClosestPoints(robot, table, .10)

            sg = min(
                (c[8] for c in sc),
                default=float("inf")
            )
            tg = min(
                (c[8] for c in tc),
                default=float("inf")
            )

            valid = (
                err < 2
                and np.all(q >= LO - 1e-6)
                and np.all(q <= HI + 1e-6)
                and sg >= -.0001
                and tg >= -.0001
                and not self_collision()
            )

            if valid:
                selected = (mid, orn, angle, width)
                break

        if selected is None:
            row["status"] = "NO_STATIC_FEASIBLE_CANDIDATE"
            return row

        grasp, orn, angle, width = selected
        row["chosen_angle_deg"] = angle
        row["chosen_width_mm"] = round(width * 1000, 4)

        # Return to home once, before dynamic execution.
        set_configuration(HOME)

        pre = grasp + np.array([0, 0, .080])
        qpre = solve(pre, orn)
        qgrasp = solve(grasp, orn)
        descent = [
            solve(pre + (grasp - pre) * i / 240, orn)
            for i in range(241)
        ]
        lifting = [
            solve(
                grasp + np.array([0, 0, .070 * i / 240]),
                orn
            )
            for i in range(241)
        ]

        def command(q, close=False):
            p.setJointMotorControlArray(
                robot, list(range(7)),
                p.POSITION_CONTROL,
                targetPositions=np.asarray(q).tolist(),
                forces=[150.] * 7,
                positionGains=[.15] * 7
            )
            for j in (9, 10):
                p.setJointMotorControl2(
                    robot, j, p.POSITION_CONTROL,
                    targetPosition=0.0 if close else .015,
                    force=40
                )

        def xyz():
            return np.asarray(
                p.getBasePositionAndOrientation(obj)[0]
            )

        def ee():
            return np.asarray(
                p.getLinkState(
                    robot, 11,
                    computeForwardKinematics=True
                )[4]
            )

        def bilateral():
            contacts = p.getContactPoints(robot, obj)
            return (
                any(c[3] == 9 for c in contacts)
                and any(c[3] == 10 for c in contacts)
            )

        collision_frames = 0
        max_motion = 0.
        min_table_gap = float("inf")

        def audit():
            nonlocal collision_frames, max_motion, min_table_gap
            collision_frames += int(
                bool(p.getContactPoints(robot, obj))
                or bool(p.getContactPoints(robot, table))
                or self_collision()
            )
            max_motion = max(
                max_motion,
                np.linalg.norm(xyz() - initial) * 1000
            )
            tc = p.getClosestPoints(robot, table, .10)
            min_table_gap = min(
                min_table_gap,
                min(
                    (c[8] for c in tc),
                    default=float("inf")
                )
            )

        # Dynamic home-to-pregrasp
        for i in range(480):
            a = (i + 1) / 480
            smooth = 3*a*a - 2*a*a*a
            command(HOME + smooth * (qpre - HOME))
            p.stepSimulation()
            audit()

        for _ in range(240):
            command(qpre)
            p.stepSimulation()
            audit()

        # Dynamic pregrasp-to-grasp
        for i in range(240):
            command(descent[i + 1])
            p.stepSimulation()
            audit()

        for _ in range(240):
            command(qgrasp)
            p.stepSimulation()
            audit()

        error = np.linalg.norm(ee() - grasp) * 1000

        row["approach_collision_frames"] = collision_frames
        row["approach_spoon_motion_mm"] = round(max_motion, 4)
        row["approach_ee_error_mm"] = round(error, 4)
        row["minimum_table_gap_mm"] = round(
            min_table_gap * 1000, 4
        )

        approach_ok = (
            collision_frames <=
                criteria["approach_collision_frames_max"]
            and max_motion <
                criteria["approach_displacement_mm_max_exclusive"]
            and error <
                criteria["approach_position_error_mm_max_exclusive"]
            and min_table_gap >= -.1e-3
        )

        if not approach_ok:
            row["status"] = "APPROACH_FAILED"
            return row

        # Closure
        close_frames = 0
        for _ in range(240):
            command(qgrasp, True)
            p.stepSimulation()
            close_frames += int(bilateral())

        row["closure_bilateral_frames"] = close_frames

        # Lift
        lift_frames = 0
        for i in range(240):
            command(lifting[i + 1], True)
            p.stepSimulation()
            lift_frames += int(bilateral())

        row["lift_bilateral_frames"] = lift_frames

        end_spoon = xyz()
        end_ee = ee()
        relative0 = end_spoon - end_ee
        hold_q = np.array([
            p.getJointState(robot, j)[0]
            for j in range(7)
        ])

        # Two-second retention
        hold_frames = 0
        table_frames = 0
        min_rise = float("inf")
        max_slip = 0.

        for _ in range(manifest["retention_steps"]):
            command(hold_q, True)
            p.stepSimulation()

            sp = xyz()
            rise = (sp[2] - initial[2]) * 1000
            slip = np.linalg.norm(
                (sp - ee()) - relative0
            ) * 1000

            hold_frames += int(bilateral())
            table_frames += int(
                bool(p.getContactPoints(obj, table))
            )
            min_rise = min(min_rise, rise)
            max_slip = max(max_slip, slip)

        row["hold_bilateral_frames"] = hold_frames
        row["hold_table_frames"] = table_frames
        row["minimum_hold_rise_mm"] = round(min_rise, 4)
        row["final_rise_mm"] = round(
            (xyz()[2] - initial[2]) * 1000, 4
        )
        row["maximum_slip_mm"] = round(max_slip, 4)

        success = (
            close_frames > 0
            and lift_frames >=
                criteria["bilateral_lift_frames_min"]
            and hold_frames >=
                criteria["bilateral_hold_frames_min"]
            and table_frames <=
                criteria["table_contact_hold_frames_max"]
            and min_rise >
                criteria["minimum_lift_mm_exclusive"]
            and max_slip <
                criteria["relative_translation_slip_mm_max_exclusive"]
        )

        row["success"] = bool(success)
        row["status"] = (
            "SUCCESS" if success else "GRASP_FAILED"
        )
        return row

    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return row

    finally:
        if env is not None:
            env.close()


print("\nSTEP 43BS — FROZEN HELD-OUT SPOON EVALUATION", flush=True)
print("=" * 85, flush=True)
print("Placements:", len(indices), flush=True)
print("Development placements excluded:", development, flush=True)
print("No artificial fixed constraints", flush=True)

results = []

for number, index in enumerate(indices, 1):
    result = trial(index)
    results.append(result)

    # Write after every placement so results survive interruption.
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(
        f"[{number:02d}/{len(indices)}] "
        f"Index {index}: {result['status']} | "
        f"candidates={result['geometric_candidates']} | "
        f"checked={result['static_candidates_checked']} | "
        f"rise={result['final_rise_mm']} | "
        f"slip={result['maximum_slip_mm']}",
        flush=True
    )
    if result["error"]:
        print("  Error:", result["error"], flush=True)

print("\nMETHOD-LEVEL RESULTS")
print("-" * 85)

res = pd.DataFrame(results)
for method, selected in methods.items():
    subset = res[
        res["archive_index"].isin(selected)
    ]
    n = len(subset)
    successes = int(subset["success"].sum())
    print(
        f"{method:29s} "
        f"{successes}/{n} successes "
        f"({100*successes/n:.1f}%)"
    )

print("\nStatus distribution:")
print(res["status"].value_counts().to_string())

print("\nSaved:", output.resolve())
print("Original evaluation CSVs and frozen protocol unchanged.")
print("Results concern a restricted simulated grasp planner.")
