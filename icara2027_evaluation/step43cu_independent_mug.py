import sys, json, pickle, csv
from pathlib import Path

import numpy as np
import pandas as pd
import pybullet_data
from scipy.spatial.transform import Rotation

here = Path.cwd()
root = here.parent.resolve()
sys.path.insert(0, str(root))

from environments.src.object_placement import ObjectPlacement

protocol_path = here / "icara2027_mug_grasp_protocol_frozen.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

assert protocol["object"] == "mug"
assert protocol["status"] == "protocol_frozen_before_independent_trials"

indices = [int(x) for x in protocol["held_out_indices"]]
development = set(protocol["development_indices"])
assert len(indices) == 34
assert not (set(indices) & development)

with (root / "paper_runs/mug/data_export.pkl").open("rb") as f:
    archive = pickle.load(f)

cfg = protocol["search"]
crit = protocol["success"]
methods = protocol["methods"]

output = here / "step43cu_independent_mug_results.csv"

if output.exists():
    raise FileExistsError(
        f"Results file already exists: {output}. "
        "Refusing to overwrite an experimental record."
    )

HOME = np.array([0,-.55,0,-2.15,0,1.65,.78])
LO = np.array([-2.8973,-1.7628,-2.8973,-3.0718,
               -2.8973,-.0175,-2.8973])
HI = np.array([2.8973,1.7628,2.8973,-.0698,
               2.8973,3.7525,2.8973])

HZ = int(protocol["physics_hz"])
OPEN = float(protocol["gripper_opening_mm"]) / 2000
PREHEIGHT = float(protocol["pregrasp_height_mm"]) / 1000
LIFTHEIGHT = float(protocol["lift_height_mm"]) / 1000
HOLD_STEPS = int(protocol["retention_steps"])
FORCE_MIN = float(crit["bilateral_contact_force_N_min_exclusive"])

fields = [
    "archive_index", "methods", "status", "success",
    "geometric_candidates", "static_candidates_checked",
    "chosen_angle_deg", "chosen_pitch_deg", "chosen_width_mm",
    "approach_proximity_frames",
    "approach_physical_object_frames",
    "approach_table_frames",
    "approach_self_collision_frames",
    "approach_max_object_motion_mm",
    "approach_position_error_mm",
    "minimum_object_gap_mm",
    "minimum_table_gap_mm",
    "closure_bilateral_force_frames",
    "lift_bilateral_force_frames",
    "hold_bilateral_force_frames",
    "hold_table_frames",
    "minimum_hold_rise_mm",
    "final_rise_mm",
    "maximum_translation_slip_mm",
    "maximum_rotation_slip_deg",
    "failed_checks", "error"
]

def evaluate(index):
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
            object_name="ycb_mug",
            target_name="table",
            display=False,
            use_concave_mesh=False
        )
        env.reset()
        env.set_6dof_obj_pose(
            archive["init_6dof_pos"][index]
        )

        p = env.bullet_client
        p.setTimeStep(1/HZ)
        p.setGravity(0,0,-9.81)

        for _ in range(int(protocol["settling_steps"])):
            p.stepSimulation()

        obj = env.obj_id
        table = env.sim_engine.target_id

        def object_pose():
            xyz, quat = p.getBasePositionAndOrientation(obj)
            return np.asarray(xyz), Rotation.from_quat(quat)

        initial, _ = object_pose()

        # --------------------------------------------------
        # 1. FROZEN RAY-BASED CANDIDATE SEARCH
        # --------------------------------------------------
        mn, mx = p.getAABB(obj)
        mn = np.asarray(mn)
        mx = np.asarray(mx)

        zlo = max(.225, float(mn[2]+.003))
        zhi = float(mx[2]-.003)

        if zlo >= zhi:
            row["status"] = "NO_HEIGHT_RANGE"
            row["failed_checks"] = "no_height_range"
            return row

        center_xy = (mn[:2]+mx[:2])/2
        candidates = []

        for z in np.linspace(
            zlo,zhi,int(cfg["sampled_heights"])
        ):
            for angle in cfg["angles_deg"]:
                t = np.radians(angle)
                axis = np.array([np.cos(t),np.sin(t),0.])
                side = np.array([-np.sin(t),np.cos(t),0.])

                for offset in cfg["lateral_offsets_m"]:
                    center = (
                        np.array([*center_xy,z])
                        + float(offset)*side
                    )

                    a = center-.30*axis
                    b = center+.30*axis

                    hit_a = p.rayTest(
                        a.tolist(),b.tolist()
                    )[0]
                    hit_b = p.rayTest(
                        b.tolist(),a.tolist()
                    )[0]

                    if hit_a[0] != obj or hit_b[0] != obj:
                        continue

                    A = np.asarray(hit_a[3])
                    B = np.asarray(hit_b[3])
                    width = float(np.linalg.norm(B-A))

                    if not (
                        cfg["min_contact_width_mm"]/1000
                        <= width
                        <= cfg["max_contact_width_mm"]/1000
                    ):
                        continue

                    direction = (B-A)/width
                    na = np.asarray(hit_a[4])
                    nb = np.asarray(hit_b[4])

                    alignment = min(
                        float(-na@direction),
                        float(nb@direction)
                    )

                    if (
                        alignment < cfg["normal_alignment_min"]
                        or float(na@nb) > cfg["normal_dot_max"]
                    ):
                        continue

                    candidates.append((
                        -alignment,
                        abs(width*1000-25),
                        -float(z),
                        int(angle),
                        (A+B)/2,
                        width
                    ))

        candidates.sort(key=lambda x:x[:3])
        row["geometric_candidates"] = len(candidates)

        if not candidates:
            row["status"] = "NO_GEOMETRIC_CANDIDATE"
            row["failed_checks"] = "no_geometric_candidate"
            return row

        # --------------------------------------------------
        # 2. STATIC PANDA FEASIBILITY
        # --------------------------------------------------
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        robot = p.loadURDF(
            "franka_panda/panda.urdf",
            [-.5,0,0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )

        def set_arm(q):
            for j,v in enumerate(q):
                p.resetJointState(robot,j,float(v))
            for j in (9,10):
                p.resetJointState(robot,j,OPEN)
            p.performCollisionDetection()

        def ik(target,orn):
            return np.asarray(
                p.calculateInverseKinematics(
                    robot,11,
                    np.asarray(target).tolist(),orn,
                    lowerLimits=LO.tolist(),
                    upperLimits=HI.tolist(),
                    jointRanges=(HI-LO).tolist(),
                    restPoses=HOME.tolist(),
                    maxNumIterations=250,
                    residualThreshold=1e-6
                )[:7]
            )

        def self_collision():
            return any(
                c[8] < -.0001
                and tuple(sorted((c[3],c[4]))) != (6,8)
                for c in p.getContactPoints(robot,robot)
            )

        chosen = None
        tested_geometric_candidates = 0

        for candidate in candidates[
            :int(cfg["max_collision_checked_candidates"])
        ]:
            tested_geometric_candidates += 1
            _,_,_,angle,target,width = candidate

            for pitch_deg in cfg["wrist_pitch_offsets_deg"]:

                orn = p.getQuaternionFromEuler([
                    np.pi,
                    np.radians(float(pitch_deg)),
                    np.radians(angle-90)
                ])

                set_arm(HOME)
                q = ik(target,orn)
                set_arm(q)

                state = p.getLinkState(
                    robot,11,computeForwardKinematics=True
                )

                position_error = 1000*np.linalg.norm(
                    np.asarray(state[4])-target
                )

                rotation_error = np.degrees(
                    (
                        Rotation.from_quat(state[5]).inv()
                        * Rotation.from_quat(orn)
                    ).magnitude()
                )

                oc = p.getClosestPoints(robot,obj,.15)
                tc = p.getClosestPoints(robot,table,.15)

                ogap = min(
                    (c[8] for c in oc),
                    default=float("inf")
                )
                tgap = min(
                    (c[8] for c in tc),
                    default=float("inf")
                )

                valid = (
                    position_error < 2
                    and rotation_error < 2
                    and np.all(q >= LO-1e-6)
                    and np.all(q <= HI+1e-6)
                    and ogap >= -.0001
                    and tgap >= -.0001
                    and not self_collision()
                )

                if valid:
                    chosen = (
                        target,orn,angle,width,pitch_deg
                    )
                    break

            if chosen is not None:
                break

        row["static_candidates_checked"] = (
            tested_geometric_candidates
        )

        if chosen is None:
            row["status"] = "NO_STATIC_FEASIBLE_CANDIDATE"
            row["failed_checks"] = "static_feasibility"
            return row

        grasp,orn,angle,width,pitch_deg = chosen
        row["chosen_angle_deg"] = angle
        row["chosen_pitch_deg"] = pitch_deg
        row["chosen_width_mm"] = round(width*1000,4)

        set_arm(HOME)
        pre = grasp + np.array([0,0,PREHEIGHT])

        qpre = ik(pre,orn)
        qgrasp = ik(grasp,orn)

        descent = [
            ik(pre+(grasp-pre)*(i/240),orn)
            for i in range(241)
        ]

        lifting = [
            ik(
                grasp+np.array([0,0,LIFTHEIGHT*i/240]),
                orn
            )
            for i in range(241)
        ]

        # --------------------------------------------------
        # 3. CONTINUOUS DYNAMIC APPROACH
        # --------------------------------------------------
        def command(q,close=False):
            p.setJointMotorControlArray(
                robot,list(range(7)),
                p.POSITION_CONTROL,
                targetPositions=np.asarray(q).tolist(),
                forces=[150.]*7,
                positionGains=[.15]*7
            )

            for j in (9,10):
                p.setJointMotorControl2(
                    robot,j,p.POSITION_CONTROL,
                    targetPosition=0. if close else OPEN,
                    force=40
                )

        def ee_pose():
            state = p.getLinkState(
                robot,11,computeForwardKinematics=True
            )
            return (
                np.asarray(state[4]),
                Rotation.from_quat(state[5])
            )

        def bilateral():
            contacts = p.getContactPoints(robot,obj)

            return (
                any(
                    c[3]==9 and c[9]>FORCE_MIN
                    for c in contacts
                )
                and any(
                    c[3]==10 and c[9]>FORCE_MIN
                    for c in contacts
                )
            )

        stats = {
            "proximity":0,
            "physical":0,
            "table":0,
            "self":0,
            "motion":0.,
            "object_gap":float("inf"),
            "table_gap":float("inf")
        }

        def audit():
            contacts = p.getContactPoints(robot,obj)
            table_contacts = p.getContactPoints(robot,table)

            oc = p.getClosestPoints(robot,obj,.10)
            tc = p.getClosestPoints(robot,table,.10)

            ogap = min(
                (c[8] for c in oc),
                default=float("inf")
            )
            tgap = min(
                (c[8] for c in tc),
                default=float("inf")
            )

            physical = any(
                c[8]*1000 <
                    crit["approach_object_penetration_mm_limit"]
                or c[9] >
                    crit["approach_object_normal_force_N_limit"]
                for c in contacts
            )

            stats["proximity"] += int(bool(contacts))
            stats["physical"] += int(physical)
            stats["table"] += int(bool(table_contacts))
            stats["self"] += int(self_collision())

            stats["motion"] = max(
                stats["motion"],
                np.linalg.norm(object_pose()[0]-initial)*1000
            )

            stats["object_gap"] = min(
                stats["object_gap"],ogap*1000
            )
            stats["table_gap"] = min(
                stats["table_gap"],tgap*1000
            )

        for i in range(480):
            a = (i+1)/480
            smooth = 3*a*a-2*a*a*a

            command(HOME+smooth*(qpre-HOME))
            p.stepSimulation()
            audit()

        for _ in range(240):
            command(qpre)
            p.stepSimulation()
            audit()

        for i in range(240):
            command(descent[i+1])
            p.stepSimulation()
            audit()

        for _ in range(240):
            command(qgrasp)
            p.stepSimulation()
            audit()

        approach_error = np.linalg.norm(
            ee_pose()[0]-grasp
        )*1000

        row["approach_proximity_frames"] = stats["proximity"]
        row["approach_physical_object_frames"] = stats["physical"]
        row["approach_table_frames"] = stats["table"]
        row["approach_self_collision_frames"] = stats["self"]
        row["approach_max_object_motion_mm"] = round(
            stats["motion"],4
        )
        row["approach_position_error_mm"] = round(
            approach_error,4
        )
        row["minimum_object_gap_mm"] = round(
            stats["object_gap"],4
        )
        row["minimum_table_gap_mm"] = round(
            stats["table_gap"],4
        )

        approach_checks = {
            "object_contact":stats["physical"]==0,
            "table_contact":
                stats["table"] <=
                crit["approach_table_contact_frames_max"],
            "self_collision":
                stats["self"] <=
                crit["approach_self_collision_frames_max"],
            "object_clearance":
                stats["object_gap"] >=
                crit["approach_object_penetration_mm_limit"],
            "object_motion":
                stats["motion"] <
                crit["approach_displacement_mm_max_exclusive"],
            "position_error":
                approach_error <
                crit["approach_position_error_mm_max_exclusive"]
        }

        failed = [
            key for key,passed in approach_checks.items()
            if not passed
        ]

        if failed:
            row["status"] = "APPROACH_FAILED"
            row["failed_checks"] = ";".join(failed)
            return row

        # --------------------------------------------------
        # 4. FORCE-BEARING FINGER CLOSURE
        # --------------------------------------------------
        closure_frames = 0

        for _ in range(240):
            command(qgrasp,True)
            p.stepSimulation()
            closure_frames += int(bilateral())

        row["closure_bilateral_force_frames"] = closure_frames

        # --------------------------------------------------
        # 5. CONTINUOUS 70 MM LIFT
        # --------------------------------------------------
        lift_frames = 0

        for i in range(240):
            command(lifting[i+1],True)
            p.stepSimulation()
            lift_frames += int(bilateral())

        row["lift_bilateral_force_frames"] = lift_frames

        lift_obj_xyz,lift_obj_rot = object_pose()
        lift_ee_xyz,lift_ee_rot = ee_pose()

        relative_translation = lift_obj_xyz-lift_ee_xyz
        relative_rotation = lift_ee_rot.inv()*lift_obj_rot

        hold_q = np.array([
            p.getJointState(robot,j)[0]
            for j in range(7)
        ])

        # --------------------------------------------------
        # 6. TWO-SECOND RETENTION WITH ROTATION AUDIT
        # --------------------------------------------------
        hold_frames = 0
        table_frames = 0
        min_rise = float("inf")
        max_translation_slip = 0.
        max_rotation_slip = 0.

        for _ in range(HOLD_STEPS):
            command(hold_q,True)
            p.stepSimulation()

            obj_xyz,obj_rot = object_pose()
            ee_xyz,ee_rot = ee_pose()

            rise = (obj_xyz[2]-initial[2])*1000

            trans_slip = np.linalg.norm(
                (obj_xyz-ee_xyz)-relative_translation
            )*1000

            current_relative_rotation = ee_rot.inv()*obj_rot

            rot_slip = np.degrees(
                (
                    relative_rotation.inv()
                    *current_relative_rotation
                ).magnitude()
            )

            hold_frames += int(bilateral())
            table_frames += int(
                bool(p.getContactPoints(obj,table))
            )

            min_rise = min(min_rise,rise)
            max_translation_slip = max(
                max_translation_slip,trans_slip
            )
            max_rotation_slip = max(
                max_rotation_slip,rot_slip
            )

        row["hold_bilateral_force_frames"] = hold_frames
        row["hold_table_frames"] = table_frames
        row["minimum_hold_rise_mm"] = round(min_rise,4)
        row["final_rise_mm"] = round(
            (object_pose()[0][2]-initial[2])*1000,4
        )
        row["maximum_translation_slip_mm"] = round(
            max_translation_slip,4
        )
        row["maximum_rotation_slip_deg"] = round(
            max_rotation_slip,4
        )

        checks = {
            "closure":
                closure_frames >=
                crit["closure_bilateral_frames_min"],
            "lift_contacts":
                lift_frames >=
                crit["lift_bilateral_frames_min"],
            "hold_contacts":
                hold_frames >=
                crit["hold_bilateral_frames_min"],
            "hold_table":
                table_frames <=
                crit["hold_table_contact_frames_max"],
            "lift_height":
                min_rise >
                crit["minimum_retention_rise_mm_exclusive"],
            "translation_slip":
                max_translation_slip <
                crit["relative_translation_slip_mm_max_exclusive"],
            "rotation_slip":
                max_rotation_slip <
                crit["relative_rotation_slip_deg_max_exclusive"]
        }

        failures = [
            key for key,passed in checks.items()
            if not passed
        ]

        row["success"] = len(failures)==0
        row["status"] = (
            "SUCCESS" if row["success"] else "GRASP_FAILED"
        )
        row["failed_checks"] = ";".join(failures)

        return row

    except Exception as exc:
        row["status"] = "ERROR"
        row["success"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["failed_checks"] = "execution_error"
        return row

    finally:
        if env is not None:
            env.close()


print("\nSTEP 43CU — INDEPENDENT MUG GRASP BENCHMARK",flush=True)
print("="*90,flush=True)
print("Held-out placements:",len(indices),flush=True)
print("Development index excluded: 19",flush=True)
print("Force-bearing bilateral contact required: YES",flush=True)
print("Rotational retention threshold: 15 degrees",flush=True)
print("Artificial fixed attachment: NONE",flush=True)

results = []

for number,index in enumerate(indices,1):
    result = evaluate(index)
    results.append(result)

    # Save progress after every completed placement.
    with output.open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(
        f"[{number:02d}/{len(indices)}] "
        f"Index {index}: {result['status']} | "
        f"candidates={result['geometric_candidates']} | "
        f"checked={result['static_candidates_checked']} | "
        f"rise={result['final_rise_mm']} | "
        f"translation_slip={result['maximum_translation_slip_mm']} | "
        f"rotation_slip={result['maximum_rotation_slip_deg']} | "
        f"failed={result['failed_checks']}",
        flush=True
    )

    if result["error"]:
        print("  ERROR:",result["error"],flush=True)

df = pd.DataFrame(results)

print("\nMETHOD-LEVEL RESULTS")
print("-"*90)

for method,selected in methods.items():
    subset = df[
        df["archive_index"].isin(selected)
    ]

    n = len(subset)
    successes = int(subset["success"].sum())

    print(
        f"{method:29s} "
        f"{successes}/{n} "
        f"({100*successes/n:.1f}%)"
    )

print("\nSTATUS DISTRIBUTION")
print(df["status"].value_counts().to_string())

print("\nFAILED PLACEMENTS")
for _,r in df[df["status"]!="SUCCESS"].iterrows():
    print(
        f"Index {r['archive_index']}: "
        f"{r['status']} | "
        f"{r['failed_checks']}"
    )

print("\nRESULTS FILE:",output.resolve())
print("Frozen protocol unchanged.")
print("Original experiment CSVs unchanged.")
print("All results are simulated and planner-dependent.")
