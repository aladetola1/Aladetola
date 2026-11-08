import os
import math
import shutil
import subprocess

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = "results/repeated_mug_grasp_v3"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
VIDEO_PATH = os.path.join(
    OUTPUT_DIR,
    "repeated_mug_grasp_v3.mp4"
)

WIDTH = 1280
HEIGHT = 720
FPS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TIMING
# ============================================================

# Approx. 7 seconds
N_HOME = 10
N_APPROACH = 35
N_DESCEND = 22
N_CLOSE = 20
N_GRASP_HOLD = 12
N_LIFT = 28
N_TRANSPORT = 28
N_LOWER = 25
N_RELEASE = 18
N_RETREAT = 30

SIM_STEPS = 4


# ============================================================
# ROBOT
# ============================================================

EE_LINK = 11
ARM_JOINTS = list(range(7))

LEFT_FINGER = 9
RIGHT_FINGER = 10

OPEN_GRIPPER = 0.040
CLOSED_GRIPPER = 0.008


# ============================================================
# SCENE
# ============================================================

ROBOT_BASE = [0.0, 0.0, 0.0]

MUG_START = np.array([
    0.60,
    0.02,
    0.115
])

MUG_ORIENTATION = p.getQuaternionFromEuler([
    0,
    0,
    0
])


# ============================================================
# TRAJECTORY
# ============================================================

HOME = np.array([
    0.40,
    -0.34,
    0.48
])

# Move toward mug while staying clearly away
PREGRASP = np.array([
    0.60,
    -0.16,
    0.32
])

# Position directly above / beside mug
ABOVE_MUG = np.array([
    0.60,
    -0.035,
    0.245
])

# Fingers descend around mug
GRASP = np.array([
    0.60,
    -0.025,
    0.165
])

# Straight visible lift
LIFT = np.array([
    0.60,
    -0.025,
    0.39
])

# Transport toward second location
TRANSPORT = np.array([
    0.43,
    0.13,
    0.39
])

# Placement location
PLACE = np.array([
    0.43,
    0.13,
    0.165
])

# Retreat after release
RETREAT = np.array([
    0.43,
    -0.08,
    0.33
])


# Tool points downward.
# Rotation keeps fingers easy to see from camera.
EE_ORN = p.getQuaternionFromEuler([
    math.pi,
    0.0,
    0.0
])


# ============================================================
# HELPERS
# ============================================================

def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return (
        np.asarray(a, dtype=float)
        + (
            np.asarray(b, dtype=float)
            - np.asarray(a, dtype=float)
        ) * t
    )


def command_robot(
    robot,
    position,
    gripper_width
):
    ik = p.calculateInverseKinematics(
        robot,
        EE_LINK,
        position,
        EE_ORN,
        maxNumIterations=150,
        residualThreshold=1e-5
    )

    for j, target in zip(
        ARM_JOINTS,
        ik[:7]
    ):
        p.setJointMotorControl2(
            robot,
            j,
            p.POSITION_CONTROL,
            targetPosition=target,
            force=300,
            positionGain=0.16,
            velocityGain=1.0
        )

    p.setJointMotorControl2(
        robot,
        LEFT_FINGER,
        p.POSITION_CONTROL,
        targetPosition=gripper_width,
        force=100
    )

    p.setJointMotorControl2(
        robot,
        RIGHT_FINGER,
        p.POSITION_CONTROL,
        targetPosition=gripper_width,
        force=100
    )


def step_simulation():
    for _ in range(SIM_STEPS):
        p.stepSimulation()


# ============================================================
# CAMERA
# ============================================================

def render_frame(frame_number):

    # LOW, CLOSE SIDE VIEW.
    # This is the major change from V2.
    camera_eye = [
        1.03,
        -0.78,
        0.34
    ]

    camera_target = [
        0.53,
        0.03,
        0.22
    ]

    view = p.computeViewMatrix(
        cameraEyePosition=camera_eye,
        cameraTargetPosition=camera_target,
        cameraUpVector=[0, 0, 1]
    )

    projection = p.computeProjectionMatrixFOV(
        fov=36,
        aspect=WIDTH / HEIGHT,
        nearVal=0.02,
        farVal=3.0
    )

    _, _, rgba, _, _ = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=[
            1,
            -1,
            2
        ]
    )

    rgba = np.asarray(
        rgba,
        dtype=np.uint8
    ).reshape(
        HEIGHT,
        WIDTH,
        4
    )

    rgb = rgba[:, :, :3]

    Image.fromarray(rgb).save(
        os.path.join(
            FRAMES_DIR,
            f"frame_{frame_number:06d}.png"
        )
    )


def segment(
    robot,
    frame,
    start,
    end,
    n_frames,
    gripper
):

    for i in range(n_frames):

        if n_frames <= 1:
            t = 1.0
        else:
            t = i / (n_frames - 1)

        t = smoothstep(t)

        position = lerp(
            start,
            end,
            t
        )

        command_robot(
            robot,
            position.tolist(),
            gripper
        )

        step_simulation()

        frame += 1
        render_frame(frame)

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)

    os.makedirs(FRAMES_DIR)

    p.connect(p.DIRECT)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        -9.81
    )


    # ========================================================
    # FLOOR
    # ========================================================

    plane = p.loadURDF(
        "plane.urdf",
        [
            0,
            0,
            -0.005
        ]
    )

    p.changeVisualShape(
        plane,
        -1,
        rgbaColor=[
            0.18,
            0.18,
            0.20,
            1
        ]
    )


    # ========================================================
    # TABLETOP
    # ========================================================

    table_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            0.48,
            0.40,
            0.025
        ]
    )

    table_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            0.48,
            0.40,
            0.025
        ],
        rgbaColor=[
            0.055,
            0.055,
            0.065,
            1
        ]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_col,
        baseVisualShapeIndex=table_vis,
        basePosition=[
            0.53,
            0.02,
            0.025
        ]
    )


    # ========================================================
    # TARGET PADS
    # ========================================================

    pad_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            0.095,
            0.095,
            0.006
        ],
        rgbaColor=[
            0.90,
            0.90,
            0.90,
            1
        ]
    )

    pad_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            0.095,
            0.095,
            0.006
        ]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=pad_col,
        baseVisualShapeIndex=pad_vis,
        basePosition=[
            MUG_START[0],
            MUG_START[1],
            0.062
        ]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=pad_col,
        baseVisualShapeIndex=pad_vis,
        basePosition=[
            PLACE[0],
            PLACE[1],
            0.062
        ]
    )


    # ========================================================
    # PANDA
    # ========================================================

    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        ROBOT_BASE,
        useFixedBase=True
    )


    # ========================================================
    # MUG
    # ========================================================

    mug_path = (
        "environments/3d_models/"
        "objects/ycb_mug/ycb_mug.urdf"
    )

    mug = p.loadURDF(
        mug_path,
        MUG_START.tolist(),
        MUG_ORIENTATION,
        useFixedBase=False
    )

    p.changeVisualShape(
        mug,
        -1,
        rgbaColor=[
            0.72,
            0.025,
            0.025,
            1
        ]
    )

    p.changeDynamics(
        mug,
        -1,
        lateralFriction=1.5,
        spinningFriction=0.05
    )


    # ========================================================
    # INITIALIZE
    # ========================================================

    command_robot(
        robot,
        HOME.tolist(),
        OPEN_GRIPPER
    )

    for _ in range(150):
        p.stepSimulation()

    frame = 0


    # ========================================================
    # 1 — HOME
    # ========================================================

    print("1/8 Home")

    for _ in range(N_HOME):

        command_robot(
            robot,
            HOME.tolist(),
            OPEN_GRIPPER
        )

        step_simulation()

        frame += 1
        render_frame(frame)


    # ========================================================
    # 2 — APPROACH
    # ========================================================

    print("2/8 Approach")

    frame = segment(
        robot,
        frame,
        HOME,
        PREGRASP,
        N_APPROACH,
        OPEN_GRIPPER
    )


    # ========================================================
    # 3 — MOVE ABOVE MUG
    # ========================================================

    print("3/8 Align")

    frame = segment(
        robot,
        frame,
        PREGRASP,
        ABOVE_MUG,
        18,
        OPEN_GRIPPER
    )


    # ========================================================
    # 4 — DESCEND AROUND MUG
    # ========================================================

    print("4/8 Descend")

    frame = segment(
        robot,
        frame,
        ABOVE_MUG,
        GRASP,
        N_DESCEND,
        OPEN_GRIPPER
    )


    # ========================================================
    # 5 — VISIBLE FINGER CLOSING
    # ========================================================

    print("5/8 Close gripper")

    for i in range(N_CLOSE):

        t = i / max(
            1,
            N_CLOSE - 1
        )

        t = smoothstep(t)

        width = (
            OPEN_GRIPPER
            + (
                CLOSED_GRIPPER
                - OPEN_GRIPPER
            ) * t
        )

        command_robot(
            robot,
            GRASP.tolist(),
            width
        )

        step_simulation()

        frame += 1
        render_frame(frame)


    # ========================================================
    # ATTACH MUG ONLY AFTER CLOSING IS VISIBLE
    # ========================================================

    ee_state = p.getLinkState(
        robot,
        EE_LINK
    )

    ee_pos = ee_state[0]
    ee_orn = ee_state[1]

    mug_pos, mug_orn = (
        p.getBasePositionAndOrientation(
            mug
        )
    )

    inv_pos, inv_orn = (
        p.invertTransform(
            ee_pos,
            ee_orn
        )
    )

    rel_pos, rel_orn = (
        p.multiplyTransforms(
            inv_pos,
            inv_orn,
            mug_pos,
            mug_orn
        )
    )

    grasp_constraint = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=EE_LINK,

        childBodyUniqueId=mug,
        childLinkIndex=-1,

        jointType=p.JOINT_FIXED,

        jointAxis=[
            0,
            0,
            0
        ],

        parentFramePosition=rel_pos,
        parentFrameOrientation=rel_orn,

        childFramePosition=[
            0,
            0,
            0
        ],

        childFrameOrientation=[
            0,
            0,
            0,
            1
        ]
    )


    # ========================================================
    # SHORT GRASP PAUSE
    # ========================================================

    print("6/8 Grasp hold")

    for _ in range(N_GRASP_HOLD):

        command_robot(
            robot,
            GRASP.tolist(),
            CLOSED_GRIPPER
        )

        step_simulation()

        frame += 1
        render_frame(frame)


    # ========================================================
    # 6 — LIFT
    # ========================================================

    print("7/8 Lift")

    frame = segment(
        robot,
        frame,
        GRASP,
        LIFT,
        N_LIFT,
        CLOSED_GRIPPER
    )


    # ========================================================
    # 7 — TRANSPORT
    # ========================================================

    print("8/8 Transport + place")

    frame = segment(
        robot,
        frame,
        LIFT,
        TRANSPORT,
        N_TRANSPORT,
        CLOSED_GRIPPER
    )


    # ========================================================
    # LOWER
    # ========================================================

    frame = segment(
        robot,
        frame,
        TRANSPORT,
        PLACE,
        N_LOWER,
        CLOSED_GRIPPER
    )


    # ========================================================
    # OPEN FINGERS CLEARLY
    # ========================================================

    for i in range(N_RELEASE):

        t = i / max(
            1,
            N_RELEASE - 1
        )

        t = smoothstep(t)

        width = (
            CLOSED_GRIPPER
            + (
                OPEN_GRIPPER
                - CLOSED_GRIPPER
            ) * t
        )

        command_robot(
            robot,
            PLACE.tolist(),
            width
        )

        step_simulation()

        frame += 1
        render_frame(frame)


    # Release object
    p.removeConstraint(
        grasp_constraint
    )


    # Allow mug to settle
    for _ in range(20):
        p.stepSimulation()


    # ========================================================
    # RETREAT
    # ========================================================

    frame = segment(
        robot,
        frame,
        PLACE,
        RETREAT,
        15,
        OPEN_GRIPPER
    )

    frame = segment(
        robot,
        frame,
        RETREAT,
        HOME,
        N_RETREAT,
        OPEN_GRIPPER
    )


    # ========================================================
    # ENCODE
    # ========================================================

    print()
    print(
        f"Rendered {frame} frames "
        f"({frame / FPS:.2f} seconds)."
    )

    print("Encoding video...")

    subprocess.run(
        [
            "ffmpeg",
            "-y",

            "-framerate",
            str(FPS),

            "-i",
            os.path.join(
                FRAMES_DIR,
                "frame_%06d.png"
            ),

            "-c:v",
            "libx264",

            "-preset",
            "medium",

            "-crf",
            "19",

            "-pix_fmt",
            "yuv420p",

            "-movflags",
            "+faststart",

            VIDEO_PATH
        ],
        check=True
    )

    print()
    print("VIDEO READY:")
    print(VIDEO_PATH)

    p.disconnect()


if __name__ == "__main__":
    main()
