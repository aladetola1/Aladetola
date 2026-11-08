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

OUTPUT_DIR = "results/repeated_mug_grasp_v4"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
VIDEO_PATH = os.path.join(
    OUTPUT_DIR,
    "repeated_mug_grasp_v4.mp4"
)

WIDTH = 1280
HEIGHT = 720
FPS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TIMING
# ============================================================

N_HOME = 10
N_APPROACH = 32
N_DESCEND = 24
N_CLOSE = 18
N_GRASP_HOLD = 10
N_LIFT = 24
N_TRANSPORT = 28
N_LOWER = 24
N_RELEASE = 18
N_RETRACT = 22
N_RETURN = 30

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
# CLEAN PICK-AND-PLACE TRAJECTORY
# ============================================================

HOME = np.array([
    0.38,
    -0.30,
    0.55
])

ABOVE_PICK = np.array([
    0.60,
    0.02,
    0.40
])

GRASP = np.array([
    0.60,
    0.02,
    0.165
])

LIFT = np.array([
    0.60,
    0.02,
    0.42
])

ABOVE_PLACE = np.array([
    0.43,
    0.15,
    0.42
])

PLACE = np.array([
    0.43,
    0.15,
    0.165
])

RETRACT = np.array([
    0.43,
    0.15,
    0.42
])


# ============================================================
# FIXED END-EFFECTOR ORIENTATION
# ============================================================

EE_ORN = p.getQuaternionFromEuler([
    math.pi,
    0.0,
    0.0
])


# ============================================================
# PANDA IK LIMITS / REST POSE
# ============================================================

LOWER_LIMITS = [
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973
]

UPPER_LIMITS = [
    2.8973,
    1.7628,
    2.8973,
    -0.0698,
    2.8973,
    3.7525,
    2.8973
]

JOINT_RANGES = [
    u - l
    for l, u in zip(
        LOWER_LIMITS,
        UPPER_LIMITS
    )
]

# Natural elbow-up posture
REST_POSES = [
    0.0,
    -0.55,
    0.0,
    -2.15,
    0.0,
    1.65,
    0.78
]

JOINT_DAMPING = [
    0.08,
    0.08,
    0.08,
    0.08,
    0.08,
    0.08,
    0.08
]


# ============================================================
# HELPERS
# ============================================================

def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return a + (b - a) * t


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

        lowerLimits=LOWER_LIMITS,
        upperLimits=UPPER_LIMITS,
        jointRanges=JOINT_RANGES,
        restPoses=REST_POSES,
        jointDamping=JOINT_DAMPING,

        maxNumIterations=200,
        residualThreshold=1e-5
    )

    for joint_id, target in zip(
        ARM_JOINTS,
        ik[:7]
    ):

        p.setJointMotorControl2(
            robot,
            joint_id,
            p.POSITION_CONTROL,
            targetPosition=target,
            force=300,
            positionGain=0.18,
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

    camera_eye = [
        1.03,
        -0.78,
        0.35
    ]

    camera_target = [
        0.52,
        0.05,
        0.23
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
        lightDirection=[1, -1, 2]
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


# ============================================================
# TRAJECTORY SEGMENT
# ============================================================

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
        [0, 0, -0.005]
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
    # TABLE
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

    pad_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            0.095,
            0.095,
            0.006
        ]
    )

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
    # ROBOT
    # ========================================================

    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        ROBOT_BASE,
        useFixedBase=True
    )


    # Put robot directly into natural initial posture
    for joint_id, q in enumerate(REST_POSES):

        p.resetJointState(
            robot,
            joint_id,
            q
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
    # INITIALIZE AT HOME
    # ========================================================

    command_robot(
        robot,
        HOME.tolist(),
        OPEN_GRIPPER
    )

    for _ in range(180):
        p.stepSimulation()

    frame = 0


    # ========================================================
    # HOME
    # ========================================================

    print("1/9 Home")

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
    # APPROACH ABOVE PICK
    # ========================================================

    print("2/9 Approach above mug")

    frame = segment(
        robot,
        frame,
        HOME,
        ABOVE_PICK,
        N_APPROACH,
        OPEN_GRIPPER
    )


    # ========================================================
    # VERTICAL DESCENT
    # ========================================================

    print("3/9 Vertical descend")

    frame = segment(
        robot,
        frame,
        ABOVE_PICK,
        GRASP,
        N_DESCEND,
        OPEN_GRIPPER
    )


    # ========================================================
    # CLOSE GRIPPER
    # ========================================================

    print("4/9 Grasp")

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
    # ATTACH MUG
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

        jointAxis=[0, 0, 0],

        parentFramePosition=rel_pos,
        parentFrameOrientation=rel_orn,

        childFramePosition=[0, 0, 0],

        childFrameOrientation=[
            0,
            0,
            0,
            1
        ]
    )


    # ========================================================
    # HOLD
    # ========================================================

    print("5/9 Hold")

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
    # VERTICAL LIFT
    # ========================================================

    print("6/9 Vertical lift")

    frame = segment(
        robot,
        frame,
        GRASP,
        LIFT,
        N_LIFT,
        CLOSED_GRIPPER
    )


    # ========================================================
    # HORIZONTAL TRANSPORT
    # ========================================================

    print("7/9 Horizontal transport")

    frame = segment(
        robot,
        frame,
        LIFT,
        ABOVE_PLACE,
        N_TRANSPORT,
        CLOSED_GRIPPER
    )


    # ========================================================
    # VERTICAL LOWER
    # ========================================================

    print("8/9 Vertical lower")

    frame = segment(
        robot,
        frame,
        ABOVE_PLACE,
        PLACE,
        N_LOWER,
        CLOSED_GRIPPER
    )


    # ========================================================
    # RELEASE
    # ========================================================

    print("9/9 Release")

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


    p.removeConstraint(
        grasp_constraint
    )

    for _ in range(15):
        p.stepSimulation()


    # ========================================================
    # VERTICAL RETRACT
    # ========================================================

    frame = segment(
        robot,
        frame,
        PLACE,
        RETRACT,
        N_RETRACT,
        OPEN_GRIPPER
    )


    # ========================================================
    # RETURN HOME
    # ========================================================

    frame = segment(
        robot,
        frame,
        RETRACT,
        HOME,
        N_RETURN,
        OPEN_GRIPPER
    )


    # ========================================================
    # ENCODE
    # ========================================================

    duration = frame / FPS

    print()
    print(
        f"Rendered {frame} frames "
        f"({duration:.2f} seconds)."
    )

    print("Encoding V4 video...")

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
    print("V4 VIDEO READY:")
    print(VIDEO_PATH)

    p.disconnect()


if __name__ == "__main__":
    main()
