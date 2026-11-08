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

OUTPUT_DIR = "results/repeated_mug_grasp_v5"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
VIDEO_PATH = os.path.join(
    OUTPUT_DIR,
    "repeated_mug_grasp_v5.mp4"
)

WIDTH = 1280
HEIGHT = 720
FPS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TIMING
# ============================================================

N_HOME = 8
N_APPROACH = 28
N_DESCEND = 22
N_CLOSE = 24
N_CONTACT_HOLD = 12
N_LIFT = 24
N_TRANSPORT = 26
N_LOWER = 24
N_RELEASE = 18
N_RETRACT = 20
N_RETURN = 26

SIM_STEPS = 4


# ============================================================
# PANDA
# ============================================================

EE_LINK = 11

ARM_JOINTS = list(range(7))

LEFT_FINGER = 9
RIGHT_FINGER = 10

OPEN_GRIPPER = 0.040
CLOSED_GRIPPER = 0.002


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
    0.0,
    0.0,
    0.0
])


# ============================================================
# CLEAN TRAJECTORY
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

# IMPORTANT:
# This is slightly lower than V4 so that
# the fingers actually surround the mug.
GRASP = np.array([
    0.60,
    0.02,
    0.145
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
    0.145
])

RETRACT = np.array([
    0.43,
    0.15,
    0.42
])


# ============================================================
# FIXED GRIPPER ORIENTATION
# ============================================================

EE_ORN = p.getQuaternionFromEuler([
    math.pi,
    0.0,
    0.0
])


# ============================================================
# IK CONFIGURATION
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
    upper - lower
    for lower, upper in zip(
        LOWER_LIMITS,
        UPPER_LIMITS
    )
]

REST_POSES = [
    0.0,
    -0.55,
    0.0,
    -2.15,
    0.0,
    1.65,
    0.78
]

JOINT_DAMPING = [0.08] * 7


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

        maxNumIterations=250,
        residualThreshold=1e-6
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
        force=120
    )

    p.setJointMotorControl2(
        robot,
        RIGHT_FINGER,
        p.POSITION_CONTROL,
        targetPosition=gripper_width,
        force=120
    )


def step_simulation():

    for _ in range(SIM_STEPS):
        p.stepSimulation()


# ============================================================
# CONTACT CHECK
# ============================================================

def finger_contacts(robot, mug):

    left = p.getContactPoints(
        bodyA=robot,
        bodyB=mug,
        linkIndexA=LEFT_FINGER
    )

    right = p.getContactPoints(
        bodyA=robot,
        bodyB=mug,
        linkIndexA=RIGHT_FINGER
    )

    return len(left) > 0, len(right) > 0


def finger_distances(robot, mug):

    left_points = p.getClosestPoints(
        bodyA=robot,
        bodyB=mug,
        distance=0.05,
        linkIndexA=LEFT_FINGER
    )

    right_points = p.getClosestPoints(
        bodyA=robot,
        bodyB=mug,
        distance=0.05,
        linkIndexA=RIGHT_FINGER
    )

    left_distance = None
    right_distance = None

    if left_points:
        left_distance = min(
            point[8]
            for point in left_points
        )

    if right_points:
        right_distance = min(
            point[8]
            for point in right_points
        )

    return left_distance, right_distance


# ============================================================
# CAMERA
# ============================================================

def render_frame(frame_number):

    # Side/front view so both finger tips
    # and the red mug remain visible.
    camera_eye = [
        1.02,
        -0.82,
        0.34
    ]

    camera_target = [
        0.52,
        0.055,
        0.22
    ]

    view = p.computeViewMatrix(
        cameraEyePosition=camera_eye,
        cameraTargetPosition=camera_target,
        cameraUpVector=[0, 0, 1]
    )

    projection = p.computeProjectionMatrixFOV(
        fov=35,
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
# MOTION SEGMENT
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
            0.92,
            0.92,
            0.92,
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

    for joint_id, q in enumerate(
        REST_POSES
    ):

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
            0.75,
            0.02,
            0.02,
            1
        ]
    )

    p.changeDynamics(
        mug,
        -1,
        lateralFriction=1.5,
        spinningFriction=0.05,
        rollingFriction=0.02
    )


    # ========================================================
    # INITIALIZE
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
    # 1. HOME
    # ========================================================

    print("1/10 Home")

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
    # 2. APPROACH ABOVE
    # ========================================================

    print("2/10 Approach above mug")

    frame = segment(
        robot,
        frame,
        HOME,
        ABOVE_PICK,
        N_APPROACH,
        OPEN_GRIPPER
    )


    # ========================================================
    # 3. VERTICAL DESCENT
    # ========================================================

    print("3/10 Descend around mug")

    frame = segment(
        robot,
        frame,
        ABOVE_PICK,
        GRASP,
        N_DESCEND,
        OPEN_GRIPPER
    )


    # Check finger-to-mug distances before closure
    left_dist, right_dist = finger_distances(
        robot,
        mug
    )

    print()
    print(
        "Before closing:"
    )

    print(
        "  Left finger distance:",
        left_dist
    )

    print(
        "  Right finger distance:",
        right_dist
    )


    # ========================================================
    # 4. CLOSE UNTIL PHYSICAL CONTACT
    # ========================================================

    print("4/10 Close fingers")

    contact_detected = False

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

        # More simulation here so that
        # finger/object contact is physically resolved.
        for _ in range(8):
            p.stepSimulation()

        left_contact, right_contact = (
            finger_contacts(
                robot,
                mug
            )
        )

        left_dist, right_dist = (
            finger_distances(
                robot,
                mug
            )
        )

        if (
            left_contact
            and right_contact
        ):
            contact_detected = True

        frame += 1
        render_frame(frame)


    # ========================================================
    # CONTACT REPORT
    # ========================================================

    left_contact, right_contact = (
        finger_contacts(
            robot,
            mug
        )
    )

    left_dist, right_dist = (
        finger_distances(
            robot,
            mug
        )
    )

    print()
    print("==============================")
    print("GRASP CONTACT CHECK")
    print("==============================")

    print(
        "Left finger touching mug:",
        left_contact
    )

    print(
        "Right finger touching mug:",
        right_contact
    )

    print(
        "Left distance:",
        left_dist
    )

    print(
        "Right distance:",
        right_dist
    )

    print("==============================")


    # Treat very small closest-point distance
    # as contact as well, to account for numerical
    # tolerance in PyBullet.
    left_near = (
        left_contact
        or (
            left_dist is not None
            and left_dist <= 0.002
        )
    )

    right_near = (
        right_contact
        or (
            right_dist is not None
            and right_dist <= 0.002
        )
    )

    if not (
        left_near
        and right_near
    ):
        print()
        print(
            "WARNING: two-sided mug contact "
            "was not achieved."
        )

        print(
            "The mug will NOT be artificially "
            "lifted."
        )

        print(
            "Adjust GRASP position before "
            "using this result."
        )

        p.disconnect()

        return


    print()
    print(
        "SUCCESS: both fingers reached the mug."
    )


    # ========================================================
    # 5. SHORT CONTACT HOLD
    # ========================================================

    print("5/10 Hold physical grasp")

    for _ in range(N_CONTACT_HOLD):

        command_robot(
            robot,
            GRASP.tolist(),
            CLOSED_GRIPPER
        )

        step_simulation()

        frame += 1
        render_frame(frame)


    # ========================================================
    # CREATE LIFT CONSTRAINT ONLY AFTER CONTACT
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
    # 6. VERTICAL LIFT
    # ========================================================

    print("6/10 Lift vertically")

    frame = segment(
        robot,
        frame,
        GRASP,
        LIFT,
        N_LIFT,
        CLOSED_GRIPPER
    )


    # ========================================================
    # 7. HIGH TRANSPORT
    # ========================================================

    print("7/10 Transport")

    frame = segment(
        robot,
        frame,
        LIFT,
        ABOVE_PLACE,
        N_TRANSPORT,
        CLOSED_GRIPPER
    )


    # ========================================================
    # 8. LOWER TO PAD
    # ========================================================

    print("8/10 Lower")

    frame = segment(
        robot,
        frame,
        ABOVE_PLACE,
        PLACE,
        N_LOWER,
        CLOSED_GRIPPER
    )


    # ========================================================
    # REMOVE LIFT CONSTRAINT AT TABLE
    # ========================================================

    p.removeConstraint(
        grasp_constraint
    )

    # Allow mug to establish support contact
    # before fingers open.
    for _ in range(25):
        p.stepSimulation()


    # ========================================================
    # 9. OPEN / RELEASE
    # ========================================================

    print("9/10 Release")

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


    # ========================================================
    # 10. RETRACT VERTICALLY
    # ========================================================

    print("10/10 Retract + home")

    frame = segment(
        robot,
        frame,
        PLACE,
        RETRACT,
        N_RETRACT,
        OPEN_GRIPPER
    )

    frame = segment(
        robot,
        frame,
        RETRACT,
        HOME,
        N_RETURN,
        OPEN_GRIPPER
    )


    # ========================================================
    # VIDEO
    # ========================================================

    duration = frame / FPS

    print()
    print(
        f"Rendered {frame} frames "
        f"({duration:.2f} seconds)."
    )

    print(
        "Encoding V5 video..."
    )

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
    print(
        "V5 VIDEO READY:"
    )

    print(
        VIDEO_PATH
    )

    p.disconnect()


if __name__ == "__main__":
    main()
