import os
import math
import shutil
import subprocess

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = "results/repeated_mug_grasp_v2"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
VIDEO_PATH = os.path.join(OUTPUT_DIR, "repeated_mug_grasp_v2.mp4")

os.makedirs(FRAMES_DIR, exist_ok=True)


# ============================================================
# Video settings
# ============================================================

WIDTH = 1280
HEIGHT = 720
FPS = 30

# ~8 seconds total
N_HOME = 15
N_APPROACH = 45
N_ALIGN = 20
N_CLOSE = 20
N_GRASP_PAUSE = 15
N_LIFT = 35
N_TRANSPORT = 30
N_LOWER = 30
N_RELEASE = 20
N_RETREAT = 40

SIM_STEPS_PER_FRAME = 4


# ============================================================
# Panda settings
# ============================================================

EE_LINK = 11
ARM_JOINTS = list(range(7))

LEFT_FINGER = 9
RIGHT_FINGER = 10

OPEN_GRIPPER = 0.040
CLOSED_GRIPPER = 0.006


# ============================================================
# Scene positions
# ============================================================

ROBOT_BASE = [0.0, 0.0, 0.0]

# Mug begins near robot and clearly visible
MUG_START = np.array([0.60, 0.02, 0.115])

# Robot trajectory
HOME = np.array([0.42, -0.32, 0.47])

PREGRASP = np.array([0.60, -0.15, 0.24])

ALIGN = np.array([0.60, -0.075, 0.175])

GRASP = np.array([0.60, -0.045, 0.155])

LIFT = np.array([0.60, -0.045, 0.40])

TRANSPORT = np.array([0.47, 0.10, 0.40])

PLACE = np.array([0.47, 0.10, 0.155])

RETREAT = np.array([0.42, -0.20, 0.35])


# End-effector orientation:
# fingers roughly vertical around mug
EE_ORN = p.getQuaternionFromEuler(
    [math.pi, 0.0, math.pi / 2.0]
)


# ============================================================
# Helpers
# ============================================================

def lerp(a, b, t):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return a + (b - a) * t


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def move_robot(robot_id, target_pos, target_orn, gripper_width):
    ik = p.calculateInverseKinematics(
        robot_id,
        EE_LINK,
        target_pos,
        target_orn,
        maxNumIterations=100,
        residualThreshold=1e-5,
    )

    for joint_id, target in zip(ARM_JOINTS, ik[:7]):
        p.setJointMotorControl2(
            robot_id,
            joint_id,
            p.POSITION_CONTROL,
            targetPosition=target,
            force=250,
            positionGain=0.12,
            velocityGain=1.0,
        )

    p.setJointMotorControl2(
        robot_id,
        LEFT_FINGER,
        p.POSITION_CONTROL,
        targetPosition=gripper_width,
        force=80,
    )

    p.setJointMotorControl2(
        robot_id,
        RIGHT_FINGER,
        p.POSITION_CONTROL,
        targetPosition=gripper_width,
        force=80,
    )


def simulate():
    for _ in range(SIM_STEPS_PER_FRAME):
        p.stepSimulation()


def render_frame(frame_number):
    # Close-up camera focused on gripper + mug
    camera_target = [0.53, 0.00, 0.25]

    view = p.computeViewMatrix(
        cameraEyePosition=[1.10, -1.05, 0.72],
        cameraTargetPosition=camera_target,
        cameraUpVector=[0, 0, 1],
    )

    projection = p.computeProjectionMatrixFOV(
        fov=45,
        aspect=WIDTH / HEIGHT,
        nearVal=0.02,
        farVal=4.0,
    )

    _, _, rgba, _, _ = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=[1, -1, 2],
    )

    rgb = np.array(rgba, dtype=np.uint8).reshape(
        HEIGHT, WIDTH, 4
    )[:, :, :3]

    frame_path = os.path.join(
        FRAMES_DIR,
        f"frame_{frame_number:06d}.png"
    )

    Image.fromarray(rgb).save(frame_path)


def animate_segment(
    robot_id,
    frame_counter,
    start_pos,
    end_pos,
    n_frames,
    gripper_start,
    gripper_end=None,
):
    if gripper_end is None:
        gripper_end = gripper_start

    for i in range(n_frames):
        if n_frames == 1:
            t = 1.0
        else:
            t = i / (n_frames - 1)

        t = smoothstep(t)

        pos = lerp(start_pos, end_pos, t)

        grip = (
            gripper_start
            + (gripper_end - gripper_start) * t
        )

        move_robot(
            robot_id,
            pos.tolist(),
            EE_ORN,
            grip,
        )

        simulate()

        frame_counter += 1
        render_frame(frame_counter)

    return frame_counter


# ============================================================
# Main
# ============================================================

def main():

    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)

    os.makedirs(FRAMES_DIR, exist_ok=True)

    p.connect(p.DIRECT)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(0, 0, -9.81)

    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI,
        0
    )

    # --------------------------------------------------------
    # Floor
    # --------------------------------------------------------

    plane = p.loadURDF(
        "plane.urdf",
        [0, 0, -0.005]
    )

    # Dark neutral floor
    p.changeVisualShape(
        plane,
        -1,
        rgbaColor=[0.15, 0.15, 0.17, 1.0]
    )

    # --------------------------------------------------------
    # Black tabletop
    # --------------------------------------------------------

    table_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.45, 0.38, 0.025]
    )

    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.45, 0.38, 0.025],
        rgbaColor=[0.05, 0.05, 0.06, 1.0]
    )

    table_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_collision,
        baseVisualShapeIndex=table_visual,
        basePosition=[0.53, 0.02, 0.025]
    )

    # --------------------------------------------------------
    # White target pad
    # --------------------------------------------------------

    pad_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.12, 0.12, 0.006]
    )

    pad_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.12, 0.12, 0.006],
        rgbaColor=[0.95, 0.95, 0.95, 1.0]
    )

    pad_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=pad_collision,
        baseVisualShapeIndex=pad_visual,
        basePosition=[
            MUG_START[0],
            MUG_START[1],
            0.062
        ]
    )

    # Second placement pad
    target_pad = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=pad_collision,
        baseVisualShapeIndex=pad_visual,
        basePosition=[
            TRANSPORT[0],
            TRANSPORT[1],
            0.062
        ]
    )

    # --------------------------------------------------------
    # Robot
    # --------------------------------------------------------

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        ROBOT_BASE,
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION
    )

    # --------------------------------------------------------
    # Mug
    # --------------------------------------------------------

    mug_urdf = (
        "environments/3d_models/objects/"
        "ycb_mug/ycb_mug.urdf"
    )

    mug_id = p.loadURDF(
        mug_urdf,
        MUG_START.tolist(),
        p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False
    )

    p.changeVisualShape(
        mug_id,
        -1,
        rgbaColor=[0.65, 0.03, 0.03, 1.0]
    )

    # Improve contact behaviour
    p.changeDynamics(
        mug_id,
        -1,
        lateralFriction=1.0,
        spinningFriction=0.02,
        rollingFriction=0.02
    )

    # --------------------------------------------------------
    # Initialize robot at home
    # --------------------------------------------------------

    move_robot(
        robot_id,
        HOME.tolist(),
        EE_ORN,
        OPEN_GRIPPER
    )

    for _ in range(120):
        p.stepSimulation()

    frame = 0

    # --------------------------------------------------------
    # 0. Home
    # --------------------------------------------------------

    print("1/8 Home")

    for _ in range(N_HOME):
        move_robot(
            robot_id,
            HOME.tolist(),
            EE_ORN,
            OPEN_GRIPPER
        )

        simulate()
        frame += 1
        render_frame(frame)

    # --------------------------------------------------------
    # 1. Approach
    # --------------------------------------------------------

    print("2/8 Approach")

    frame = animate_segment(
        robot_id,
        frame,
        HOME,
        PREGRASP,
        N_APPROACH,
        OPEN_GRIPPER,
    )

    # --------------------------------------------------------
    # 2. Align
    # --------------------------------------------------------

    print("3/8 Align")

    frame = animate_segment(
        robot_id,
        frame,
        PREGRASP,
        ALIGN,
        N_ALIGN,
        OPEN_GRIPPER,
    )

    # --------------------------------------------------------
    # 3. Final grasp approach
    # --------------------------------------------------------

    frame = animate_segment(
        robot_id,
        frame,
        ALIGN,
        GRASP,
        10,
        OPEN_GRIPPER,
    )

    # --------------------------------------------------------
    # 4. Close fingers
    # --------------------------------------------------------

    print("4/8 Grasp")

    for i in range(N_CLOSE):

        t = i / max(1, N_CLOSE - 1)

        grip = (
            OPEN_GRIPPER
            + (CLOSED_GRIPPER - OPEN_GRIPPER) * t
        )

        move_robot(
            robot_id,
            GRASP.tolist(),
            EE_ORN,
            grip
        )

        simulate()

        frame += 1
        render_frame(frame)

    # --------------------------------------------------------
    # Attach mug after fingers visibly close
    # --------------------------------------------------------

    ee_pos, ee_orn = p.getLinkState(
        robot_id,
        EE_LINK
    )[:2]

    mug_pos, mug_orn = p.getBasePositionAndOrientation(
        mug_id
    )

    inv_pos, inv_orn = p.invertTransform(
        ee_pos,
        ee_orn
    )

    rel_pos, rel_orn = p.multiplyTransforms(
        inv_pos,
        inv_orn,
        mug_pos,
        mug_orn
    )

    grasp_constraint = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=EE_LINK,
        childBodyUniqueId=mug_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=rel_pos,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=rel_orn,
        childFrameOrientation=[0, 0, 0, 1]
    )

    # --------------------------------------------------------
    # Grasp pause
    # --------------------------------------------------------

    print("5/8 Grasp hold")

    for _ in range(N_GRASP_PAUSE):

        move_robot(
            robot_id,
            GRASP.tolist(),
            EE_ORN,
            CLOSED_GRIPPER
        )

        simulate()

        frame += 1
        render_frame(frame)

    # --------------------------------------------------------
    # 5. Lift
    # --------------------------------------------------------

    print("6/8 Lift")

    frame = animate_segment(
        robot_id,
        frame,
        GRASP,
        LIFT,
        N_LIFT,
        CLOSED_GRIPPER,
    )

    # --------------------------------------------------------
    # 6. Transport
    # --------------------------------------------------------

    print("7/8 Transport")

    frame = animate_segment(
        robot_id,
        frame,
        LIFT,
        TRANSPORT,
        N_TRANSPORT,
        CLOSED_GRIPPER,
    )

    # --------------------------------------------------------
    # 7. Lower
    # --------------------------------------------------------

    print("8/8 Place")

    frame = animate_segment(
        robot_id,
        frame,
        TRANSPORT,
        PLACE,
        N_LOWER,
        CLOSED_GRIPPER,
    )

    # --------------------------------------------------------
    # Release
    # --------------------------------------------------------

    for i in range(N_RELEASE):

        t = i / max(1, N_RELEASE - 1)

        grip = (
            CLOSED_GRIPPER
            + (OPEN_GRIPPER - CLOSED_GRIPPER) * t
        )

        move_robot(
            robot_id,
            PLACE.tolist(),
            EE_ORN,
            grip
        )

        simulate()

        frame += 1
        render_frame(frame)

    p.removeConstraint(grasp_constraint)

    # Let mug settle briefly
    for _ in range(10):
        p.stepSimulation()

    # --------------------------------------------------------
    # Retreat
    # --------------------------------------------------------

    frame = animate_segment(
        robot_id,
        frame,
        PLACE,
        RETREAT,
        18,
        OPEN_GRIPPER,
    )

    frame = animate_segment(
        robot_id,
        frame,
        RETREAT,
        HOME,
        N_RETREAT,
        OPEN_GRIPPER,
    )

    print()
    print(f"Rendered {frame} frames.")

    # --------------------------------------------------------
    # Encode video
    # --------------------------------------------------------

    if shutil.which("ffmpeg"):

        print("Encoding MP4...")

        command = [
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
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            VIDEO_PATH,
        ]

        subprocess.run(
            command,
            check=True
        )

        print()
        print("Video created:")
        print(VIDEO_PATH)

    else:
        print("ffmpeg not found.")

    p.disconnect()


if __name__ == "__main__":
    main()
