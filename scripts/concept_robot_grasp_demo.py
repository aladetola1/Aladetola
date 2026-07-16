
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont


TABLE_RGBA = [0.66, 0.49, 0.32, 1.0]
MUG_RGBA = [0.44, 0.11, 0.11, 1.0]

WIDTH = 1280
HEIGHT = 720


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a conceptual robot grasp animation."
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Video frame rate.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/concept_robot_grasp_demo"),
        help="Output directory.",
    )

    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]

    for font_path in font_paths:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()


def apply_colour(body_id: int, rgba: list[float]) -> None:
    p.changeVisualShape(
        objectUniqueId=body_id,
        linkIndex=-1,
        rgbaColor=rgba,
    )

    for link_index in range(p.getNumJoints(body_id)):
        p.changeVisualShape(
            objectUniqueId=body_id,
            linkIndex=link_index,
            rgbaColor=rgba,
        )


def create_scene() -> dict[str, int]:
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf")

    table_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.45, 0.45, 0.025],
    )

    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.45, 0.45, 0.025],
        rgbaColor=TABLE_RGBA,
    )

    table_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_collision,
        baseVisualShapeIndex=table_visual,
        basePosition=[0.65, 0.0, 0.70],
    )

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
    )

    mug_path = (
        "environments/3d_models/objects/"
        "ycb_mug/ycb_mug.urdf"
    )

    mug_id = p.loadURDF(
        mug_path,
        basePosition=[0.65, 0.0, 0.80],
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        useFixedBase=False,
    )

    apply_colour(mug_id, MUG_RGBA)

    return {
        "table": table_id,
        "robot": robot_id,
        "mug": mug_id,
    }


def set_gripper(robot_id: int, opening: float) -> None:
    finger_joints = [9, 10]

    for joint_index in finger_joints:
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=opening,
            force=80,
        )


def move_arm(
    robot_id: int,
    target_position: list[float],
    target_orientation: list[float],
    steps: int,
) -> None:
    end_effector_link = 11

    joint_positions = p.calculateInverseKinematics(
        robot_id,
        end_effector_link,
        target_position,
        target_orientation,
        maxNumIterations=200,
        residualThreshold=1e-5,
    )

    arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]

    for joint_index, target in zip(
        arm_joint_indices,
        joint_positions[:7],
    ):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=200,
        )

    for _ in range(steps):
        p.stepSimulation()


def render_frame(
    title: str,
    subtitle: str,
) -> Image.Image:
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.75, -1.55, 1.35],
        cameraTargetPosition=[0.55, 0.0, 0.75],
        cameraUpVector=[0.0, 0.0, 1.0],
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=50,
        aspect=WIDTH / HEIGHT,
        nearVal=0.01,
        farVal=5.0,
    )

    image = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
    )

    rgba = np.asarray(
        image[2],
        dtype=np.uint8,
    ).reshape(HEIGHT, WIDTH, 4)

    frame = Image.fromarray(
        rgba[:, :, :3],
        mode="RGB",
    )

    draw = ImageDraw.Draw(frame, mode="RGBA")
    title_font = load_font(36)
    subtitle_font = load_font(24)

    draw.rounded_rectangle(
        (25, 22, 790, 125),
        radius=18,
        fill=(0, 0, 0, 175),
    )

    draw.text(
        (50, 42),
        title,
        font=title_font,
        fill=(255, 255, 255, 255),
    )

    draw.text(
        (50, 88),
        subtitle,
        font=subtitle_font,
        fill=(235, 235, 235, 255),
    )

    draw.rounded_rectangle(
        (285, 650, 995, 705),
        radius=14,
        fill=(0, 0, 0, 155),
    )

    draw.text(
        (320, 665),
        "Concept visualization only — hardware validation pending",
        font=subtitle_font,
        fill=(255, 255, 255, 255),
    )

    return frame


def save_phase_frames(
    frames_directory: Path,
    frame_number: int,
    count: int,
    title: str,
    subtitle: str,
) -> int:
    for _ in range(count):
        frame = render_frame(title, subtitle)
        frame.save(
            frames_directory / f"frame_{frame_number:06d}.png"
        )
        frame_number += 1

    return frame_number

def repeat_frame_sequence(
    frames_directory: Path,
    repetitions: int,
) -> None:
    """Repeat the rendered frame sequence."""

    if repetitions <= 1:
        return

    original_frames = sorted(
        frames_directory.glob("frame_*.png")
    )

    if not original_frames:
        raise RuntimeError("No frames were generated.")

    next_frame_number = len(original_frames) + 1

    for _ in range(repetitions - 1):
        for source_frame in original_frames:
            destination_frame = (
                frames_directory
                / f"frame_{next_frame_number:06d}.png"
            )

            shutil.copy2(
                source_frame,
                destination_frame,
            )

            next_frame_number += 1

def create_video(
    frames_directory: Path,
    output_video: Path,
    fps: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is not installed. Install it with: "
            "sudo apt install ffmpeg"
        )

    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_directory / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]

    subprocess.run(command, check=True)


def main() -> int:
    arguments = parse_arguments()

    if arguments.fps <= 0:
        print("ERROR: --fps must be greater than zero.", file=sys.stderr)
        return 1

    output_directory = arguments.output_dir
    frames_directory = output_directory / "frames"

    if frames_directory.exists():
        shutil.rmtree(frames_directory)

    frames_directory.mkdir(parents=True, exist_ok=True)

    client_id = p.connect(p.DIRECT)

    if client_id < 0:
        print("ERROR: PyBullet connection failed.", file=sys.stderr)
        return 1

    try:
        scene = create_scene()
        robot_id = scene["robot"]
        mug_id = scene["mug"]

        downward_orientation = p.getQuaternionFromEuler(
            [np.pi, 0.0, 0.0]
        )

        frame_number = 1
        fps = arguments.fps

        set_gripper(robot_id, opening=0.04)

        move_arm(
            robot_id,
            target_position=[0.65, 0.0, 1.15],
            target_orientation=downward_orientation,
            steps=180,
        )

        frame_number = save_phase_frames(
            frames_directory,
            frame_number,
            fps * 3,
            "Top-1 GFI Placement Selected",
            "Planned downstream integration with a robotic grasping system",
        )

        move_arm(
            robot_id,
            target_position=[0.65, 0.0, 0.96],
            target_orientation=downward_orientation,
            steps=180,
        )

        move_arm(
            robot_id,
            target_position=[0.65, 0.0, 0.935],
            target_orientation=downward_orientation,
            steps=80,
        )

        frame_number = save_phase_frames(
            frames_directory,
            frame_number,
            fps,
            "Final Approach",
            "Gripper aligns around the selected mug placement",
        )

        for opening in np.linspace(0.04, 0.005, 35):
            set_gripper(
                robot_id,
                opening=float(opening),
            )

            for _ in range(3):
                p.stepSimulation()

        frame_number = save_phase_frames(
            frames_directory,
            frame_number,
            fps * 3,
            "Grasp Established",
            "Gripper closes around the mug before lifting",
        )

        grasp_constraint = p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=11,
            childBodyUniqueId=mug_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, 0.10],
            childFramePosition=[0.0, 0.0, 0.0],
        )

        move_arm(
            robot_id,
            target_position=[0.65, 0.0, 1.05],
            target_orientation=downward_orientation,
            steps=120,
        )

        move_arm(
            robot_id,
            target_position=[0.65, 0.0, 1.25],
            target_orientation=downward_orientation,
            steps=120,
        )

        frame_number = save_phase_frames(
            frames_directory,
            frame_number,
            fps * 3,
            "Successful Lift",
            "Illustrative PlaceIt → RGMP → robot integration",
        )

        p.removeConstraint(grasp_constraint)

        repeat_frame_sequence(
            frames_directory=frames_directory,
            repetitions=3,
        )

        output_video = (
            output_directory / "concept_robot_grasp_demo.mp4"
        )

        create_video(
            frames_directory=frames_directory,
            output_video=output_video,
            fps=fps,
        )

        print(f"Video saved to: {output_video}")

    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    finally:
        p.disconnect()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
PY
