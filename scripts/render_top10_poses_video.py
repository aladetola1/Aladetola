
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont


OBJECTS = {
    "mug": {
        "run": "results/mug/run87",
        "urdf": "environments/3d_models/objects/ycb_mug/ycb_mug.urdf",
    },
    "bowl": {
        "run": "results/bowl/run89",
        "urdf": "environments/3d_models/objects/ycb_bowl/ycb_bowl.urdf",
    },
    "spoon": {
        "run": "results/spoon/run94",
        "urdf": "environments/3d_models/objects/ycb_spoon/ycb_spoon.urdf",
    },
    "power_drill": {
        "run": "results/power_drill/run97",
        "urdf": (
            "environments/3d_models/objects/"
            "ycb_power_drill/ycb_power_drill.urdf"
        ),
    },
    "plate": {
        "run": "results/plate/run100",
        "urdf": "environments/3d_models/objects/ycb_plate/ycb_plate.urdf",
    },
    "banana": {
        "run": "results/banana/run103",
        "urdf": "environments/3d_models/objects/ycb_banana/ycb_banana.urdf",
    },
}


TABLE_RGBA = [0.66, 0.49, 0.32, 1.0]
MUG_RGBA = [0.44, 0.11, 0.11, 1.0]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MP4 of the Top-10 GFI-ranked placements."
    )

    parser.add_argument(
        "--object",
        choices=sorted(OBJECTS.keys()),
        default="mug",
        help="Object to render. Default: mug.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ranked placements to display. Default: 10.",
    )

    parser.add_argument(
        "--seconds-per-pose",
        type=float,
        default=2.5,
        help="Display duration of each pose. Default: 2.5 seconds.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Output video frame rate. Default: 25.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/top10_pose_videos"),
        help="Directory for frames and video.",
    )

    return parser.parse_args()


def convert_candidate(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, dict):
        return candidate

    if hasattr(candidate, "item"):
        converted = candidate.item()

        if isinstance(converted, dict):
            return converted

    raise TypeError(
        "Each Top-10 entry must be a dictionary or a NumPy object "
        "containing a dictionary."
    )


def extract_final_pose(
    candidate: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:

    if "final_pose" not in candidate:
        raise KeyError(
            "Candidate does not contain the key 'final_pose'. "
            f"Available keys: {sorted(candidate.keys())}"
        )

    final_pose = candidate["final_pose"]

    if isinstance(final_pose, dict):
        if "xyz" not in final_pose or "euler_rpy" not in final_pose:
            raise KeyError(
                "final_pose must contain 'xyz' and 'euler_rpy'. "
                f"Available keys: {sorted(final_pose.keys())}"
            )

        xyz = np.asarray(final_pose["xyz"], dtype=float)
        rpy = np.asarray(final_pose["euler_rpy"], dtype=float)

    else:
        pose_array = np.asarray(final_pose, dtype=float).reshape(-1)

        if len(pose_array) < 6:
            raise ValueError(
                "final_pose must contain at least six values: "
                "x, y, z, roll, pitch, yaw."
            )

        xyz = pose_array[:3]
        rpy = pose_array[3:6]

    return xyz, rpy


def extract_gfi(candidate: dict[str, Any]) -> float:
    possible_keys = [
        "grasp_proxy",
        "gfi",
        "grasp_fitness",
        "grasp_score",
    ]

    for key in possible_keys:
        if key in candidate:
            return float(candidate[key])

    return float("nan")


def load_font(size: int) -> ImageFont.ImageFont:
    possible_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]

    for font_path in possible_fonts:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()


def apply_colour_to_body(
    body_id: int,
    rgba_colour: list[float],
) -> None:
    """Apply a visual colour to the base and all links of a body."""

    p.changeVisualShape(
        objectUniqueId=body_id,
        linkIndex=-1,
        rgbaColor=rgba_colour,
    )

    for link_index in range(p.getNumJoints(body_id)):
        p.changeVisualShape(
            objectUniqueId=body_id,
            linkIndex=link_index,
            rgbaColor=rgba_colour,
        )


def create_scene(
    object_name: str,
    object_urdf: str,
    xyz: np.ndarray,
    rpy: np.ndarray,
) -> None:
    """Create the light-brown table and load one final stable pose."""

    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf")

    table_half_extents = [0.35, 0.35, 0.02]
    table_position = [0.0, 0.0, 0.20]

    table_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=table_half_extents,
    )

    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=table_half_extents,
        rgbaColor=TABLE_RGBA,
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_collision,
        baseVisualShapeIndex=table_visual,
        basePosition=table_position,
    )

    quaternion = p.getQuaternionFromEuler(rpy.tolist())

    object_id = p.loadURDF(
        object_urdf,
        basePosition=xyz.tolist(),
        baseOrientation=quaternion,
        useFixedBase=True,
    )

    if object_name == "mug":
        apply_colour_to_body(
            body_id=object_id,
            rgba_colour=MUG_RGBA,
        )

        print(
            f"[COLOUR] Mug body {object_id} forced to "
            f"{MUG_RGBA}"
        )


def render_scene(
    width: int = 1400,
    height: int = 1000,
) -> Image.Image:

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.55, -0.75, 0.55],
        cameraTargetPosition=[0.0, 0.0, 0.25],
        cameraUpVector=[0.0, 0.0, 1.0],
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=45,
        aspect=width / height,
        nearVal=0.01,
        farVal=3.0,
    )

    camera_image = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
    )

    rgba = np.asarray(camera_image[2], dtype=np.uint8)
    rgba = rgba.reshape(height, width, 4)

    return Image.fromarray(rgba[:, :, :3], mode="RGB")


def add_information_overlay(
    image: Image.Image,
    object_name: str,
    rank: int,
    total: int,
    gfi: float,
    xyz: np.ndarray,
    rpy: np.ndarray,
) -> Image.Image:

    output = image.copy()
    draw = ImageDraw.Draw(output, mode="RGBA")

    title_font = load_font(42)
    text_font = load_font(27)

    panel = (25, 25, 760, 245)

    draw.rounded_rectangle(
        panel,
        radius=20,
        fill=(0, 0, 0, 175),
    )

    draw.text(
        (55, 48),
        f"Top GFI Placement {rank}/{total}",
        font=title_font,
        fill=(255, 255, 255, 255),
    )

    gfi_text = "not stored" if np.isnan(gfi) else f"{gfi:.4f}"

    lines = [
        f"Object: {object_name.replace('_', ' ').title()}",
        f"GFI: {gfi_text}",
        (
            "Position [m]: "
            f"x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}"
        ),
        (
            "Orientation [rad]: "
            f"roll={rpy[0]:.3f}, pitch={rpy[1]:.3f}, "
            f"yaw={rpy[2]:.3f}"
        ),
    ]

    for line_number, line in enumerate(lines):
        draw.text(
            (58, 105 + 31 * line_number),
            line,
            font=text_font,
            fill=(255, 255, 255, 255),
        )

    footer_font = load_font(30)

    footer_text = (
        "Top-ranked placement according to predicted grasp friendliness"
    )

    footer_box = (260, 910, 1140, 975)

    draw.rounded_rectangle(
        footer_box,
        radius=18,
        fill=(0, 0, 0, 165),
    )

    draw.text(
        (300, 925),
        footer_text,
        font=footer_font,
        fill=(255, 255, 255, 255),
    )

    return output


def generate_frames(
    object_name: str,
    top_k: int,
    seconds_per_pose: float,
    fps: int,
    output_directory: Path,
) -> tuple[Path, int]:

    configuration = OBJECTS[object_name]

    candidates_path = (
        Path(configuration["run"]) / "top10_grasp_candidates.npy"
    )

    object_urdf = Path(configuration["urdf"])

    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Top-10 candidate file not found: {candidates_path}"
        )

    if not object_urdf.exists():
        raise FileNotFoundError(
            f"Object URDF not found: {object_urdf}"
        )

    loaded_candidates = np.load(
        candidates_path,
        allow_pickle=True,
    )

    number_to_render = min(top_k, len(loaded_candidates))

    if number_to_render == 0:
        raise RuntimeError("The candidate file is empty.")

    selected_candidates = [
        convert_candidate(candidate)
        for candidate in loaded_candidates[:number_to_render]
    ]

    frames_directory = (
        output_directory
        / object_name
        / "frames"
    )

    if frames_directory.exists():
        shutil.rmtree(frames_directory)

    frames_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames_per_pose = max(
        1,
        int(round(seconds_per_pose * fps)),
    )

    frame_number = 1

    print(f"\nObject: {object_name}")
    print(f"Candidate file: {candidates_path}")
    print(f"Rendering {number_to_render} placements")
    print(f"Frames per pose: {frames_per_pose}")

    for rank, candidate in enumerate(
        selected_candidates,
        start=1,
    ):
        xyz, rpy = extract_final_pose(candidate)
        gfi = extract_gfi(candidate)

        create_scene(
            object_name=object_name,
            object_urdf=str(object_urdf),
            xyz=xyz,
            rpy=rpy,
        )

        rendered_image = render_scene()

        annotated_image = add_information_overlay(
            image=rendered_image,
            object_name=object_name,
            rank=rank,
            total=number_to_render,
            gfi=gfi,
            xyz=xyz,
            rpy=rpy,
        )

        still_path = (
            output_directory
            / object_name
            / f"rank_{rank:02d}_gfi_pose.png"
        )

        still_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        annotated_image.save(still_path)

        for _ in range(frames_per_pose):
            frame_path = (
                frames_directory
                / f"frame_{frame_number:06d}.png"
            )

            annotated_image.save(frame_path)
            frame_number += 1

        print(
            f"Rank {rank:02d}: "
            f"GFI={gfi:.6f}, "
            f"pose={np.concatenate([xyz, rpy])}"
        )

    return frames_directory, frame_number - 1


def create_mp4(
    frames_directory: Path,
    output_video: Path,
    fps: int,
) -> None:

    ffmpeg_executable = shutil.which("ffmpeg")

    if ffmpeg_executable is None:
        raise RuntimeError(
            "ffmpeg is not installed.\n"
            "Install it with:\n\n"
            "    sudo apt install ffmpeg\n"
        )

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        ffmpeg_executable,
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

    print("\nCreating MP4 with ffmpeg...")
    subprocess.run(command, check=True)


def main() -> int:
    arguments = parse_arguments()

    if arguments.top_k <= 0:
        print(
            "ERROR: --top-k must be greater than zero.",
            file=sys.stderr,
        )
        return 1

    if arguments.fps <= 0:
        print(
            "ERROR: --fps must be greater than zero.",
            file=sys.stderr,
        )
        return 1

    if arguments.seconds_per_pose <= 0:
        print(
            "ERROR: --seconds-per-pose must be greater than zero.",
            file=sys.stderr,
        )
        return 1

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    client_id = p.connect(p.DIRECT)

    if client_id < 0:
        print(
            "ERROR: PyBullet connection failed.",
            file=sys.stderr,
        )
        return 1

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    try:
        frames_directory, total_frames = generate_frames(
            object_name=arguments.object,
            top_k=arguments.top_k,
            seconds_per_pose=arguments.seconds_per_pose,
            fps=arguments.fps,
            output_directory=arguments.output_dir,
        )

        output_video = (
            arguments.output_dir
            / arguments.object
            / f"{arguments.object}_top{arguments.top_k}_gfi.mp4"
        )

        create_mp4(
            frames_directory=frames_directory,
            output_video=output_video,
            fps=arguments.fps,
        )

        print("\nCompleted successfully.")
        print(f"Total frames: {total_frames}")
        print(f"Video saved to: {output_video}")

    except Exception as error:
        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )
        return 1

    finally:
        p.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
