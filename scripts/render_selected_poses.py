
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image


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


OUT_DIR = "results/paper_pose_figures"

TABLE_RGBA = [0.66, 0.49, 0.32, 1.0]
MUG_RGBA = [0.44, 0.11, 0.11, 1.0]


def convert_candidate(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, dict):
        return candidate

    if hasattr(candidate, "item"):
        converted = candidate.item()

        if isinstance(converted, dict):
            return converted

    raise TypeError(
        "Candidate must be a dictionary or a NumPy object "
        "containing a dictionary."
    )


def apply_colour_to_body(
    body_id: int,
    rgba_colour: list[float],
) -> None:

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

    image = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
    )

    rgba = np.asarray(
        image[2],
        dtype=np.uint8,
    ).reshape(height, width, 4)

    return Image.fromarray(
        rgba[:, :, :3],
        mode="RGB",
    )


def main() -> int:
    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    client_id = p.connect(p.DIRECT)

    if client_id < 0:
        print("ERROR: PyBullet connection failed.")
        return 1

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    try:
        for object_name, configuration in OBJECTS.items():
            print(f"\nRendering: {object_name}")

            top_path = os.path.join(
                configuration["run"],
                "top10_grasp_candidates.npy",
            )

            object_urdf = configuration["urdf"]

            if not os.path.exists(top_path):
                print(f"Missing candidate file: {top_path}")
                continue

            if not os.path.exists(object_urdf):
                print(f"Missing URDF: {object_urdf}")
                continue

            candidates = np.load(
                top_path,
                allow_pickle=True,
            )

            if len(candidates) == 0:
                print(f"Empty candidate file: {top_path}")
                continue

            candidate = convert_candidate(
                candidates[0]
            )

            final_pose = candidate["final_pose"]

            xyz = np.asarray(
                final_pose["xyz"],
                dtype=float,
            )

            rpy = np.asarray(
                final_pose["euler_rpy"],
                dtype=float,
            )

            quaternion = p.getQuaternionFromEuler(
                rpy.tolist()
            )

            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setAdditionalSearchPath(
                pybullet_data.getDataPath()
            )

            p.loadURDF("plane.urdf")

            table_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.35, 0.35, 0.02],
            )

            table_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.35, 0.35, 0.02],
                rgbaColor=TABLE_RGBA,
            )

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=table_collision,
                baseVisualShapeIndex=table_visual,
                basePosition=[0.0, 0.0, 0.20],
            )

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

            rendered_image = render_scene()

            output_path = os.path.join(
                OUT_DIR,
                f"{object_name}_best_grasp_pose.png",
            )

            rendered_image.save(output_path)

            print(f"Saved: {output_path}")

    finally:
        p.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
