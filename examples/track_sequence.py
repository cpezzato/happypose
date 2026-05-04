"""
6D pose tracking over a sequence of images using CosyPose.
Saves a mesh-overlay video to examples/tracking_output.mp4.

Usage:
    uv run python examples/track_sequence.py
    # or with a custom data dir:
    HAPPYPOSE_DATA_DIR=~/my_data uv run python examples/track_sequence.py

The mustard0 dataset is expected at $HAPPYPOSE_DATA_DIR/mustard0.
Run examples/download_mustard0.py first if you don't have it.

Note: CosyPose models here are trained on HOPE dataset objects. The mustard
bottle is a YCB object, so pose accuracy may be limited. MegaPose generalises
better to novel objects.
"""

import os
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import (
    CosyPoseWrapper,
)
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.inference.types import ObservationTensor, PoseEstimatesType
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d
from happypose.toolbox.utils.logging import get_logger, set_logging_level
from happypose.toolbox.utils.tensor_collection import PandasTensorCollection
from happypose.toolbox.visualization.utils import make_contour_overlay

logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_data_root = Path(os.environ.get("HAPPYPOSE_DATA_DIR", Path.home() / "happypose_data")).expanduser().resolve()
DATASET_DIR = _data_root / "mustard0"
OBJECT_LABEL = "mustard0"
MAX_FRAMES = 1000
OUTPUT_VIDEO = Path(__file__).parent / "tracking_output.mp4"
FPS = 30

LIGHT_DATAS = [Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1))]

TRANS_CONVERGED_M = 0.005    # 5 mm
ROT_CONVERGED_DEG = 2.0      # 2 degrees
STABLE_FRAMES_NEEDED = 3


def pose_delta(pose_a: np.ndarray, pose_b: np.ndarray) -> Tuple[float, float]:
    """Return (translation_delta_m, rotation_delta_deg) between two 4x4 poses."""
    trans_delta = float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))
    R_rel = pose_a[:3, :3].T @ pose_b[:3, :3]
    angle_rad = np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    return trans_delta, float(np.degrees(angle_rad))


def load_K() -> np.ndarray:
    return np.loadtxt(DATASET_DIR / "cam_K.txt")


def load_sequence(load_depth: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Yield (rgb [H,W,3] uint8, depth [H,W] float32 meters or None, K [3,3])."""
    K = load_K()
    frames = sorted((DATASET_DIR / "rgb").glob("*.png"), key=lambda p: int(p.stem))
    for path in frames[:MAX_FRAMES]:
        rgb = np.array(Image.open(path), dtype=np.uint8)[:, :, :3]  # RGBA -> RGB
        depth = np.array(Image.open(DATASET_DIR / "depth" / path.name), dtype=np.float32) / 1000.0 if load_depth else None
        yield rgb, depth, K


def make_first_frame_detection() -> PandasTensorCollection:
    """Build a detection from the single mask file (covers frame 0)."""
    mask_path = sorted((DATASET_DIR / "masks").glob("*.png"))[0]
    mask = np.array(Image.open(mask_path))
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    ymin, ymax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    xmin, xmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    infos = pd.DataFrame({
        "label": [OBJECT_LABEL],
        "batch_im_id": [0],
        "instance_id": [0],
    })
    bboxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
    return PandasTensorCollection(infos=infos, bboxes=bboxes)


def make_observation(rgb: np.ndarray, depth: np.ndarray | None, K: np.ndarray) -> ObservationTensor:
    return ObservationTensor.from_numpy(rgb, depth=depth, K=K).to(device)


AXIS_LENGTH_M = 0.05  # 5 cm axes
AXIS_COLORS = {"x": (0, 0, 255), "y": (0, 255, 0), "z": (255, 0, 0)}  # BGR


def draw_axes(img: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Draw XYZ coordinate frame projected from the object pose onto the image."""
    def project(pt3d: np.ndarray) -> Tuple[int, int]:
        p = K @ pt3d
        return (int(p[0] / p[2]), int(p[1] / p[2]))

    origin = pose[:3, 3]
    tip_x = origin + pose[:3, 0] * AXIS_LENGTH_M
    tip_y = origin + pose[:3, 1] * AXIS_LENGTH_M
    tip_z = origin + pose[:3, 2] * AXIS_LENGTH_M

    o2d = project(origin)
    out = img.copy()
    for tip, color in [(tip_x, AXIS_COLORS["x"]), (tip_y, AXIS_COLORS["y"]), (tip_z, AXIS_COLORS["z"])]:
        cv2.arrowedLine(out, o2d, project(tip), color, 2, tipLength=0.2)
    return out


def render_overlay(
    renderer: Panda3dSceneRenderer,
    rgb: np.ndarray,
    K: np.ndarray,
    preds: PoseEstimatesType,
) -> np.ndarray:
    """Render mesh contour overlay + coordinate axes for the predicted poses."""
    labels = preds.infos["label"].tolist()
    poses = preds.poses.numpy()

    object_datas: List[ObjectData] = [
        ObjectData(label=label, TWO=Transform(pose))
        for label, pose in zip(labels, poses)
    ]

    cam = CameraData()
    cam.K = K
    cam.resolution = rgb.shape[:2]
    cam.TWC = Transform(np.eye(4))

    cam_p3d, obj_p3d = convert_scene_observation_to_panda3d(cam, object_datas)
    renderings = renderer.render_scene(
        obj_p3d,
        [cam_p3d],
        LIGHT_DATAS,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]

    out = make_contour_overlay(rgb, renderings.rgb, dilate_iterations=2, color=(0, 255, 0))["img"]
    for pose in poses:
        out = draw_axes(out, pose, K)
    return out


def main() -> None:
    set_logging_level("info")

    object_dataset = RigidObjectDataset([
        RigidObject(
            label=OBJECT_LABEL,
            mesh_path=DATASET_DIR / "mesh" / "textured_simple.obj",
            mesh_units="m",
        )
    ])

    logger.info("Loading CosyPose hope models...")
    cosy = CosyPoseWrapper(
        dataset_name="hope",
        object_dataset=object_dataset,
        n_workers=1,
        depth_refiner_type="icp",
    )
    renderer = Panda3dSceneRenderer(object_dataset)

    first_frame_detection = make_first_frame_detection().to(device)
    prev_estimates: PoseEstimatesType | None = None

    first_rgb = np.array(Image.open(
        sorted((DATASET_DIR / "rgb").glob("*.png"), key=lambda p: int(p.stem))[0]
    ))[:, :, :3]
    H, W = first_rgb.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (W, H))
    logger.info(f"Writing video to {OUTPUT_VIDEO}")

    converged = False
    stable_count = 0

    try:
        for frame_idx, (rgb, depth, K) in enumerate(load_sequence()):
            obs = make_observation(rgb, depth, K)

            if not converged:
                # Init phase: coarse only on frame 0, warm-start + ICP after.
                if prev_estimates is None:
                    preds, extra = cosy.pose_predictor.run_inference_pipeline(
                        observation=obs,
                        detections=first_frame_detection,
                        run_detector=False,
                        n_coarse_iterations=1,
                        n_refiner_iterations=5,
                    )
                else:
                    prev_on_device = prev_estimates.to(device)
                    preds, extra = cosy.pose_predictor.run_inference_pipeline(
                        observation=obs,
                        coarse_estimates=prev_on_device,
                        data_TCO_init=prev_on_device,
                        n_coarse_iterations=0,
                        n_refiner_iterations=5,
                    )
                preds, _ = cosy.depth_refiner.refine_poses(
                    predictions=preds, depth=obs.depth, K=obs.K
                )
                preds = preds.cpu()

                if prev_estimates is not None:
                    dt, dr = pose_delta(prev_estimates.poses[0].cpu().numpy(), preds.poses[0].numpy())
                    stable_count = stable_count + 1 if dt < TRANS_CONVERGED_M and dr < ROT_CONVERGED_DEG else 0
                    if stable_count >= STABLE_FRAMES_NEEDED:
                        converged = True
                        logger.info(f"Frame {frame_idx:03d}: converged — switching to RGB-only tracking")
                    else:
                        logger.info(f"Frame {frame_idx:03d}: INIT+ICP  dt={dt*1000:.1f}mm  dr={dr:.1f}deg  stable={stable_count}/{STABLE_FRAMES_NEEDED}")
                else:
                    logger.info(f"Frame {frame_idx:03d}: INIT+ICP  first frame")
            else:
                # Tracking phase: RGB-only warm-start, no ICP.
                prev_on_device = prev_estimates.to(device)
                preds, extra = cosy.pose_predictor.run_inference_pipeline(
                    observation=obs,
                    coarse_estimates=prev_on_device,
                    data_TCO_init=prev_on_device,
                    n_coarse_iterations=0,
                    n_refiner_iterations=3,
                )
                preds = preds.cpu()
                T = preds.poses[0, :3, 3].numpy()
                logger.info(f"Frame {frame_idx:03d}: TRACK  T={T}  {extra['timing_str']}")

            prev_estimates = preds

            overlay = render_overlay(renderer, rgb, K, preds)
            status_text, status_color = ("TRACKING", (0, 255, 0)) if converged else ("INIT+ICP", (0, 165, 255))
            cv2.putText(overlay, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        logger.info(f"Saved {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
