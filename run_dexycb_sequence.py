#!/usr/bin/env python3
"""Register and RGB-D track a reconstructed object in one DexYCB camera stream."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh
import yaml

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import draw_posed_3d_box, draw_xyz_axis, dr, set_logging_format, set_seed


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Register a reconstructed object with DexYCB RGB-D and its visible object mask, "
      "then track every frame without reading DexYCB pose_y."
    )
  )
  parser.add_argument("--stream_dir", required=True, help="DexYCB camera directory containing color_*.jpg.")
  parser.add_argument("--mesh_file", required=True, help="Canonical object GLB/OBJ/PLY used by FoundationPose.")
  parser.add_argument(
    "--mesh_scale",
    type=float,
    default=1.0,
    help="Metric scale applied once to mesh vertices before registration.",
  )
  parser.add_argument("--intrinsics", required=True, help="3x3 K stored as JSON, NPY, NPZ, or text.")
  parser.add_argument("--out_dir", required=True)
  parser.add_argument("--start_frame", type=int, default=0)
  parser.add_argument("--end_frame", type=int, default=999999)
  parser.add_argument("--init_frame", type=int, default=None)
  parser.add_argument("--frame_stride", type=int, default=1)
  parser.add_argument("--bidirectional", action="store_true")
  parser.add_argument("--est_refine_iter", type=int, default=5)
  parser.add_argument("--track_refine_iter", type=int, default=2)
  parser.add_argument("--score_weights_dir", default=None)
  parser.add_argument("--refine_weights_dir", default=None)
  parser.add_argument("--debug", type=int, default=1)
  parser.add_argument("--save_overlays", action="store_true")
  parser.add_argument("--overwrite", action="store_true")
  return parser.parse_args()


def frame_token(image_path: Path) -> str:
  if not image_path.stem.startswith("color_"):
    raise ValueError(f"Unexpected DexYCB image filename: {image_path.name}")
  return image_path.stem.split("_", 1)[1]


def load_intrinsics(path: Path) -> np.ndarray:
  suffix = path.suffix.lower()
  if suffix == ".npy":
    raw = np.load(path)
  elif suffix == ".npz":
    archive = np.load(path)
    key = "K" if "K" in archive else ("intrinsics" if "intrinsics" in archive else archive.files[0])
    raw = archive[key]
  elif suffix == ".json":
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
      payload = payload.get("K", payload.get("intrinsics", payload))
    raw = payload
  else:
    raw = np.loadtxt(path)
  K = np.asarray(raw, dtype=np.float64)
  if K.size == 9:
    K = K.reshape(3, 3)
  if K.shape != (3, 3) or not np.isfinite(K).all():
    raise ValueError(f"Invalid 3x3 intrinsics in {path}: shape={K.shape}")
  return K


def load_mesh(path: Path, scale: float) -> trimesh.Trimesh:
  loaded = trimesh.load(path, process=False)
  if isinstance(loaded, trimesh.Scene):
    geometries = [
      geometry.copy()
      for geometry in loaded.geometry.values()
      if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0
    ]
    if not geometries:
      raise ValueError(f"No triangle geometry in {path}")
    loaded = trimesh.util.concatenate(geometries)
  if not isinstance(loaded, trimesh.Trimesh):
    raise ValueError(f"No triangle mesh in {path}")
  mesh = loaded.copy()
  mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
  mesh.remove_unreferenced_vertices()
  return mesh


def target_object_id(stream_dir: Path) -> tuple[int | None, Path]:
  meta_path = stream_dir.parent / "meta.yml"
  if not meta_path.is_file():
    return None, meta_path
  with meta_path.open("r", encoding="utf-8") as handle:
    meta = yaml.safe_load(handle) or {}
  ycb_ids = list(meta.get("ycb_ids", []) or [])
  grasp_index = int(meta.get("ycb_grasp_ind", 0))
  if not 0 <= grasp_index < len(ycb_ids):
    return None, meta_path
  return int(ycb_ids[grasp_index]), meta_path


def load_object_mask(stream_dir: Path, frame: str, object_id: int | None, image_hw: tuple[int, int]) -> np.ndarray:
  label_path = stream_dir / f"labels_{frame}.npz"
  if not label_path.is_file():
    raise FileNotFoundError(label_path)
  with np.load(label_path) as archive:
    if "seg" not in archive:
      raise KeyError(f"seg missing from {label_path}")
    seg = np.asarray(archive["seg"])
  if seg.ndim == 3:
    seg = seg[0]
  mask = (seg == object_id) if object_id is not None else ((seg > 0) & (seg != 255))
  if mask.shape != image_hw:
    mask = cv2.resize(mask.astype(np.uint8), (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
  return np.asarray(mask, dtype=bool)


def depth_path(stream_dir: Path, frame: str) -> Path:
  candidates = [
    stream_dir / f"aligned_depth_to_color_{frame}.png",
    stream_dir / f"depth_{frame}.png",
  ]
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  raise FileNotFoundError(f"No DexYCB depth image for frame={frame}; tried={candidates}")


def load_depth(stream_dir: Path, frame: str, image_hw: tuple[int, int]) -> tuple[np.ndarray, Path]:
  path = depth_path(stream_dir, frame)
  raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
  if raw is None:
    raise FileNotFoundError(path)
  if raw.ndim == 3:
    raw = raw[..., 0]
  depth = np.asarray(raw, dtype=np.float32)
  valid = np.isfinite(depth) & (depth > 0)
  if valid.any() and float(np.median(depth[valid])) > 10.0:
    depth /= 1000.0
  if depth.shape != image_hw:
    depth = cv2.resize(depth, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST)
  depth[~np.isfinite(depth) | (depth < 0.001)] = 0.0
  return depth, path


def write_json_atomic(path: Path, payload: object) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  os.replace(temporary, path)


def main() -> None:
  args = parse_args()
  if args.frame_stride < 1:
    raise ValueError("--frame_stride must be >= 1")
  set_logging_format()
  set_seed(0)

  stream_dir = Path(args.stream_dir).expanduser().resolve()
  mesh_path = Path(args.mesh_file).expanduser().resolve()
  intrinsics_path = Path(args.intrinsics).expanduser().resolve()
  out_dir = Path(args.out_dir).expanduser().resolve()
  pose_path = out_dir / "foundationpose_poses.json"
  if pose_path.exists() and not args.overwrite:
    raise FileExistsError(f"Output exists: {pose_path}; pass --overwrite")

  all_images = sorted(stream_dir.glob("color_*.jpg")) + sorted(stream_dir.glob("color_*.png"))
  by_frame = {frame_token(path): path for path in all_images}
  selected = [
    frame
    for frame in sorted(by_frame, key=int)
    if args.start_frame <= int(frame) <= args.end_frame
    and (int(frame) - args.start_frame) % args.frame_stride == 0
  ]
  if not selected:
    raise ValueError(f"No selected frames in {stream_dir}")
  width = max(len(frame) for frame in selected)
  init_frame = f"{args.init_frame:0{width}d}" if args.init_frame is not None else selected[0]
  if init_frame not in selected:
    raise ValueError(f"init frame {init_frame} is absent from selected frames")

  out_dir.mkdir(parents=True, exist_ok=True)
  overlay_dir = out_dir / "overlays"
  debug_dir = out_dir / "foundationpose_debug"
  overlay_dir.mkdir(parents=True, exist_ok=True)
  debug_dir.mkdir(parents=True, exist_ok=True)

  K = load_intrinsics(intrinsics_path)
  mesh = load_mesh(mesh_path, args.mesh_scale)
  tracked_anchor_path = out_dir / "tracked_model_anchor.npz"
  np.savez_compressed(
    tracked_anchor_path,
    vertices=np.asarray(mesh.vertices, dtype=np.float32),
    faces=np.asarray(mesh.faces, dtype=np.int64),
    scale=np.asarray([1.0], dtype=np.float32),
    source_model=np.asarray(str(mesh_path)),
    source_mesh_scale=np.asarray([float(args.mesh_scale)], dtype=np.float32),
  )
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)

  score_weights = str(Path(args.score_weights_dir).expanduser().resolve()) if args.score_weights_dir else None
  refine_weights = str(Path(args.refine_weights_dir).expanduser().resolve()) if args.refine_weights_dir else None
  scorer = ScorePredictor(weights_dir=score_weights)
  refiner = PoseRefinePredictor(weights_dir=refine_weights)
  estimator = FoundationPose(
    model_pts=np.asarray(mesh.vertices),
    model_normals=np.asarray(mesh.vertex_normals),
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=str(debug_dir),
    debug=args.debug,
    glctx=dr.RasterizeCudaContext(),
  )
  object_id, meta_path = target_object_id(stream_dir)
  rows: dict[str, dict[str, object]] = {}
  depth_files: dict[str, str] = {}

  def frame_inputs(frame: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = imageio.imread(by_frame[frame])[..., :3]
    depth, used_depth_path = load_depth(stream_dir, frame, rgb.shape[:2])
    mask = load_object_mask(stream_dir, frame, object_id, rgb.shape[:2])
    depth_files[frame] = str(used_depth_path)
    return rgb, depth, mask

  def render_overlay(rgb: np.ndarray, pose: np.ndarray) -> np.ndarray:
    center_pose = pose @ np.linalg.inv(to_origin)
    overlay = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=center_pose, bbox=bbox)
    return draw_xyz_axis(
      overlay,
      ob_in_cam=center_pose,
      scale=max(float(np.max(extents)) * 0.6, 0.03),
      K=K,
      thickness=2,
      transparency=0,
      is_input_rgb=True,
    )

  def write_payload() -> None:
    ordered = {frame: rows[frame] for frame in selected if frame in rows}
    payload = {
      "source": "foundationpose_dexycb_rgbd_tracking",
      "stream_dir": str(stream_dir),
      "coordinate_system": "opencv_camera",
      "pose_convention": "object_model_to_camera",
      "uses_gt_object_pose": False,
      "foundationpose_input_mode": "dexycb_rgbd_register_and_track",
      "depth_source": "dexycb_aligned_depth_to_color",
      "object_mask_source": "dexycb_seg_target_ycb_id_register_only",
      "target_ycb_id": object_id,
      "meta_path": str(meta_path),
      "init_frame": init_frame,
      "bidirectional": bool(args.bidirectional),
      "model_source": "mesh_file",
      "model_path": str(mesh_path),
      "mesh_file": str(mesh_path),
      "tracked_model_anchor": str(tracked_anchor_path),
      "decoded_scale": 1.0,
      "final_global_scale": 1.0,
      "scale_source": "mesh_file.baked_vertices",
      "source_mesh_scale": float(args.mesh_scale),
      "intrinsics": K.tolist(),
      "intrinsics_path": str(intrinsics_path),
      "score_weights_dir": score_weights or "default",
      "refine_weights_dir": refine_weights or "default",
      "depth_files": depth_files,
      "frames": ordered,
      "by_frame": ordered,
    }
    write_json_atomic(pose_path, payload)

  def save_result(frame: str, pose: np.ndarray, mode: str) -> None:
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(pose).all():
      raise RuntimeError(f"Non-finite pose for frame={frame}")
    rgb, depth, mask = frame_inputs(frame)
    valid_mask_depth = mask & (depth >= 0.001)
    rows[frame] = {
      "frame": frame,
      "mode": mode,
      "object_in_camera": pose.tolist(),
      "translation_m": pose[:3, 3].tolist(),
      "mask_pixels": int(mask.sum()),
      "valid_mask_depth_pixels": int(valid_mask_depth.sum()),
      "depth_path": depth_files[frame],
    }
    np.savetxt(out_dir / f"{frame}.txt", pose)
    if args.save_overlays:
      imageio.imwrite(overlay_dir / f"{frame}.jpg", render_overlay(rgb, pose))
    write_payload()
    print(f"[{len(rows)}/{len(selected)}] frame={frame} mode={mode} t={pose[:3, 3].tolist()}", flush=True)

  rgb, depth, mask = frame_inputs(init_frame)
  init_pose = estimator.register(
    K=K,
    rgb=rgb,
    depth=depth,
    ob_mask=mask,
    iteration=args.est_refine_iter,
  )
  init_pose = np.asarray(init_pose, dtype=np.float64).reshape(4, 4)
  init_state = estimator.pose_last.detach().clone()
  save_result(init_frame, init_pose, "register")
  init_index = selected.index(init_frame)

  estimator.pose_last = init_state.detach().clone()
  for frame in selected[init_index + 1:]:
    rgb, depth, _ = frame_inputs(frame)
    pose = estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=args.track_refine_iter)
    save_result(frame, pose, "track_forward")

  if args.bidirectional:
    estimator.pose_last = init_state.detach().clone()
    for frame in reversed(selected[:init_index]):
      rgb, depth, _ = frame_inputs(frame)
      pose = estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=args.track_refine_iter)
      save_result(frame, pose, "track_backward")

  print(json.dumps({
    "num_frames": len(rows),
    "pose_json": str(pose_path),
    "selected_init_frame": init_frame,
    "bidirectional": bool(args.bidirectional),
    "target_ycb_id": object_id,
    "source_mesh_scale": float(args.mesh_scale),
  }, indent=2), flush=True)


if __name__ == "__main__":
  main()
