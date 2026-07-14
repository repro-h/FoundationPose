#!/usr/bin/env python3
"""Run FoundationPose on one HO3D sequence without using GT object poses."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import draw_posed_3d_box, draw_xyz_axis, dr, set_logging_format, set_seed


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Register an object from the first selected HO3D frame, then track it "
      "through the sequence using RGB-D. HO3D objRot/objTrans are never read."
    )
  )
  parser.add_argument("--ho3d_root", required=True, help="HO3D train root containing <sequence>/rgb.")
  parser.add_argument("--sequence", required=True)
  parser.add_argument("--anchor_npz", required=True, help="Canonical MV/SAM3D anchor NPZ.")
  parser.add_argument(
    "--scale_json",
    default=None,
    help="Optional 2D scale correction JSON with final_global_scale or correction_factor.",
  )
  parser.add_argument("--out_dir", required=True)
  parser.add_argument("--start_frame", type=int, default=0)
  parser.add_argument("--end_frame", type=int, default=100)
  parser.add_argument("--init_frame", type=int, default=None)
  parser.add_argument("--frame_stride", type=int, default=1, help="Tracking stride; keep 1 for reliable tracking.")
  parser.add_argument("--est_refine_iter", type=int, default=5)
  parser.add_argument("--track_refine_iter", type=int, default=2)
  parser.add_argument("--debug", type=int, default=1)
  parser.add_argument("--save_overlays", action="store_true")
  parser.add_argument("--overwrite", action="store_true")
  return parser.parse_args()


def load_anchor_mesh(anchor_path: Path, scale_json_path: Path | None) -> tuple[trimesh.Trimesh, dict[str, object]]:
  raw = np.load(anchor_path, allow_pickle=True)
  vertices = np.asarray(raw["vertices"], dtype=np.float32).reshape(-1, 3)
  faces = np.asarray(raw["faces"], dtype=np.int64).reshape(-1, 3)
  decoded_scale = float(np.asarray(raw["scale"], dtype=np.float32).reshape(-1)[0])
  final_scale = decoded_scale
  scale_source = "anchor.scale"
  if scale_json_path is not None:
    scale_info = json.loads(scale_json_path.read_text(encoding="utf-8"))
    if "final_global_scale" in scale_info:
      final_scale = float(scale_info["final_global_scale"])
      scale_source = "scale_json.final_global_scale"
    elif "correction_factor" in scale_info:
      final_scale = decoded_scale * float(scale_info["correction_factor"])
      scale_source = "scale_json.correction_factor"
    else:
      raise KeyError(f"No final_global_scale or correction_factor in {scale_json_path}")
  mesh = trimesh.Trimesh(vertices=vertices * final_scale, faces=faces, process=False)
  mesh.remove_unreferenced_vertices()
  return mesh, {
    "decoded_scale": decoded_scale,
    "final_global_scale": final_scale,
    "scale_source": scale_source,
  }


def decode_ho3d_depth(path: Path) -> np.ndarray:
  raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
  if raw is None:
    raise FileNotFoundError(path)
  if raw.ndim == 3:
    # cv2 reads HO3D's encoded RGB(A) depth as BGR(A): depth=(R+256*G)/10000 m.
    depth = (raw[..., 2].astype(np.float32) + 256.0 * raw[..., 1].astype(np.float32)) / 10000.0
  else:
    depth = raw.astype(np.float32)
    valid = depth > 0
    if valid.any() and float(np.median(depth[valid])) > 10.0:
      depth /= 1000.0
  depth[~np.isfinite(depth) | (depth < 0.001)] = 0.0
  return depth


def load_mask(path: Path, image_hw: tuple[int, int]) -> np.ndarray:
  mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
  if mask is None:
    raise FileNotFoundError(path)
  if mask.shape != image_hw:
    mask = cv2.resize(mask, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST)
  # obj_mask_white is expected to use white foreground.
  return mask > 127


def find_frame_file(directory: Path, frame: str, suffix: str = ".png") -> Path:
  candidates = [directory / f"{frame}{suffix}", directory / f"{frame}_mask{suffix}"]
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  raise FileNotFoundError(f"No file for frame {frame} in {directory}")


def load_intrinsics(meta_path: Path) -> np.ndarray:
  with meta_path.open("rb") as handle:
    meta = pickle.load(handle, encoding="latin1")
  if "camMat" not in meta:
    raise KeyError(f"camMat missing from {meta_path}")
  return np.asarray(meta["camMat"], dtype=np.float32).reshape(3, 3)


def pose_record(frame: str, pose: np.ndarray, mode: str, mask: np.ndarray, depth: np.ndarray) -> dict[str, object]:
  valid_depth = mask & (depth >= 0.001)
  return {
    "frame": frame,
    "mode": mode,
    "object_in_camera": np.asarray(pose, dtype=np.float64).reshape(4, 4).tolist(),
    "mask_pixels": int(mask.sum()),
    "valid_mask_depth_pixels": int(valid_depth.sum()),
    "translation_m": np.asarray(pose[:3, 3], dtype=np.float64).tolist(),
  }


def main() -> None:
  args = parse_args()
  if args.frame_stride < 1:
    raise ValueError("--frame_stride must be >= 1")
  set_logging_format()
  set_seed(0)

  sequence_dir = Path(args.ho3d_root).expanduser().resolve() / args.sequence
  rgb_dir = sequence_dir / "rgb"
  depth_dir = sequence_dir / "depth"
  mask_dir = sequence_dir / "obj_mask_white"
  meta_dir = sequence_dir / "meta"
  all_rgb = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
  by_frame = {path.stem: path for path in all_rgb}
  selected = [
    frame for frame in sorted(by_frame)
    if args.start_frame <= int(frame) <= args.end_frame
    and (int(frame) - args.start_frame) % args.frame_stride == 0
  ]
  if not selected:
    raise ValueError(f"No frames selected from {rgb_dir}")
  init_frame = f"{args.init_frame:04d}" if args.init_frame is not None else selected[0]
  if init_frame not in selected:
    raise ValueError("--init_frame must be included by the selected frame range/stride")
  selected = selected[selected.index(init_frame):]

  out_dir = Path(args.out_dir).expanduser().resolve()
  pose_path = out_dir / "foundationpose_poses.json"
  if pose_path.exists() and not args.overwrite:
    raise FileExistsError(f"Output exists: {pose_path}; pass --overwrite to replace it")
  overlay_dir = out_dir / "overlays"
  fp_debug_dir = out_dir / "foundationpose_debug"
  overlay_dir.mkdir(parents=True, exist_ok=True)
  fp_debug_dir.mkdir(parents=True, exist_ok=True)

  scale_json_path = Path(args.scale_json).expanduser().resolve() if args.scale_json else None
  mesh, scale_info = load_anchor_mesh(Path(args.anchor_npz).expanduser().resolve(), scale_json_path)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  estimator = FoundationPose(
    model_pts=np.asarray(mesh.vertices),
    model_normals=np.asarray(mesh.vertex_normals),
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=str(fp_debug_dir),
    debug=args.debug,
    glctx=glctx,
  )

  rows: dict[str, dict[str, object]] = {}
  for index, frame in enumerate(selected):
    rgb = imageio.imread(by_frame[frame])[..., :3]
    depth = decode_ho3d_depth(find_frame_file(depth_dir, frame))
    mask = load_mask(find_frame_file(mask_dir, frame), rgb.shape[:2])
    K = load_intrinsics(meta_dir / f"{frame}.pkl")
    if depth.shape != rgb.shape[:2]:
      raise ValueError(f"Depth/RGB shape mismatch for {frame}: {depth.shape} vs {rgb.shape[:2]}")

    if index == 0:
      mode = "register"
      pose = estimator.register(
        K=K,
        rgb=rgb,
        depth=depth,
        ob_mask=mask,
        iteration=args.est_refine_iter,
      )
    else:
      mode = "track"
      pose = estimator.track_one(
        rgb=rgb,
        depth=depth,
        K=K,
        iteration=args.track_refine_iter,
      )
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(pose).all():
      raise RuntimeError(f"FoundationPose returned a non-finite pose for frame {frame}")
    rows[frame] = pose_record(frame, pose, mode, mask, depth)

    np.savetxt(out_dir / f"{frame}.txt", pose)
    if args.save_overlays:
      center_pose = pose @ np.linalg.inv(to_origin)
      overlay = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=center_pose, bbox=bbox)
      overlay = draw_xyz_axis(
        overlay,
        ob_in_cam=center_pose,
        scale=max(float(np.max(extents)) * 0.6, 0.03),
        K=K,
        thickness=2,
        transparency=0,
        is_input_rgb=True,
      )
      imageio.imwrite(overlay_dir / f"{frame}.jpg", overlay)
    print(f"[{index + 1}/{len(selected)}] frame={frame} mode={mode} t={pose[:3, 3].tolist()}", flush=True)

    payload = {
      "source": "foundationpose_ho3d_rgbd_tracking",
      "sequence": args.sequence,
      "coordinate_system": "opencv_camera",
      "pose_convention": "object_model_to_camera",
      "uses_gt_object_pose": False,
      "init_frame": init_frame,
      "anchor_npz": str(Path(args.anchor_npz).expanduser().resolve()),
      "scale_json": str(scale_json_path) if scale_json_path else None,
      **scale_info,
      "frames": rows,
      "by_frame": rows,
    }
    tmp_path = pose_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, pose_path)

  print(json.dumps({"num_frames": len(rows), "pose_json": str(pose_path), **scale_info}, indent=2))


if __name__ == "__main__":
  main()
