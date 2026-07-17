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
import torch
import trimesh

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import (
  draw_posed_3d_box,
  draw_xyz_axis,
  dr,
  nvdiffrast_render,
  set_logging_format,
  set_seed,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Register an object from the first selected HO3D frame, then track it "
      "through the sequence using RGB-D. HO3D objRot/objTrans are never read."
    )
  )
  parser.add_argument("--ho3d_root", required=True, help="HO3D train root containing <sequence>/rgb.")
  parser.add_argument("--sequence", required=True)
  model_group = parser.add_mutually_exclusive_group(required=True)
  model_group.add_argument("--anchor_npz", help="Canonical MV/SAM3D anchor NPZ.")
  model_group.add_argument("--mesh_file", help="Metric object mesh, such as a textured YCB CAD OBJ.")
  parser.add_argument(
    "--mesh_scale",
    type=float,
    default=1.0,
    help="Scale applied to --mesh_file vertices. HO3D YCB models normally use 1.0.",
  )
  parser.add_argument(
    "--scale_json",
    default=None,
    help="Optional 2D scale correction JSON with final_global_scale or correction_factor.",
  )
  parser.add_argument("--out_dir", required=True)
  parser.add_argument("--start_frame", type=int, default=0)
  parser.add_argument("--end_frame", type=int, default=100)
  parser.add_argument("--init_frame", type=int, default=None)
  parser.add_argument(
    "--auto_init_frames",
    type=int,
    nargs="+",
    default=None,
    help="Independently register and score these candidate frames, then use the best one.",
  )
  parser.add_argument(
    "--bidirectional",
    action="store_true",
    help="Track both forward and backward from the selected initialization frame.",
  )
  parser.add_argument(
    "--auto_init_depth_tolerance_mm",
    type=float,
    default=12.0,
    help="Occlusion tolerance used when scoring rendered candidate depth.",
  )
  parser.add_argument(
    "--auto_init_score_tie_margin",
    type=float,
    default=0.02,
    help="Prefer a temporally central candidate when its score is within this margin of the best.",
  )
  parser.add_argument("--frame_stride", type=int, default=1, help="Tracking stride; keep 1 for reliable tracking.")
  parser.add_argument("--est_refine_iter", type=int, default=5)
  parser.add_argument("--track_refine_iter", type=int, default=2)
  parser.add_argument(
    "--score_weights_dir",
    default=None,
    help="Optional ScorePredictor directory containing config.yml and model_best.pth.",
  )
  parser.add_argument(
    "--refine_weights_dir",
    default=None,
    help="Optional PoseRefinePredictor directory containing config.yml and model_best.pth.",
  )
  parser.add_argument(
    "--rgb_only",
    action="store_true",
    help=(
      "Use RGB-D for initial registration, then pass zero depth during tracking. "
      "GRAIL uses RGB-only tracking but starts from a known first-frame pose; "
      "HO3D registration still needs depth to initialize translation."
    ),
  )
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


def load_mesh_file(mesh_path: Path, mesh_scale: float) -> tuple[trimesh.Trimesh, dict[str, object]]:
  loaded = trimesh.load(mesh_path, process=False)
  if isinstance(loaded, trimesh.Scene):
    geometries = [
      geometry for geometry in loaded.geometry.values()
      if hasattr(geometry, "vertices") and hasattr(geometry, "faces")
    ]
    if not geometries:
      raise ValueError(f"No triangle geometry found in {mesh_path}")
    # FoundationPose accepts one Trimesh. Keep the largest geometry intact so
    # TextureVisuals/UV/material data survive instead of being discarded by a
    # generic scene concatenation.
    loaded = max(geometries, key=lambda geometry: len(geometry.faces)).copy()
  if not isinstance(loaded, trimesh.Trimesh):
    raise ValueError(f"No triangle mesh found in {mesh_path}")
  mesh = loaded.copy()
  mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(mesh_scale)
  mesh.remove_unreferenced_vertices()
  return mesh, {
    # mesh_scale is already baked into the exported tracked_model_anchor
    # vertices, so downstream consumers must apply a unit scale.
    "decoded_scale": 1.0,
    "final_global_scale": 1.0,
    "scale_source": "mesh_file.baked_vertices",
    "source_mesh_scale": float(mesh_scale),
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
  raw = meta["camMat"]
  values = np.asarray(raw, dtype=np.float64)
  if values.size != 9:
    raise ValueError(
      f"Invalid camMat in {meta_path}: expected 9 values, got "
      f"shape={values.shape}, size={values.size}, value={raw!r}"
    )
  # trimesh stores vertices as float64 and FoundationPose's official readers
  # likewise keep K in float64. Matching them avoids a torch matmul dtype error
  # in compute_crop_window_tf_batch during registration.
  K = values.reshape(3, 3)
  if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or K[2, 2] == 0:
    raise ValueError(f"Invalid camMat values in {meta_path}: {K.tolist()}")
  return K


def load_sequence_intrinsics(
  meta_dir: Path,
  frames: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
  """Load per-frame K, replacing malformed metadata with the nearest valid K."""
  intrinsics: dict[str, np.ndarray] = {}
  errors: dict[str, str] = {}
  for frame in frames:
    try:
      intrinsics[frame] = load_intrinsics(meta_dir / f"{frame}.pkl")
    except (FileNotFoundError, KeyError, TypeError, ValueError, pickle.UnpicklingError) as exc:
      errors[frame] = f"{type(exc).__name__}: {exc}"

  if not intrinsics:
    examples = "; ".join(f"{frame}: {error}" for frame, error in list(errors.items())[:3])
    raise ValueError(f"No valid camMat found in {meta_dir}. {examples}")

  valid_frames = sorted(intrinsics, key=int)
  fallback_sources: dict[str, str] = {}
  for frame in errors:
    source = min(valid_frames, key=lambda candidate: abs(int(candidate) - int(frame)))
    intrinsics[frame] = intrinsics[source].copy()
    fallback_sources[frame] = source
    print(
      f"[WARN] frame={frame} has invalid camMat; using nearest valid frame={source}. "
      f"reason={errors[frame]}",
      flush=True,
    )
  return intrinsics, fallback_sources


def pose_record(
  frame: str,
  pose: np.ndarray,
  mode: str,
  mask: np.ndarray,
  depth: np.ndarray,
  intrinsics_source_frame: str,
) -> dict[str, object]:
  valid_depth = mask & (depth >= 0.001)
  return {
    "frame": frame,
    "mode": mode,
    "object_in_camera": np.asarray(pose, dtype=np.float64).reshape(4, 4).tolist(),
    "mask_pixels": int(mask.sum()),
    "valid_mask_depth_pixels": int(valid_depth.sum()),
    "translation_m": np.asarray(pose[:3, 3], dtype=np.float64).tolist(),
    "intrinsics_source_frame": intrinsics_source_frame,
    "intrinsics_fallback": intrinsics_source_frame != frame,
  }


def score_registered_pose(
  estimator: FoundationPose,
  pose: np.ndarray,
  rgb: np.ndarray,
  depth: np.ndarray,
  mask: np.ndarray,
  K: np.ndarray,
  depth_tolerance_mm: float,
) -> dict[str, float]:
  """Score a registration using visible silhouette and RGB-D agreement."""
  pose_tensor = torch.as_tensor(
    np.asarray(pose, dtype=np.float32).reshape(1, 4, 4),
    device="cuda",
    dtype=torch.float32,
  )
  with torch.inference_mode():
    _, rendered_depth, _ = nvdiffrast_render(
      K=K,
      H=rgb.shape[0],
      W=rgb.shape[1],
      ob_in_cams=pose_tensor,
      glctx=estimator.glctx,
      mesh_tensors=estimator.mesh_tensors,
    )
  rendered_depth_np = rendered_depth[0].detach().cpu().numpy()
  rendered = np.isfinite(rendered_depth_np) & (rendered_depth_np >= 0.001)
  scene_valid = np.isfinite(depth) & (depth >= 0.001)
  tolerance_m = float(depth_tolerance_mm) / 1000.0

  # Remove rendered pixels hidden by observed scene geometry, especially hand
  # occlusion, before comparing against the visible object mask.
  rendered_visible = rendered & (
    ~scene_valid | (rendered_depth_np <= depth + tolerance_m)
  )
  intersection = rendered_visible & mask
  union = rendered_visible | mask
  iou = float(intersection.sum() / max(int(union.sum()), 1))
  coverage = float(intersection.sum() / max(int(mask.sum()), 1))
  precision = float(intersection.sum() / max(int(rendered_visible.sum()), 1))

  depth_overlap = intersection & scene_valid
  if depth_overlap.any():
    residual = np.abs(rendered_depth_np[depth_overlap] - depth[depth_overlap])
    depth_median_mm = float(np.median(residual) * 1000.0)
    depth_p90_mm = float(np.quantile(residual, 0.9) * 1000.0)
    depth_score = float(np.exp(-depth_median_mm / 20.0))
  else:
    depth_median_mm = float("inf")
    depth_p90_mm = float("inf")
    depth_score = 0.0

  combined = 0.55 * iou + 0.20 * coverage + 0.10 * precision + 0.15 * depth_score
  learned_scores = getattr(estimator, "scores", None)
  learned_score = (
    float(torch.as_tensor(learned_scores[0]).detach().cpu())
    if learned_scores is not None and len(learned_scores) > 0
    else float("nan")
  )
  return {
    "combined_score": float(combined),
    "visible_mask_iou": iou,
    "mask_coverage": coverage,
    "render_precision": precision,
    "depth_median_abs_mm": depth_median_mm,
    "depth_p90_abs_mm": depth_p90_mm,
    "foundationpose_score": learned_score,
    "mask_pixels": int(mask.sum()),
    "valid_mask_depth_pixels": int((mask & scene_valid).sum()),
  }


def is_cuda_oom(exc: BaseException) -> bool:
  message = str(exc).lower()
  return isinstance(exc, torch.OutOfMemoryError) or any(
    marker in message
    for marker in ("cuda out of memory", "cuda error: 2", "cudamalloc")
  )


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
  intrinsics_by_frame, intrinsics_fallback_sources = load_sequence_intrinsics(meta_dir, selected)

  out_dir = Path(args.out_dir).expanduser().resolve()
  pose_path = out_dir / "foundationpose_poses.json"
  if pose_path.exists() and not args.overwrite:
    raise FileExistsError(f"Output exists: {pose_path}; pass --overwrite to replace it")
  overlay_dir = out_dir / "overlays"
  fp_debug_dir = out_dir / "foundationpose_debug"
  overlay_dir.mkdir(parents=True, exist_ok=True)
  fp_debug_dir.mkdir(parents=True, exist_ok=True)

  scale_json_path = Path(args.scale_json).expanduser().resolve() if args.scale_json else None
  if args.mesh_file:
    if scale_json_path is not None:
      raise ValueError("--scale_json is only valid with --anchor_npz")
    mesh_file_path = Path(args.mesh_file).expanduser().resolve()
    mesh, scale_info = load_mesh_file(mesh_file_path, args.mesh_scale)
    model_source = "mesh_file"
    model_path = mesh_file_path
  else:
    mesh_file_path = None
    model_path = Path(args.anchor_npz).expanduser().resolve()
    mesh, scale_info = load_anchor_mesh(model_path, scale_json_path)
    model_source = "anchor_npz"

  tracked_anchor_path = None
  if args.mesh_file:
    tracked_anchor_path = out_dir / "tracked_model_anchor.npz"
    np.savez_compressed(
      tracked_anchor_path,
      vertices=np.asarray(mesh.vertices, dtype=np.float32),
      faces=np.asarray(mesh.faces, dtype=np.int64),
      scale=np.asarray([1.0], dtype=np.float32),
      source_model=np.asarray(str(model_path)),
    )
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)

  score_weights_dir = (
    Path(args.score_weights_dir).expanduser().resolve()
    if args.score_weights_dir else None
  )
  refine_weights_dir = (
    Path(args.refine_weights_dir).expanduser().resolve()
    if args.refine_weights_dir else None
  )
  for label, directory in (
    ("score", score_weights_dir),
    ("refine", refine_weights_dir),
  ):
    if directory is None:
      continue
    for filename in ("config.yml", "model_best.pth"):
      if not (directory / filename).is_file():
        raise FileNotFoundError(f"Missing {label} weight file: {directory / filename}")

  scorer = ScorePredictor(
    weights_dir=str(score_weights_dir) if score_weights_dir else None,
  )
  refiner = PoseRefinePredictor(
    weights_dir=str(refine_weights_dir) if refine_weights_dir else None,
  )
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

  def frame_inputs(frame: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = imageio.imread(by_frame[frame])[..., :3]
    depth = decode_ho3d_depth(find_frame_file(depth_dir, frame))
    mask = load_mask(find_frame_file(mask_dir, frame), rgb.shape[:2])
    K = intrinsics_by_frame[frame]
    if depth.shape != rgb.shape[:2]:
      raise ValueError(f"Depth/RGB shape mismatch for {frame}: {depth.shape} vs {rgb.shape[:2]}")
    return rgb, depth, mask, K

  def tracking_depth(depth: np.ndarray) -> np.ndarray:
    return np.zeros_like(depth) if args.rgb_only else depth

  def render_overlay(rgb: np.ndarray, K: np.ndarray, pose: np.ndarray) -> np.ndarray:
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

  candidate_diagnostics: list[dict[str, object]] = []
  init_pose: np.ndarray | None = None
  init_state = None
  if args.auto_init_frames:
    requested = {f"{value:04d}" for value in args.auto_init_frames}
    candidate_frames = [frame for frame in selected if frame in requested]
    missing = sorted(requested - set(candidate_frames))
    if missing:
      print(f"[WARN] auto-init candidates not selected/present: {missing}", flush=True)
    if not candidate_frames:
      raise ValueError("No valid --auto_init_frames remain in the selected frame range")

    candidate_dir = out_dir / "init_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[str, np.ndarray, object, dict[str, float]]] = []
    for candidate_index, frame in enumerate(candidate_frames):
      try:
        rgb, depth, mask, K = frame_inputs(frame)
        pose = estimator.register(
          K=K,
          rgb=rgb,
          depth=depth,
          ob_mask=mask,
          iteration=args.est_refine_iter,
        )
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        metrics = score_registered_pose(
          estimator,
          pose,
          rgb,
          depth,
          mask,
          K,
          args.auto_init_depth_tolerance_mm,
        )
        state = estimator.pose_last.detach().clone()
        candidates.append((frame, pose.copy(), state, metrics))
        diagnostic = {
          "frame": frame,
          "status": "valid",
          "intrinsics_source_frame": intrinsics_fallback_sources.get(frame, frame),
          **metrics,
          "translation_m": pose[:3, 3].tolist(),
        }
        candidate_diagnostics.append(diagnostic)
        np.savetxt(candidate_dir / f"{frame}.txt", pose)
        imageio.imwrite(candidate_dir / f"{frame}.jpg", render_overlay(rgb, K, pose))
        print(
          f"[auto-init {candidate_index + 1}/{len(candidate_frames)}] frame={frame} "
          f"score={metrics['combined_score']:.4f} iou={metrics['visible_mask_iou']:.4f} "
          f"depth_med={metrics['depth_median_abs_mm']:.2f}mm",
          flush=True,
        )
      except Exception as exc:
        diagnostic = {
          "frame": frame,
          "status": "failed",
          "error": f"{type(exc).__name__}: {exc}",
        }
        candidate_diagnostics.append(diagnostic)
        print(f"[WARN] auto-init frame={frame} failed: {type(exc).__name__}: {exc}", flush=True)
        if is_cuda_oom(exc):
          (out_dir / "auto_init_scores.json").write_text(
            json.dumps({
              "selected_frame": None,
              "depth_tolerance_mm": args.auto_init_depth_tolerance_mm,
              "score_tie_margin": args.auto_init_score_tie_margin,
              "highest_candidate_score": None,
              "candidates": candidate_diagnostics,
              "aborted_reason": "cuda_out_of_memory",
            }, indent=2),
            encoding="utf-8",
          )
          raise

    auto_init_payload = {
      "selected_frame": None,
      "depth_tolerance_mm": args.auto_init_depth_tolerance_mm,
      "score_tie_margin": args.auto_init_score_tie_margin,
      "highest_candidate_score": None,
      "candidates": candidate_diagnostics,
    }
    if not candidates:
      (out_dir / "auto_init_scores.json").write_text(
        json.dumps(auto_init_payload, indent=2),
        encoding="utf-8",
      )
      raise RuntimeError("Every automatic initialization candidate failed")
    highest_score = max(item[3]["combined_score"] for item in candidates)
    near_best = [
      item for item in candidates
      if item[3]["combined_score"] >= highest_score - args.auto_init_score_tie_margin
    ]
    temporal_center = (int(selected[0]) + int(selected[-1])) / 2.0
    init_frame, init_pose, init_state, best_metrics = min(
      near_best,
      key=lambda item: abs(int(item[0]) - temporal_center),
    )
    print(
      f"Selected auto-init frame={init_frame} score={best_metrics['combined_score']:.4f} "
      f"iou={best_metrics['visible_mask_iou']:.4f} "
      f"depth_med={best_metrics['depth_median_abs_mm']:.2f}mm",
      flush=True,
    )
    auto_init_payload.update({
      "selected_frame": init_frame,
      "highest_candidate_score": highest_score,
    })
    (out_dir / "auto_init_scores.json").write_text(
      json.dumps(auto_init_payload, indent=2),
      encoding="utf-8",
    )
  else:
    rgb, depth, mask, K = frame_inputs(init_frame)
    init_pose = estimator.register(
      K=K,
      rgb=rgb,
      depth=depth,
      ob_mask=mask,
      iteration=args.est_refine_iter,
    )
    init_pose = np.asarray(init_pose, dtype=np.float64).reshape(4, 4)
    init_state = estimator.pose_last.detach().clone()

  if init_pose is None or init_state is None:
    raise RuntimeError("Failed to establish an initialization pose")

  rows: dict[str, dict[str, object]] = {}

  def write_payload() -> None:
    ordered_rows = {frame: rows[frame] for frame in selected if frame in rows}
    payload = {
      "source": "foundationpose_ho3d_rgbd_tracking",
      "sequence": args.sequence,
      "coordinate_system": "opencv_camera",
      "pose_convention": "object_model_to_camera",
      "uses_gt_object_pose": False,
      "foundationpose_input_mode": (
        "rgbd_register_rgb_only_track" if args.rgb_only else "rgbd"
      ),
      "score_weights_dir": str(score_weights_dir) if score_weights_dir else "default",
      "refine_weights_dir": str(refine_weights_dir) if refine_weights_dir else "default",
      "init_frame": init_frame,
      "auto_init": bool(args.auto_init_frames),
      "auto_init_candidates": candidate_diagnostics,
      "bidirectional": bool(args.bidirectional),
      "model_source": model_source,
      "model_path": str(model_path),
      "anchor_npz": str(Path(args.anchor_npz).expanduser().resolve()) if args.anchor_npz else None,
      "mesh_file": str(mesh_file_path) if mesh_file_path else None,
      "tracked_model_anchor": str(tracked_anchor_path) if tracked_anchor_path else None,
      "scale_json": str(scale_json_path) if scale_json_path else None,
      "num_intrinsics_fallback_frames": len(intrinsics_fallback_sources),
      "intrinsics_fallback_by_frame": intrinsics_fallback_sources,
      **scale_info,
      "frames": ordered_rows,
      "by_frame": ordered_rows,
    }
    tmp_path = pose_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, pose_path)

  def save_result(frame: str, pose: np.ndarray, mode: str) -> None:
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(pose).all():
      raise RuntimeError(f"FoundationPose returned a non-finite pose for frame {frame}")
    rgb, depth, mask, K = frame_inputs(frame)
    rows[frame] = pose_record(
      frame,
      pose,
      mode,
      mask,
      depth,
      intrinsics_fallback_sources.get(frame, frame),
    )
    np.savetxt(out_dir / f"{frame}.txt", pose)
    if args.save_overlays:
      imageio.imwrite(overlay_dir / f"{frame}.jpg", render_overlay(rgb, K, pose))
    write_payload()
    print(
      f"[{len(rows)}/{len(selected)}] frame={frame} mode={mode} t={pose[:3, 3].tolist()}",
      flush=True,
    )

  save_result(init_frame, init_pose, "register_auto" if args.auto_init_frames else "register")
  init_index = selected.index(init_frame)

  estimator.pose_last = init_state.detach().clone()
  for frame in selected[init_index + 1:]:
    rgb, depth, _, K = frame_inputs(frame)
    pose = estimator.track_one(
      rgb=rgb,
      depth=tracking_depth(depth),
      K=K,
      iteration=args.track_refine_iter,
    )
    save_result(frame, pose, "track_forward")

  if args.bidirectional:
    estimator.pose_last = init_state.detach().clone()
    for frame in reversed(selected[:init_index]):
      rgb, depth, _, K = frame_inputs(frame)
      pose = estimator.track_one(
        rgb=rgb,
        depth=tracking_depth(depth),
        K=K,
        iteration=args.track_refine_iter,
      )
      save_result(frame, pose, "track_backward")

  print(json.dumps({
    "num_frames": len(rows),
    "pose_json": str(pose_path),
    "selected_init_frame": init_frame,
    "bidirectional": bool(args.bidirectional),
    **scale_info,
  }, indent=2))


if __name__ == "__main__":
  main()
