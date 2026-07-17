#!/usr/bin/env python3
"""Render sampled HO3D FoundationPose results without rerunning tracking."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from Utils import dr, make_mesh_tensors, nvdiffrast_render
from run_ho3d_sequence import (
  decode_ho3d_depth,
  find_frame_file,
  load_mask,
  load_mesh_file,
  load_sequence_intrinsics,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Render existing FoundationPose poses, compare the visible rendered mesh "
      "against HO3D object masks, and save sampled QA overlays."
    )
  )
  parser.add_argument("--ho3d_root", required=True)
  parser.add_argument("--sequence", required=True)
  parser.add_argument("--pose_json", required=True)
  parser.add_argument("--out_dir", required=True)
  parser.add_argument("--mesh_file", default=None)
  parser.add_argument("--mesh_scale", type=float, default=1.0)
  parser.add_argument("--frame_stride", type=int, default=20)
  parser.add_argument("--frames", type=int, nargs="+", default=None)
  parser.add_argument("--depth_tolerance_mm", type=float, default=12.0)
  parser.add_argument("--overlay_alpha", type=float, default=0.35)
  parser.add_argument("--worst_k", type=int, default=20)
  return parser.parse_args()


def frame_keys(payload: dict[str, object]) -> list[str]:
  rows = payload.get("by_frame", payload.get("frames", {}))
  if not isinstance(rows, dict):
    raise TypeError("pose JSON must contain a by_frame mapping")
  return sorted((str(key).zfill(4) for key in rows), key=int)


def choose_frames(
  available: list[str],
  stride: int,
  requested: list[int] | None,
  init_frame: object,
) -> list[str]:
  available_set = set(available)
  if requested:
    selected = [f"{value:04d}" for value in requested]
    missing = [frame for frame in selected if frame not in available_set]
    if missing:
      raise KeyError(f"Frames are absent from pose JSON: {missing}")
    return sorted(set(selected), key=int)
  if stride < 1:
    raise ValueError("--frame_stride must be positive")
  selected = available[::stride]
  for frame in (available[0], str(init_frame).zfill(4), available[-1]):
    if frame in available_set:
      selected.append(frame)
  return sorted(set(selected), key=int)


def pose_for_frame(payload: dict[str, object], frame: str) -> np.ndarray:
  rows = payload.get("by_frame", payload.get("frames", {}))
  row = rows[frame]
  if isinstance(row, dict):
    value = row.get("object_in_camera", row.get("pose", row.get("transform")))
  else:
    value = row
  pose = np.asarray(value, dtype=np.float32)
  if pose.size != 16 or not np.isfinite(pose).all():
    raise ValueError(f"Invalid object pose for frame {frame}")
  return pose.reshape(4, 4)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float | None:
  union = first | second
  if not union.any():
    return None
  return float(np.count_nonzero(first & second) / np.count_nonzero(union))


def tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
  if not mask.any():
    return
  values = image[mask].astype(np.float32)
  image[mask] = np.clip(
    values * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha,
    0,
    255,
  ).astype(np.uint8)


def draw_contour(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
  contours, _ = cv2.findContours(
    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
  )
  cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)


def put_label(image: np.ndarray, lines: list[str]) -> None:
  for index, line in enumerate(lines):
    origin = (8, 24 + index * 21)
    cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
  args = parse_args()
  pose_path = Path(args.pose_json).expanduser().resolve()
  payload = json.loads(pose_path.read_text(encoding="utf-8"))
  available = frame_keys(payload)
  selected = choose_frames(
    available,
    args.frame_stride,
    args.frames,
    payload.get("selected_init_frame", payload.get("init_frame")),
  )

  sequence_dir = Path(args.ho3d_root).expanduser().resolve() / args.sequence
  rgb_dir = sequence_dir / "rgb"
  depth_dir = sequence_dir / "depth"
  mask_dir = sequence_dir / "obj_mask_white"
  meta_dir = sequence_dir / "meta"
  intrinsics, fallback_sources = load_sequence_intrinsics(meta_dir, selected)

  model_value = args.mesh_file or payload.get("model_path") or payload.get("mesh_file")
  if not model_value:
    raise ValueError("No mesh path in arguments or pose JSON")
  model_path = Path(str(model_value)).expanduser().resolve()
  mesh, _ = load_mesh_file(model_path, args.mesh_scale)
  mesh_tensors = make_mesh_tensors(mesh, device="cuda")
  glctx = dr.RasterizeCudaContext()

  output_dir = Path(args.out_dir).expanduser().resolve()
  preview_dir = output_dir / "previews"
  preview_dir.mkdir(parents=True, exist_ok=True)
  tolerance_m = float(args.depth_tolerance_mm) / 1000.0
  rows: list[dict[str, object]] = []

  for index, frame in enumerate(selected, start=1):
    rgb_path = find_frame_file(rgb_dir, frame)
    rgb = imageio.imread(rgb_path)[..., :3]
    depth = decode_ho3d_depth(find_frame_file(depth_dir, frame))
    mask = load_mask(find_frame_file(mask_dir, frame), rgb.shape[:2])
    pose = pose_for_frame(payload, frame)
    pose_tensor = torch.as_tensor(pose[None], dtype=torch.float32, device="cuda")
    with torch.inference_mode():
      _, rendered_depth, _ = nvdiffrast_render(
        K=intrinsics[frame],
        H=rgb.shape[0],
        W=rgb.shape[1],
        ob_in_cams=pose_tensor,
        glctx=glctx,
        mesh_tensors=mesh_tensors,
      )
    rendered_depth_np = rendered_depth[0].detach().cpu().numpy()
    rendered = np.isfinite(rendered_depth_np) & (rendered_depth_np >= 0.001)
    scene_valid = np.isfinite(depth) & (depth >= 0.001)
    visible = rendered & (~scene_valid | (rendered_depth_np <= depth + tolerance_m))
    iou = mask_iou(visible, mask)
    containment = (
      float(np.count_nonzero(visible & mask) / np.count_nonzero(visible))
      if visible.any() else None
    )

    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tint(canvas, visible, (255, 0, 255), args.overlay_alpha)
    draw_contour(canvas, mask, (255, 255, 0))
    draw_contour(canvas, visible, (255, 0, 255))
    iou_text = "n/a" if iou is None else f"{iou:.3f}"
    containment_text = "n/a" if containment is None else f"{containment:.3f}"
    put_label(canvas, [
      "cyan=object mask | magenta=visible rendered mesh",
      f"{args.sequence} {frame} IoU={iou_text} containment={containment_text}",
    ])
    preview_path = preview_dir / f"{frame}.jpg"
    cv2.imwrite(str(preview_path), canvas)
    rows.append({
      "frame": frame,
      "visible_iou": iou,
      "visible_containment": containment,
      "mask_pixels": int(mask.sum()),
      "rendered_visible_pixels": int(visible.sum()),
      "intrinsics_source_frame": fallback_sources.get(frame, frame),
      "preview": str(preview_path),
    })
    print(f"[{index}/{len(selected)}] {frame} IoU={iou_text} containment={containment_text}", flush=True)

  valid_iou = np.asarray(
    [row["visible_iou"] for row in rows if row["visible_iou"] is not None],
    dtype=np.float64,
  )
  worst = sorted(
    (row for row in rows if row["visible_iou"] is not None),
    key=lambda row: float(row["visible_iou"]),
  )[:max(0, args.worst_k)]
  summary = {
    "source": "foundationpose_ho3d_offline_pose_qa",
    "sequence": args.sequence,
    "pose_json": str(pose_path),
    "mesh_file": str(model_path),
    "num_pose_frames": len(available),
    "num_rendered_frames": len(rows),
    "frame_stride": args.frame_stride,
    "depth_tolerance_mm": args.depth_tolerance_mm,
    "visible_iou": {
      "count": int(valid_iou.size),
      "min": float(valid_iou.min()) if valid_iou.size else None,
      "median": float(np.median(valid_iou)) if valid_iou.size else None,
      "p90": float(np.quantile(valid_iou, 0.9)) if valid_iou.size else None,
      "max": float(valid_iou.max()) if valid_iou.size else None,
    },
    "worst_frames": worst,
    "frames": rows,
  }
  summary_path = output_dir / "summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps({
    "summary": str(summary_path),
    "num_rendered_frames": len(rows),
    "visible_iou": summary["visible_iou"],
    "worst_frames": [row["frame"] for row in worst],
  }, indent=2))


if __name__ == "__main__":
  main()
