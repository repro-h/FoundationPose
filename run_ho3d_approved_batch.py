#!/usr/bin/env python3
"""Run full-sequence bidirectional FoundationPose for approved HO3D sequences."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Resolve one MV-SAM3D GLB per approved HO3D sequence, choose distributed "
      "registration candidates, and run full-sequence bidirectional FoundationPose."
    )
  )
  parser.add_argument("--approved_sequences_json", required=True)
  parser.add_argument("--mv_status_json", default=None)
  parser.add_argument("--mv_visualization_root", required=True)
  parser.add_argument("--ho3d_root", required=True)
  parser.add_argument("--out_root", required=True)
  parser.add_argument("--foundationpose_python", required=True)
  parser.add_argument(
    "--runner_path",
    default=str(Path(__file__).resolve().with_name("run_ho3d_sequence.py")),
  )
  parser.add_argument("--train_manifest_jsonl", default=None)
  parser.add_argument("--val_manifest_jsonl", default=None)
  parser.add_argument("--test_manifest_jsonl", default=None)
  parser.add_argument("--sequences", nargs="+", default=None)
  parser.add_argument("--cuda_visible_devices", default="0")
  parser.add_argument("--num_init_candidates", type=int, default=8)
  parser.add_argument("--candidate_scan_stride", type=int, default=5)
  parser.add_argument("--min_mask_ratio", type=float, default=0.005)
  parser.add_argument("--border_margin_px", type=int, default=8)
  parser.add_argument("--border_penalty", type=float, default=2.0)
  parser.add_argument("--auto_init_depth_tolerance_mm", type=float, default=12.0)
  parser.add_argument("--auto_init_score_tie_margin", type=float, default=0.02)
  parser.add_argument("--est_refine_iter", type=int, default=5)
  parser.add_argument("--track_refine_iter", type=int, default=2)
  parser.add_argument("--score_weights_dir", default=None)
  parser.add_argument("--refine_weights_dir", default=None)
  parser.add_argument("--rgb_only", action="store_true")
  parser.add_argument("--save_overlays", action="store_true")
  parser.add_argument("--min_pose_coverage", type=float, default=1.0)
  parser.add_argument("--status_json", default=None)
  parser.add_argument("--summary_json", default=None)
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--fail_fast", action="store_true")
  parser.add_argument("--dry_run", action="store_true")
  return parser.parse_args()


def load_json(path: str | Path) -> Any:
  return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  os.replace(temporary, path)


def _sequence_from_item(item: Any) -> tuple[str, str | None] | None:
  if isinstance(item, str):
    return item, None
  if not isinstance(item, dict):
    return None
  sequence = item.get("sequence", item.get("seq_name", item.get("name")))
  if sequence is None:
    return None
  split = item.get("split", item.get("subset"))
  return str(sequence), str(split) if split else None


def approved_sequence_entries(payload: Any) -> dict[str, str | None]:
  """Accept the approved-list variants used by hand-uni tools."""
  if isinstance(payload, list):
    values = payload
  elif isinstance(payload, dict):
    values = payload.get("sequences", payload.get("selected_sequences"))
    if values is None:
      values = []
      for split in ("train", "val", "test"):
        split_values = payload.get(split)
        if isinstance(split_values, list):
          values.extend(
            {"sequence": item, "split": split} if isinstance(item, str) else item
            for item in split_values
          )
  else:
    raise TypeError("approved sequence JSON must be a list or object")

  entries: dict[str, str | None] = {}
  for item in values or []:
    parsed = _sequence_from_item(item)
    if parsed is None:
      continue
    sequence, split = parsed
    entries[sequence] = split or entries.get(sequence)
  if not entries:
    raise ValueError("No approved sequences found")
  return entries


def sequence_names_from_manifest(path: str | Path) -> set[str]:
  sequences: set[str] = set()
  with Path(path).expanduser().open("r", encoding="utf-8") as handle:
    for line in handle:
      if not line.strip():
        continue
      row = json.loads(line)
      sequence = row.get("sequence", row.get("seq_name"))
      if sequence is not None:
        sequences.add(str(sequence))
  return sequences


def resolve_splits(entries: dict[str, str | None], manifests: dict[str, str | None]) -> dict[str, str]:
  by_split = {
    split: sequence_names_from_manifest(path)
    for split, path in manifests.items()
    if path
  }
  resolved: dict[str, str] = {}
  for sequence, embedded_split in entries.items():
    if embedded_split:
      resolved[sequence] = embedded_split
      continue
    matches = [split for split, names in by_split.items() if sequence in names]
    if len(matches) > 1:
      raise ValueError(f"Sequence {sequence} appears in multiple split manifests: {matches}")
    resolved[sequence] = matches[0] if matches else "unknown"
  return resolved


def resolve_mv_glb(
  sequence: str,
  status_payload: Any,
  visualization_root: Path,
) -> tuple[Path | None, str]:
  if isinstance(status_payload, dict):
    job = status_payload.get("jobs", {}).get(sequence, {})
    output = job.get("sam3d_output") if isinstance(job, dict) else None
    if output:
      candidate = Path(output).expanduser() / "result.glb"
      if candidate.is_file():
        return candidate.resolve(), "status_json"

  candidates = list((visualization_root / sequence).glob("**/result.glb"))
  candidates = [path for path in candidates if path.is_file()]
  if not candidates:
    return None, "missing"
  return max(candidates, key=lambda path: path.stat().st_mtime).resolve(), "visualization_fallback"


def sequence_rgb_frames(sequence_dir: Path) -> list[str]:
  rgb_dir = sequence_dir / "rgb"
  paths = list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png"))
  frames = sorted({path.stem for path in paths if path.stem.isdigit()}, key=int)
  if not frames:
    raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
  return frames


def find_mask_file(mask_dir: Path, frame: str) -> Path | None:
  for candidate in (mask_dir / f"{frame}.png", mask_dir / f"{frame}_mask.png"):
    if candidate.is_file():
      return candidate
  return None


def mask_candidate_metrics(mask_path: Path, border_margin_px: int, border_penalty: float) -> dict[str, float]:
  import cv2

  mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
  if mask is None:
    raise ValueError(f"Unreadable mask: {mask_path}")
  foreground = mask > 127
  area_ratio = float(foreground.mean())
  margin = max(1, min(int(border_margin_px), min(mask.shape) // 2))
  border = foreground.copy()
  border[margin:-margin, margin:-margin] = False
  border_ratio = float(border.sum() / max(int(foreground.sum()), 1))
  score = math.log(max(area_ratio, 1e-8)) - float(border_penalty) * border_ratio
  return {"area_ratio": area_ratio, "border_ratio": border_ratio, "quality": score}


def choose_distributed_candidates(
  candidates: list[dict[str, Any]],
  num_candidates: int,
) -> list[dict[str, Any]]:
  if num_candidates < 1:
    raise ValueError("num_candidates must be positive")
  if len(candidates) <= num_candidates:
    return sorted(candidates, key=lambda row: int(row["frame"]))

  ordered = sorted(candidates, key=lambda row: int(row["frame"]))
  first = int(ordered[0]["frame"])
  last = int(ordered[-1]["frame"])
  span = max(last - first + 1, 1)
  selected: list[dict[str, Any]] = []
  selected_frames: set[str] = set()
  for bin_index in range(num_candidates):
    low = first + span * bin_index / num_candidates
    high = first + span * (bin_index + 1) / num_candidates
    center = (low + high) / 2.0
    in_bin = [
      row for row in ordered
      if low <= int(row["frame"]) < high or (bin_index == num_candidates - 1 and int(row["frame"]) == last)
    ]
    if not in_bin:
      continue
    best = max(in_bin, key=lambda row: (float(row["quality"]), -abs(int(row["frame"]) - center)))
    selected.append(best)
    selected_frames.add(str(best["frame"]))

  remaining = [row for row in ordered if str(row["frame"]) not in selected_frames]
  remaining.sort(key=lambda row: float(row["quality"]), reverse=True)
  selected.extend(remaining[:max(0, num_candidates - len(selected))])
  return sorted(selected[:num_candidates], key=lambda row: int(row["frame"]))


def select_init_candidates(
  sequence_dir: Path,
  frames: list[str],
  scan_stride: int,
  num_candidates: int,
  min_mask_ratio: float,
  border_margin_px: int,
  border_penalty: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  if scan_stride < 1:
    raise ValueError("candidate_scan_stride must be positive")
  mask_dir = sequence_dir / "obj_mask_white"
  eligible: list[dict[str, Any]] = []
  rejected: list[dict[str, Any]] = []
  sampled = frames[::scan_stride]
  if frames[-1] not in sampled:
    sampled.append(frames[-1])
  for frame in sampled:
    mask_path = find_mask_file(mask_dir, frame)
    if mask_path is None:
      rejected.append({"frame": frame, "reason": "missing_mask"})
      continue
    try:
      metrics = mask_candidate_metrics(mask_path, border_margin_px, border_penalty)
    except Exception as exc:
      rejected.append({"frame": frame, "reason": f"{type(exc).__name__}: {exc}"})
      continue
    row = {"frame": frame, "mask_path": str(mask_path.resolve()), **metrics}
    if metrics["area_ratio"] < min_mask_ratio:
      rejected.append({**row, "reason": "mask_too_small"})
      continue
    eligible.append(row)
  if not eligible:
    raise RuntimeError(f"No usable registration masks in {mask_dir}")
  return choose_distributed_candidates(eligible, num_candidates), rejected


def validate_pose_json(
  pose_path: Path,
  expected_frames: Iterable[str],
  expected_mesh: Path,
  min_coverage: float,
) -> tuple[bool, dict[str, Any]]:
  if not pose_path.is_file():
    return False, {"reason": "missing_pose_json"}
  try:
    payload = load_json(pose_path)
  except Exception as exc:
    return False, {"reason": f"invalid_pose_json:{type(exc).__name__}:{exc}"}
  frames = payload.get("by_frame", payload.get("frames", {}))
  expected = list(expected_frames)
  present = set(frames) if isinstance(frames, dict) else set()
  coverage = len(set(expected) & present) / max(len(expected), 1)
  model_path = payload.get("model_path", payload.get("mesh_file"))
  same_mesh = False
  if model_path:
    try:
      same_mesh = Path(model_path).expanduser().resolve() == expected_mesh.resolve()
    except OSError:
      same_mesh = False
  diagnostics = {
    "coverage": coverage,
    "num_expected_frames": len(expected),
    "num_pose_frames": len(present),
    "model_path": model_path,
    "expected_mesh": str(expected_mesh),
    "bidirectional": bool(payload.get("bidirectional")),
    "uses_gt_object_pose": payload.get("uses_gt_object_pose"),
  }
  valid = (
    coverage >= min_coverage
    and payload.get("model_source") == "mesh_file"
    and same_mesh
    and bool(payload.get("bidirectional"))
    and payload.get("uses_gt_object_pose") is False
  )
  diagnostics["reason"] = "valid" if valid else "metadata_or_coverage_mismatch"
  return valid, diagnostics


def command_for_job(
  args: argparse.Namespace,
  sequence: str,
  frames: list[str],
  candidates: list[dict[str, Any]],
  mesh_path: Path,
  out_dir: Path,
  overwrite_output: bool,
) -> list[str]:
  command = [
    str(Path(args.foundationpose_python).expanduser()),
    str(Path(args.runner_path).expanduser()),
    "--ho3d_root", str(Path(args.ho3d_root).expanduser()),
    "--sequence", sequence,
    "--mesh_file", str(mesh_path),
    "--mesh_scale", "1.0",
    "--out_dir", str(out_dir),
    "--start_frame", str(int(frames[0])),
    "--end_frame", str(int(frames[-1])),
    "--frame_stride", "1",
    "--auto_init_frames", *[str(int(row["frame"])) for row in candidates],
    "--bidirectional",
    "--auto_init_depth_tolerance_mm", str(args.auto_init_depth_tolerance_mm),
    "--auto_init_score_tie_margin", str(args.auto_init_score_tie_margin),
    "--est_refine_iter", str(args.est_refine_iter),
    "--track_refine_iter", str(args.track_refine_iter),
  ]
  if args.score_weights_dir:
    command.extend(["--score_weights_dir", args.score_weights_dir])
  if args.refine_weights_dir:
    command.extend(["--refine_weights_dir", args.refine_weights_dir])
  if args.rgb_only:
    command.append("--rgb_only")
  if args.save_overlays:
    command.append("--save_overlays")
  if overwrite_output:
    command.append("--overwrite")
  return command


def clean_subprocess_environment(python_path: Path, cuda_visible_devices: str) -> dict[str, str]:
  environment = os.environ.copy()
  for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "LD_LIBRARY_PATH"):
    environment.pop(key, None)
  conda_prefix = python_path.parent.parent
  environment["CONDA_PREFIX"] = str(conda_prefix)
  environment["PATH"] = str(conda_prefix / "bin") + os.pathsep + environment.get("PATH", "")
  environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
  return environment


def summarize_failure_log(log_path: Path, max_lines: int = 40) -> dict[str, Any]:
  """Extract the final traceback or error tail from a failed worker log."""
  if not log_path.is_file():
    return {"error_summary": "worker log is missing", "error_tail": []}
  lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
  markers = (
    "Traceback (most recent call last):",
    "RuntimeError:",
    "ValueError:",
    "FileNotFoundError:",
    "torch.OutOfMemoryError:",
    "CUDA error",
  )
  traceback_indices = [
    index for index, line in enumerate(lines)
    if "Traceback (most recent call last):" in line
  ]
  marker_indices = [
    index for index, line in enumerate(lines)
    if any(marker in line for marker in markers)
  ]
  if traceback_indices:
    start = traceback_indices[-1]
  elif marker_indices:
    start = marker_indices[-1]
  else:
    start = max(0, len(lines) - max_lines)
  tail = lines[start:]
  if len(tail) > max_lines:
    tail = tail[-max_lines:]
  summary = next((line.strip() for line in reversed(tail) if line.strip()), "unknown worker error")
  return {"error_summary": summary, "error_tail": tail}


def main() -> None:
  args = parse_args()
  if not 0 < args.min_pose_coverage <= 1:
    raise ValueError("--min_pose_coverage must be in (0, 1]")
  approved = approved_sequence_entries(load_json(args.approved_sequences_json))
  if args.sequences:
    requested = set(args.sequences)
    unknown = sorted(requested - set(approved))
    if unknown:
      raise ValueError(f"Requested sequences are not approved: {unknown}")
    approved = {sequence: split for sequence, split in approved.items() if sequence in requested}

  split_by_sequence = resolve_splits(approved, {
    "train": args.train_manifest_jsonl,
    "val": args.val_manifest_jsonl,
    "test": args.test_manifest_jsonl,
  })
  status_payload = load_json(args.mv_status_json) if args.mv_status_json else {}
  visualization_root = Path(args.mv_visualization_root).expanduser().resolve()
  ho3d_root = Path(args.ho3d_root).expanduser().resolve()
  out_root = Path(args.out_root).expanduser().resolve()
  status_path = Path(args.status_json).expanduser() if args.status_json else out_root / "status.json"
  summary_path = Path(args.summary_json).expanduser() if args.summary_json else out_root / "summary.json"
  python_path = Path(args.foundationpose_python).expanduser().resolve()
  if not args.dry_run and not python_path.is_file():
    raise FileNotFoundError(python_path)

  state: dict[str, Any] = {
    "source": "foundationpose_ho3d_approved_batch",
    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "jobs": {},
  }
  if status_path.is_file():
    previous = load_json(status_path)
    if isinstance(previous, dict):
      state["jobs"].update(previous.get("jobs", {}))

  sequences = sorted(approved)
  for index, sequence in enumerate(sequences, start=1):
    split = split_by_sequence[sequence]
    print(f"[{index}/{len(sequences)}] {split}/{sequence}", flush=True)
    job: dict[str, Any] = {"sequence": sequence, "split": split, "status": "preparing"}
    state["jobs"][sequence] = job
    try:
      mesh_path, mesh_source = resolve_mv_glb(sequence, status_payload, visualization_root)
      if mesh_path is None:
        raise FileNotFoundError(f"No MV-SAM3D result.glb found for {sequence}")
      sequence_dir = ho3d_root / sequence
      frames = sequence_rgb_frames(sequence_dir)
      candidates, rejected = select_init_candidates(
        sequence_dir,
        frames,
        args.candidate_scan_stride,
        args.num_init_candidates,
        args.min_mask_ratio,
        args.border_margin_px,
        args.border_penalty,
      )
      out_dir = out_root / split / sequence
      pose_path = out_dir / "foundationpose_poses.json"
      valid, validation = validate_pose_json(
        pose_path, frames, mesh_path, args.min_pose_coverage,
      )
      job.update({
        "mv_glb": str(mesh_path),
        "mv_glb_source": mesh_source,
        "out_dir": str(out_dir),
        "pose_json": str(pose_path),
        "num_rgb_frames": len(frames),
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "init_candidates": candidates,
        "num_rejected_candidates": len(rejected),
        "validation_before": validation,
      })
      if valid and not args.force:
        job["status"] = "cached"
        print(f"  cached: {pose_path}", flush=True)
      else:
        command = command_for_job(
          args,
          sequence,
          frames,
          candidates,
          mesh_path,
          out_dir,
          overwrite_output=args.force or pose_path.exists(),
        )
        log_path = out_root / "logs" / f"{split}_{sequence}.log"
        job.update({"status": "dry_run" if args.dry_run else "running", "command": command, "log": str(log_path)})
        print("  candidates:", [row["frame"] for row in candidates], flush=True)
        print("  command:", " ".join(command), flush=True)
        if not args.dry_run:
          out_dir.mkdir(parents=True, exist_ok=True)
          log_path.parent.mkdir(parents=True, exist_ok=True)
          environment = clean_subprocess_environment(python_path, args.cuda_visible_devices)
          with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("COMMAND\n" + " ".join(command) + "\n\n")
            log_handle.flush()
            try:
              subprocess.run(
                command,
                cwd=str(Path(args.runner_path).expanduser().resolve().parent),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
              )
            except subprocess.CalledProcessError:
              log_handle.flush()
              job.update(summarize_failure_log(log_path))
              raise
          valid, validation = validate_pose_json(
            pose_path, frames, mesh_path, args.min_pose_coverage,
          )
          job["validation_after"] = validation
          if not valid:
            raise RuntimeError(f"Output validation failed: {validation}")
          job["status"] = "done"
          print(f"  done: {pose_path}", flush=True)
    except Exception as exc:
      job["status"] = "failed"
      job["error"] = f"{type(exc).__name__}: {exc}"
      print(f"  failed: {job['error']}", flush=True)
      if args.fail_fast:
        state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_json(status_path, state)
        raise
    finally:
      state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
      atomic_write_json(status_path, state)

  counts: dict[str, int] = {}
  for sequence in sequences:
    job = state["jobs"].get(sequence, {})
    status = str(job.get("status", "unknown"))
    counts[status] = counts.get(status, 0) + 1
  summary = {
    "num_requested": len(sequences),
    "status_counts": counts,
    "missing_mv_sequences": [
      sequence for sequence in sequences
      if state["jobs"].get(sequence, {}).get("status") == "failed"
      and "No MV-SAM3D" in state["jobs"].get(sequence, {}).get("error", "")
    ],
    "failures": {
      sequence: {
        "split": state["jobs"].get(sequence, {}).get("split"),
        "error": state["jobs"].get(sequence, {}).get("error"),
        "error_summary": state["jobs"].get(sequence, {}).get("error_summary"),
        "log": state["jobs"].get(sequence, {}).get("log"),
      }
      for sequence in sequences
      if state["jobs"].get(sequence, {}).get("status") == "failed"
    },
    "status_json": str(status_path.resolve()),
  }
  atomic_write_json(summary_path, summary)
  print(json.dumps(summary, indent=2), flush=True)
  if counts.get("failed", 0):
    sys.exit(1)


if __name__ == "__main__":
  main()
