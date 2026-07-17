import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_ho3d_approved_batch import (
  approved_sequence_entries,
  choose_distributed_candidates,
  load_tracking_mesh_overrides,
  resolve_mv_glb,
  resolve_mv_mesh_scale,
  summarize_failure_log,
  validate_pose_json,
)


class ApprovedBatchTests(unittest.TestCase):
  def test_summarize_failure_log_uses_final_error(self):
    with tempfile.TemporaryDirectory() as temporary:
      path = Path(temporary) / "worker.log"
      path.write_text(
        "[WARN] auto-init frame=0001 failed: ValueError: bad depth\n"
        "old output\nTraceback (most recent call last):\nold failure\n"
        "progress\nTraceback (most recent call last):\nlast detail\n"
        "RuntimeError: final failure\n",
        encoding="utf-8",
      )
      result = summarize_failure_log(path)
      self.assertEqual(result["error_summary"], "RuntimeError: final failure")
      self.assertEqual(result["error_tail"][0], "Traceback (most recent call last):")
      self.assertEqual(
        result["candidate_failure_summaries"],
        ["[WARN] auto-init frame=0001 failed: ValueError: bad depth"],
      )

  def test_approved_sequence_variants(self):
    self.assertEqual(
      approved_sequence_entries({
        "sequences": [
          {"sequence": "MC4", "split": "train"},
          "MC6",
        ]
      }),
      {"MC4": "train", "MC6": None},
    )
    self.assertEqual(
      approved_sequence_entries({"train": ["MC4"], "val": ["MC6"]}),
      {"MC4": "train", "MC6": "val"},
    )

  def test_candidates_cover_timeline(self):
    candidates = [
      {"frame": f"{frame:04d}", "quality": float(frame % 13)}
      for frame in range(0, 101, 5)
    ]
    selected = choose_distributed_candidates(candidates, 5)
    values = [int(row["frame"]) for row in selected]
    self.assertEqual(len(values), 5)
    self.assertLess(values[0], 25)
    self.assertGreater(values[-1], 75)
    self.assertEqual(values, sorted(values))

  def test_mv_glb_status_and_fallback(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      status_output = root / "status_output"
      status_output.mkdir()
      status_glb = status_output / "result.glb"
      status_glb.touch()
      path, source = resolve_mv_glb(
        "MC4", {"jobs": {"MC4": {"sam3d_output": str(status_output)}}}, root,
      )
      self.assertEqual(path, status_glb.resolve())
      self.assertEqual(source, "status_json")

      fallback = root / "MC6" / "object" / "run" / "result.glb"
      fallback.parent.mkdir(parents=True)
      fallback.touch()
      path, source = resolve_mv_glb("MC6", {}, root)
      self.assertEqual(path, fallback.resolve())
      self.assertEqual(source, "visualization_fallback")

  def test_mv_mesh_scale_comes_from_result_params(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      mesh = root / "result.glb"
      mesh.touch()
      np.savez(root / "params.npz", scale=np.asarray([0.19, 0.19, 0.19]))
      scale, params = resolve_mv_mesh_scale(mesh)
      self.assertAlmostEqual(scale, 0.19)
      self.assertEqual(params, (root / "params.npz").resolve())

  def test_tracking_mesh_overrides(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      mesh = root / "tracking.ply"
      mesh.touch()
      path = root / "overrides.json"
      path.write_text(json.dumps({
        "by_sequence": {"GPMF14": {"mesh_file": str(mesh)}}
      }))
      self.assertEqual(
        load_tracking_mesh_overrides(str(path)),
        {"GPMF14": mesh.resolve()},
      )

  def test_pose_validation_requires_full_matching_mv_track(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      mesh = root / "result.glb"
      mesh.touch()
      pose_path = root / "foundationpose_poses.json"
      pose_path.write_text(json.dumps({
        "model_source": "mesh_file",
        "model_path": str(mesh),
        "source_mesh_scale": 0.19,
        "bidirectional": True,
        "uses_gt_object_pose": False,
        "by_frame": {"0000": {}, "0001": {}, "0002": {}},
      }))
      valid, diagnostics = validate_pose_json(
        pose_path, ["0000", "0001", "0002"], mesh, 1.0,
        expected_mesh_scale=0.19,
      )
      self.assertTrue(valid, diagnostics)

      valid, diagnostics = validate_pose_json(
        pose_path, ["0000", "0001", "0002", "0003"], mesh, 1.0,
        expected_mesh_scale=0.19,
      )
      self.assertFalse(valid)
      self.assertEqual(diagnostics["coverage"], 0.75)

      valid, diagnostics = validate_pose_json(
        pose_path, ["0000", "0001", "0002"], mesh, 1.0,
        expected_mesh_scale=1.0,
      )
      self.assertFalse(valid)
      self.assertEqual(diagnostics["mesh_scale"], 0.19)


if __name__ == "__main__":
  unittest.main()
