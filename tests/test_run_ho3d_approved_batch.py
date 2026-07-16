import json
import tempfile
import unittest
from pathlib import Path

from run_ho3d_approved_batch import (
  approved_sequence_entries,
  choose_distributed_candidates,
  resolve_mv_glb,
  validate_pose_json,
)


class ApprovedBatchTests(unittest.TestCase):
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

  def test_pose_validation_requires_full_matching_mv_track(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      mesh = root / "result.glb"
      mesh.touch()
      pose_path = root / "foundationpose_poses.json"
      pose_path.write_text(json.dumps({
        "model_source": "mesh_file",
        "model_path": str(mesh),
        "bidirectional": True,
        "uses_gt_object_pose": False,
        "by_frame": {"0000": {}, "0001": {}, "0002": {}},
      }))
      valid, diagnostics = validate_pose_json(
        pose_path, ["0000", "0001", "0002"], mesh, 1.0,
      )
      self.assertTrue(valid, diagnostics)

      valid, diagnostics = validate_pose_json(
        pose_path, ["0000", "0001", "0002", "0003"], mesh, 1.0,
      )
      self.assertFalse(valid)
      self.assertEqual(diagnostics["coverage"], 0.75)


if __name__ == "__main__":
  unittest.main()
