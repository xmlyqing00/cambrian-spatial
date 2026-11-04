import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class B200RunbookTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.repo = root / "repo"
        scripts = self.repo / "scripts" / "vggt17"
        scripts.mkdir(parents=True)
        source = Path(__file__).resolve().parent
        for name in ("run_b200.sh", "run_train.sh", "run_eval.sh"):
            shutil.copyfile(source / name, scripts / name)
        self.bin = root / "bin"
        self.bin.mkdir()
        (self.bin / "python").symlink_to(sys.executable)
        self.base = root / "data"
        self.env = {**os.environ, "BASE": str(self.base), "REPO": str(self.repo), "PY": str(self.bin / "python"), "PATH": f"{self.bin}:{os.environ['PATH']}"}

    def run_stage(self, *args):
        return subprocess.run(["bash", str(self.repo / "scripts/vggt17/run_b200.sh"), *args], env=self.env, capture_output=True, text=True, timeout=30)

    def test_v2_preserves_matched_launch_arguments_except_lora_rate_and_output(self):
        recorder = self.bin / "record"
        recorder.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        recorder.chmod(0o755)
        self.env["PY"] = str(recorder)
        for arm in ("cam", "self", "all"):
            with self.subTest(arm=arm):
                original = self.run_stage(f"train-{arm}")
                revised = self.run_stage(f"train-{arm}-v2")
                self.assertEqual(original.returncode, 0, original.stderr)
                self.assertEqual(revised.returncode, 0, revised.stderr)
                old, new = json.loads(original.stdout), json.loads(revised.stdout)
                rate = old.index("--learning_rate") + 1
                self.assertEqual(float(old[rate]), 1e-4)
                self.assertEqual(float(new[rate]), 1e-5)
                old[rate] = new[rate]
                output = old.index("--output_dir") + 1
                self.assertEqual(Path(old[output]).name, arm)
                self.assertEqual(Path(new[output]).name, f"{arm}_v2")
                old[output] = new[output]
                self.assertEqual(old, new)
                if arm != "cam":
                    self.assertEqual(float(new[new.index("--vggt_projector_lr") + 1]), 1e-4)

    def write_evaluation(self, rows, metrics=None, jsonl=False):
        metadata = self.base / "data/vggt17_feats/train_manifest/scene10pct_s42/stats.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps({"evaluation": {"qas": 2}}))
        root = self.base / "runs/scene10pct_s42/evals/base/run"
        root.mkdir(parents=True, exist_ok=True)
        (root / "results.json").write_text(json.dumps({"results": {"vsibench": metrics if metrics is not None else {"vsibench_score,none": {"overall": 70.0}}}}))
        if jsonl:
            (root / "samples_vsibench_test.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        else:
            (root / "vsibench.json").write_text(json.dumps({"logs": rows}))
        return root

    def test_selected_evaluation_accepts_legacy_json_and_jsonl(self):
        rows = [{"doc_id": 0, "filtered_resps": ["A"]}, {"doc_id": 1, "resps": [["2"]]}]
        for jsonl in (False, True):
            with self.subTest(jsonl=jsonl):
                root = self.write_evaluation(rows, jsonl=jsonl)
                result = self.run_stage("check-evals", "base")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("VERIFIED base: 2 unique QAs, no empty generations", result.stdout)
                (root / ("samples_vsibench_test.jsonl" if jsonl else "vsibench.json")).unlink()

    def test_evaluation_rejects_duplicate_empty_and_nonfinite_results(self):
        valid = [{"doc_id": 0, "filtered_resps": ["A"]}, {"doc_id": 1, "filtered_resps": ["B"]}]
        duplicate, empty = copy.deepcopy(valid), copy.deepcopy(valid)
        duplicate[1]["doc_id"] = 0
        empty[1]["filtered_resps"] = [[" "]]
        for rows, metrics, error in (
            (duplicate, None, "Duplicated evaluation questions"),
            (empty, None, "empty/failed generations"),
            (valid, {"overall": float("nan")}, "Nonfinite metric"),
        ):
            with self.subTest(error=error):
                self.write_evaluation(rows, metrics)
                result = self.run_stage("check-evals", "base")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)

    def test_v2_evaluation_loads_the_matching_adapter_and_unversioned_scope(self):
        self.write_evaluation([{"doc_id": 0, "filtered_resps": ["A"]}, {"doc_id": 1, "filtered_resps": ["B"]}])
        accelerator = self.bin / "accelerate"
        accelerator.write_text(
            f"#!{sys.executable}\n"
            "import json, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "out = Path(args[args.index('--output_path') + 1])\n"
            "out.mkdir(parents=True)\n"
            "(out / 'results.json').write_text(json.dumps({'results': {'vsibench': {'overall': 70.0}}}))\n"
            "(out / 'vsibench.json').write_text(json.dumps({'logs': [{'doc_id': 0, 'filtered_resps': ['A']}, {'doc_id': 1, 'filtered_resps': ['B']}]}))\n"
            "print(json.dumps(args))\n"
        )
        accelerator.chmod(0o755)
        for arm in ("cam", "self", "all"):
            with self.subTest(arm=arm):
                result = self.run_stage(f"eval-{arm}-v2")
                self.assertEqual(result.returncode, 0, result.stderr)
                log = self.repo / f"output/vggt17/logs/eval_{arm}_v2.log"
                args = json.loads(log.read_text())
                model_args = dict(item.split("=", 1) for item in args[args.index("--model_args") + 1].split(","))
                self.assertEqual(Path(model_args["lora_path"]).name, f"{arm}_v2")
                self.assertEqual(model_args.get("vggt_gather_scope"), None if arm == "cam" else arm)
                self.assertEqual("vggt_feat_dir" in model_args, arm != "cam")
                self.assertNotIn("--limit", args)

    def test_evaluation_launch_failure_and_missing_results_return_nonzero(self):
        accelerator = self.bin / "accelerate"
        self.env["OUT_ROOT"] = str(self.base / "evals")
        for status in (0, 7):
            with self.subTest(status=status):
                accelerator.write_text(f"#!/usr/bin/env bash\nexit {status}\n")
                accelerator.chmod(0o755)
                result = subprocess.run(["bash", str(self.repo / "scripts/vggt17/run_eval.sh"), "0", "29630", "test", "all"], env=self.env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, status or 1)
                self.assertIn("EVAL test FAILED", result.stderr)


if __name__ == "__main__":
    unittest.main()
