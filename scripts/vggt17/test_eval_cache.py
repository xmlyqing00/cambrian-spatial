import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lmms-eval"))
from lmms_eval.api.task import ConfigurableTask


class EvaluationCacheTests(unittest.TestCase):
    def test_cached_benchmark_preserves_split_rows_and_processing_without_hub_access(self):
        original = Dataset.from_dict({"id": [3, 4], "question": ["First?", "Second?"], "ground_truth": ["1", "2"]})
        with tempfile.TemporaryDirectory() as name:
            original.save_to_disk(name)
            arrow = next(Path(name).glob("*.arrow"))
            task = SimpleNamespace(
                DATASET_PATH="nyu-visionx/VSI-Bench",
                config=SimpleNamespace(test_split="test", process_docs=lambda dataset: dataset.select([1, 0])),
            )
            with mock.patch.dict(os.environ, {"VSI_BENCH_ARROW": str(arrow)}), mock.patch("lmms_eval.api.task.datasets.load_dataset", side_effect=AssertionError("Unexpected Hub dataset lookup")), mock.patch(
                "lmms_eval.api.task.snapshot_download", side_effect=AssertionError("Unexpected video download")
            ):
                ConfigurableTask.download.__wrapped__(task, {"video": True, "cache_dir": "vsibench", "token": True})
            self.assertEqual(list(task.dataset), ["test"])
            self.assertEqual(task.dataset["test"].to_list(), original.select([1, 0]).to_list())
            self.assertEqual(task.dataset_no_image["test"].to_list(), task.dataset["test"].to_list())

    def test_missing_explicit_cache_does_not_fall_back_to_downloads(self):
        task = SimpleNamespace(DATASET_PATH="nyu-visionx/VSI-Bench", config=SimpleNamespace(test_split="test", process_docs=None))
        with tempfile.TemporaryDirectory() as name, mock.patch.dict(os.environ, {"VSI_BENCH_ARROW": str(Path(name) / "missing.arrow")}), mock.patch(
            "lmms_eval.api.task.datasets.load_dataset", side_effect=AssertionError("Unexpected download fallback")
        ):
            with self.assertRaises(FileNotFoundError):
                ConfigurableTask.download.__wrapped__(task)


if __name__ == "__main__":
    unittest.main()
