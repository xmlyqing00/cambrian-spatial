import json
from pathlib import Path
import tempfile
import unittest

from scripts.vggt17.prepare_scene_subset import choose_groups, media_group, prepare_subset


class SceneSubsetTests(unittest.TestCase):
    def test_related_views_share_selection_units(self):
        pairs = [
            ("scannet/scene0010_00.mp4", "scannet/scene0010_01.mp4"),
            ("scannetppv2/abcd.mp4", "scannetpp/abcd.mp4"),
            ("procthor/001252/raw_navigation_camera__1.mp4", "procthor/001252/raw_navigation_camera__2.mp4"),
            ("hypersim/ai_001_001/images/camera00/frame.0000.jpg", "hypersim/ai_001_001/images/camera01/frame.0001.jpg"),
            ("robotics/agibot_ea949482_videos_head_right_color_0100.jpg", "robotics/agibot_ea949482_videos_back_left_color_0200.jpg"),
            ("robotics/robotics_kuka_00290_sample_001.jpg", "robotics/robotics_kuka_00290_sample_002.jpg"),
            ("robotics/agibot_back_left_fisheye_frame_403.jpg", "robotics/agibot_back_left_fisheye_frame_404.jpg"),
            ("ytb_roomtour/a_b-frame_frame_0001.jpg", "ytb_roomtour/a_b-frame_frame_0002.jpg"),
        ]
        for left, right in pairs:
            with self.subTest(left=left):
                self.assertEqual(media_group(left), media_group(right))

    def test_larger_fractions_are_nested_and_input_order_independent(self):
        groups = {"scannet": [f"scene{i:04d}" for i in range(31)]}
        small = choose_groups(groups, 0.1, 42)
        large = choose_groups({"scannet": list(reversed(groups["scannet"]))}, 0.2, 42)
        self.assertLess(small["scannet"], large["scannet"])
        self.assertEqual(small, choose_groups({"scannet": list(reversed(groups["scannet"]))}, 0.1, 42))

    def test_all_sources_keep_original_conversations_and_exclude_benchmark_rooms(self):
        media = [
            "scannet/scene0001_00.mp4",
            "scannet/scene0001_01.mp4",
            "scannetppv2/abcd.mp4",
            "arkitscenes/12345.mp4",
            "adt/seq1.mp4",
            "procthor/001252/navigation1.mp4",
            "s3dis/area_1/Area_1_office_1.mp4",
            "hypersim/ai_001_001/images/frame.0000.jpg",
            "robotics/robotics_kuka_00290_sample_001.jpg",
            "ytb_roomtour/video_frame_0001.jpg",
            "scannet/scene9999_00.mp4",
        ]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows = []
            for index, path in enumerate(media):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"media")
                row = {
                    ("video" if path.endswith(".mp4") else "image"): path,
                    "id": index,
                    "conversations": [
                        {"from": "human", "value": "<image>\nQuestion?"},
                        {"from": "gpt", "value": str(index)},
                        {"from": "human", "value": "Another question?"},
                        {"from": "gpt", "value": "Another answer"},
                    ],
                }
                rows.append(row)
            original = root / "original.jsonl"
            original.write_text("".join(json.dumps(row) + "\n" for row in rows))
            before = original.read_bytes()
            benchmark = root / "scannet/scene9999_01.mp4"
            benchmark.write_bytes(b"benchmark")
            out = root / "subset"
            stats = prepare_subset(original, root, out, 1.0, 42, {"scannet/scene9999_01.mp4"}, root)
            selected = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
            self.assertEqual(selected, rows[:-1])
            self.assertEqual(original.read_bytes(), before)
            self.assertEqual(stats["totals"]["qas"], 2 * len(selected))
            self.assertEqual(stats["sources"]["scannet"]["excluded_benchmark_groups"], 1)
            self.assertEqual(len((out / "smoke_train.jsonl").read_text().splitlines()), 9)
            videos = (out / "train_videos.txt").read_text().splitlines()
            self.assertEqual(len(videos), 7)
            self.assertTrue(all(path.endswith(".mp4") for path in videos))
            with self.assertRaises(FileExistsError):
                prepare_subset(original, root, out, 1.0, 42, set(), root)

    def test_invalid_fractions_and_unsafe_paths_are_rejected(self):
        for fraction in (0, -0.1, 1.1):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                choose_groups({"scannet": {"scene1"}}, fraction, 42)
        for path in ("/scannet/a.mp4", "scannet/../a.mp4", "unknown/a.mp4"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                media_group(path)


if __name__ == "__main__":
    unittest.main()
