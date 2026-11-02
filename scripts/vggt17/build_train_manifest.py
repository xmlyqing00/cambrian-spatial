"""VSI-590K scan videos for VGGT-17 training: the ScanNet / ScanNet++ / ARKitScenes videos referenced by the jsonl, minus
VSI-Bench (ScanNet at room level, i.e. every scan sceneXXXX_*), in a seeded random order. Precompute features in this
order and train on the videos that are done, so any prefix is an unbiased sample.

  python scripts/vggt17/build_train_manifest.py            # -> <out>/videos.txt (paths relative to --video_root), stats.json
  python scripts/vggt17/build_train_manifest.py --subset_qas 10000 --feat_dir /data/dataset/vggt17_feats/train
      # -> <out>/train_<N>.jsonl: N QAs drawn uniformly (seeded) from the videos that have features
"""

import argparse
import collections
import glob
import json
import os
import random

from cambrianp.datasets.vqa_dataloading_utils import video_scene_key

# VSI-590K video dir -> VSI-Bench video dir
SOURCES = {"scannet": "scannet", "scannetppv2": "scannetpp", "arkitscenes": "arkitscenes"}


def scene_key(source, stem):
    return video_scene_key(os.path.join(source, stem + ".mp4"))


def split_samples(items, val_fraction, seed, held_out):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    by_source = collections.defaultdict(set)
    for item in items:
        if not item.get("video"):
            raise ValueError("scene-disjoint splitting requires a video path in every sample")
        key = video_scene_key(item["video"])
        by_source[key[0]].add(key)
    scenes = set().union(*by_source.values())
    overlap = scenes & held_out
    if overlap:
        raise ValueError(f"Input contains benchmark scenes ({len(overlap)}), e.g. {sorted(overlap)[:3]}")
    if len(scenes) < 2:
        raise ValueError("at least two scenes are needed for train/validation")
    rng = random.Random(seed)
    val_scenes = set()
    for source in sorted(by_source):
        keys = sorted(by_source[source])
        rng.shuffle(keys)
        count = min(len(keys) - 1, max(1, round(len(keys) * val_fraction))) if len(keys) > 1 else 0
        val_scenes.update(keys[:count])
    if not val_scenes:
        val_scenes.add(rng.choice(sorted(scenes)))
    train, val = [], []
    for item in items:
        (val if video_scene_key(item["video"]) in val_scenes else train).append(item)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/data/dataset/VSI-590K/vsi_590k.jsonl")
    ap.add_argument("--video_root", default="~/datasets/VSI-590K")
    ap.add_argument("--vsibench_root", default="~/.cache/huggingface/vsibench")
    ap.add_argument("--out", default="/data/dataset/vggt17_feats/train_manifest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset_qas", type=int, help="write train_<N>.jsonl from videos with features in --feat_dir")
    ap.add_argument("--feat_dir", default="/data/dataset/vggt17_feats/train")
    ap.add_argument("--split_jsonl", help="split an existing packed/unpacked manifest by source and scene, preserving all turns")
    ap.add_argument("--val_fraction", type=float, default=0.1)
    a = ap.parse_args()
    root, bench = os.path.expanduser(a.video_root), os.path.expanduser(a.vsibench_root)
    os.makedirs(a.out, exist_ok=True)

    held_out = {scene_key(src, os.path.splitext(os.path.basename(p))[0]) for src, bench_dir in SOURCES.items() for p in glob.glob(os.path.join(bench, bench_dir, "*.mp4"))}
    if a.split_jsonl:
        if not held_out:
            raise SystemExit(f"No benchmark videos in {bench}; cannot verify test-set exclusion")
        with open(a.split_jsonl) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        train, val = split_samples(rows, a.val_fraction, a.seed, held_out)
        names = ("train.jsonl", "val.jsonl", "train_videos.txt", "val_videos.txt", "split_stats.json")
        if any(os.path.exists(os.path.join(a.out, name)) for name in names):
            raise SystemExit(f"Split outputs already exist in {a.out}; choose a new output directory")
        stats = dict(seed=a.seed, val_fraction=a.val_fraction, held_out_scene_keys=len(held_out), input=a.split_jsonl)
        for name, subset in (("train", train), ("val", val)):
            videos = sorted({d["video"] for d in subset})
            with open(os.path.join(a.out, name + ".jsonl"), "w") as f:
                f.writelines(json.dumps(d) + "\n" for d in subset)
            with open(os.path.join(a.out, name + "_videos.txt"), "w") as f:
                f.write("\n".join(videos) + "\n")
            stats[name] = dict(
                conversations=len(subset), qas=sum(len(d["conversations"]) // 2 for d in subset), videos=len(videos), scenes=len({video_scene_key(v) for v in videos}), sources=dict(collections.Counter(video_scene_key(v)[0] for v in videos))
            )
        with open(os.path.join(a.out, "split_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(json.dumps(stats, indent=2))
        return
    items = collections.defaultdict(list)
    with open(a.jsonl) as f:
        for line in f:
            d = json.loads(line)
            v = d.get("video")
            if v and v.split("/")[0] in SOURCES:
                items[v].append(d)
    stats = collections.Counter()
    keep = []
    for v in sorted(items):
        src, stem = v.split("/")[0], os.path.splitext(os.path.basename(v))[0]
        if scene_key(src, stem) in held_out:
            stats[f"{src}/held_out_videos"] += 1
        elif not os.path.exists(os.path.join(root, v)):
            stats[f"{src}/missing_videos"] += 1
        else:
            keep.append(v)
            stats[f"{src}/videos"] += 1
            stats[f"{src}/qas"] += len(items[v])
    random.Random(a.seed).shuffle(keep)

    if a.subset_qas is None:
        with open(os.path.join(a.out, "videos.txt"), "w") as f:
            f.write("\n".join(keep) + "\n")
        stats = dict(sorted(stats.items()), held_out_scene_keys=len(held_out), seed=a.seed)
        with open(os.path.join(a.out, "stats.json"), "w") as f:
            json.dump(stats, f, indent=1)
        print(json.dumps(stats, indent=1))
        return

    have = [v for v in keep if os.path.exists(os.path.join(a.feat_dir, os.path.splitext(v)[0] + ".npz"))]
    pool = [d for v in have for d in items[v]]
    if len(pool) < a.subset_qas:
        raise SystemExit(f"only {len(pool)} QAs from {len(have)} videos with features, asked for {a.subset_qas}")
    subset = random.Random(a.seed).sample(pool, a.subset_qas)
    dst = os.path.join(a.out, f"train_{a.subset_qas}.jsonl")
    with open(dst, "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in subset)
    types = collections.Counter(d.get("question_type") for d in subset)
    print(f"{dst}: {len(subset)} QAs from {len({d['video'] for d in subset})}/{len(have)} videos with features; {dict(types.most_common())}")


if __name__ == "__main__":
    main()
