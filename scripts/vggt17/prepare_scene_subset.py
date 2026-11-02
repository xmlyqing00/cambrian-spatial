"""Select nested, source-stratified training groups without changing original annotations or evaluation."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

GROUPING = {
    "scannet": "room ID; alternate sceneXXXX_* captures stay together",
    "scannetppv2": "scene ID",
    "arkitscenes": "recording ID; physical-room IDs are not supplied",
    "adt": "sequence/recording ID; not the apartment category",
    "procthor": "house directory; navigation recordings stay together",
    "s3dis": "area and room",
    "hypersim": "scene directory; camera views stay together",
    "robotics": "Agibot recording ID or exported dataset shard/camera stream; physical-scene IDs are not supplied",
    "ytb_roomtour": "original video ID; exported frames stay together",
}


def media_group(media):
    path = PurePosixPath(media)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        raise ValueError(f"Expected a relative dataset media path: {media}")
    source, stem = path.parts[0], path.stem
    source = "scannetppv2" if source == "scannetpp" else source
    if source not in GROUPING:
        raise ValueError(f"Unknown training source: {source}")
    if source == "scannet":
        group = stem.rsplit("_", 1)[0]
    elif source in ("hypersim", "procthor"):
        if len(path.parts) < 3:
            raise ValueError(f"Missing scene/house directory: {media}")
        group = path.parts[1]
    elif source == "s3dis":
        group = "/".join(path.parts[1:-1] + (stem,))
    elif source == "ytb_roomtour":
        match = re.fullmatch(r"(.+)_frame_\d+", stem)
        if not match:
            raise ValueError(f"Cannot identify original room-tour video: {media}")
        group = match[1]
    elif source == "robotics":
        if "_sample_" in stem:
            group = stem.split("_sample_", 1)[0]
        elif stem.startswith("agibot_"):
            recording = re.match(r"(agibot_[0-9a-f]+)_", stem)
            group = recording[1] if recording else re.sub(r"(?:_frame)?_\d+$", "", stem)
        else:
            raise ValueError(f"Cannot identify robotics recording/shard: {media}")
    else:
        group = stem
    return source, group


def row_media(row):
    video, image = row.get("video"), row.get("image")
    if bool(video) == bool(image):
        raise ValueError("Each training row must have exactly one of video or image")
    paths = [video] if video else [image] if isinstance(image, str) else image
    groups = {media_group(path) for path in paths}
    if len(groups) != 1:
        raise ValueError(f"One conversation spans multiple scene groups: {paths}")
    source, group = groups.pop()
    return source, group, "videos" if video else "images", paths


def choose_groups(groups, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return {
        source: set(
            sorted(
                keys,
                key=lambda key: (hashlib.sha256(f"{seed}\0{source}\0{key}".encode()).hexdigest(), key),
            )[: math.ceil(len(keys) * fraction)]
        )
        for source, keys in groups.items()
    }


def require_media(root, paths):
    missing = [path for path in sorted(paths) if not (root / path).is_file() or (root / path).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing/empty inputs under {root}, e.g. {missing[:3]}")


def prepare_subset(jsonl, root, out, fraction, seed, eval_videos, eval_root):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty; preserve it and choose a new one: {out}")
    held_out = {media_group(video) for video in eval_videos}
    require_media(eval_root, eval_videos)
    groups = defaultdict(lambda: defaultdict(lambda: dict(records=0, qas=0, videos=set(), images=set())))
    excluded = defaultdict(set)
    digest = hashlib.sha256()
    with jsonl.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            source, group, modality, paths = row_media(row)
            if (source, group) in held_out:
                excluded[source].add(group)
                continue
            entry = groups[source][group]
            entry["records"] += 1
            entry["qas"] += sum(turn.get("from") == "gpt" or turn.get("role") == "assistant" for turn in row["conversations"])
            entry[modality].update(paths)
    if set(groups) != set(GROUPING):
        raise ValueError(f"Expected all nine sources; missing {sorted(set(GROUPING) - set(groups))}")
    selected = choose_groups(groups, fraction, seed)
    stats = dict(seed=seed, fraction=fraction, input_jsonl=str(jsonl), input_sha256=digest.hexdigest(), grouping=GROUPING, sources={})
    train_videos = set()
    for source in sorted(groups):
        chosen = [groups[source][key] for key in selected[source]]
        videos = set().union(*(entry["videos"] for entry in chosen))
        images = set().union(*(entry["images"] for entry in chosen))
        require_media(root, videos | images)
        train_videos.update(videos)
        stats["sources"][source] = dict(
            eligible_groups=len(groups[source]),
            selected_groups=len(chosen),
            excluded_benchmark_groups=len(excluded[source]),
            records=sum(entry["records"] for entry in chosen),
            qas=sum(entry["qas"] for entry in chosen),
            videos=len(videos),
            images=len(images),
            selected_group_ids=sorted(selected[source]),
        )
    out.mkdir(parents=True, exist_ok=True)
    smoke = {}
    with jsonl.open("rb") as stream, (out / "train.jsonl").open("wb") as target:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            source, group, _, _ = row_media(row)
            if group not in selected[source]:
                continue
            target.write(line if line.endswith(b"\n") else line + b"\n")
            length = sum(len(turn["value"]) for turn in row["conversations"])
            if source not in smoke or length > smoke[source][0]:
                smoke[source] = length, line, row
    (out / "smoke_train.jsonl").write_bytes(b"".join(smoke[source][1].rstrip(b"\n") + b"\n" for source in sorted(smoke)))
    lists = {
        "train_videos.txt": train_videos,
        "smoke_train_videos.txt": {entry[2]["video"] for entry in smoke.values() if "video" in entry[2]},
        "eval_videos.txt": eval_videos,
    }
    for name, videos in lists.items():
        (out / name).write_text("".join(video + "\n" for video in sorted(videos)))
    stats["totals"] = {key: sum(source[key] for source in stats["sources"].values()) for key in ("selected_groups", "records", "qas", "videos", "images")}
    stats["evaluation"] = dict(videos=len(eval_videos), unchanged=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--hf_home", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    hf = Path(args.hf_home).expanduser()
    arrows = list((hf / "datasets/nyu-visionx___vsi-bench/full").rglob("vsi-bench-test.arrow"))
    if len(arrows) != 1:
        raise ValueError(f"Expected one original full benchmark Arrow cache under {hf}, found {len(arrows)}")
    import pyarrow as pa

    with pa.memory_map(str(arrows[0]), "r") as stream:
        table = pa.ipc.open_stream(stream).read_all()
    eval_videos = {f"{row['dataset']}/{row['scene_name']}.mp4" for row in table.select(["dataset", "scene_name"]).to_pylist()}
    out = Path(args.out).expanduser()
    stats = prepare_subset(Path(args.jsonl).expanduser(), Path(args.root).expanduser(), out, args.fraction, args.seed, eval_videos, hf / "vsibench")
    stats["evaluation"].update(qas=table.num_rows, annotation_arrow=str(arrows[0]))
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({key: value for key, value in stats.items() if key != "sources"}, indent=2))
    for source, counts in stats["sources"].items():
        print(source, {key: value for key, value in counts.items() if key != "selected_group_ids"})


if __name__ == "__main__":
    main()
