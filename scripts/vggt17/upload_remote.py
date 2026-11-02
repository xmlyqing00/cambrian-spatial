"""Upload the VGGT-17 training and eval data to a Kubernetes pod: tar | kubectl exec -i POD -- tar -x, in parallel chunks.

Remote layout under DEST, mirroring the local one (point the launchers at it with DATA_PATH, FEATS and HF_HOME):
  VSI-590K/<source>/<video>.mp4                          the training set's videos       DATA_PATH=DEST/VSI-590K
  vggt17_feats/train/<source>/<stem>.npz                 their VGGT-17 features          FEATS=DEST/vggt17_feats
  vggt17_feats/train_manifest/<set>.jsonl, .stats.json   the training set
  vggt17_feats/stats_train_group.npz                     training feature stats
  vggt17_feats/vsibench/<dataset>/<scene>.npz, vggt17_feats/stats_eval_group.npz   VSI-Bench features
  hf_home/vsibench/<dataset>/<scene>.mp4                 VSI-Bench videos (lmms-eval reads $HF_HOME/vsibench)  HF_HOME=DEST/hf_home
  group vsc (not in the default groups): vggt17_feats/vsc/, vggt17_feats/vsc_cuts.json, hf_home/cambrians_vsc/
Files already on the pod with the local size are skipped, so a rerun resumes. At the end every file's size is checked and,
with --md5, its checksum; files that differ are uploaded again (up to --rounds times). Exits 1 if a file is missing
locally (e.g. features not computed yet) or still differs on the pod.

  python scripts/vggt17/upload_remote.py [--set JSONL] [--groups train,eval] [--pod POD] [--dest DIR] [--md5] [--dry_run]
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time

FEATS = "/data/dataset/vggt17_feats"


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def kexec(pod, script, stdin=None, timeout=600, attempts=3):
    """stdout of `sh -c script` on the pod; retries transient kubectl failures."""
    cmd = ["kubectl", "exec", *(["-i"] if stdin is not None else []), pod, "--", "sh", "-c", script]
    for attempt in range(1, attempts + 1):
        try:
            r = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)
            if r.returncode == 0:
                return r.stdout.decode()
            err = r.stderr.decode().strip()[-500:]
        except subprocess.TimeoutExpired:
            err = f"timeout after {timeout} s"
        log(f"kubectl exec failed (attempt {attempt}/{attempts}): {err}")
        time.sleep(10 * attempt)
    raise RuntimeError(f"kubectl exec on {pod} failed: {script[:200]}")


def walk(root, sub):
    out = []
    for d, _, files in os.walk(os.path.join(root, sub), followlinks=True):
        out += [os.path.relpath(os.path.join(d, f), root) for f in files]
    return sorted(out)


def plan(args):
    """{(local root, remote dir): [paths relative to both]} of the selected groups, and the paths missing locally."""
    trees, missing = {}, []
    rfeats, rhf = f"{args.dest}/vggt17_feats", f"{args.dest}/hf_home"

    def add(root, rdir, rels):
        for rel in rels:
            if os.path.isfile(os.path.join(root, rel)):
                trees.setdefault((root, rdir), []).append(rel)
            else:
                missing.append(os.path.join(root, rel))

    def add_dir(root, rdir, sub):
        if os.path.isdir(os.path.join(root, sub)):
            add(root, rdir, walk(root, sub))
        else:
            missing.append(os.path.join(root, sub) + "/")

    if "train" in args.groups:
        videos = sorted({json.loads(line)["video"] for line in open(args.set)})
        add(args.videos, f"{args.dest}/VSI-590K", videos)
        # <feat_dir>/<video parent dir>/<video stem>.npz, as cambrianp.model.multimodal_projector.vggt_token.vggt_feature_path
        add(args.feats, rfeats, [os.path.join("train", os.path.basename(os.path.dirname(v)),
                                              os.path.splitext(os.path.basename(v))[0] + ".npz") for v in videos])
        add(args.feats, rfeats, ["stats_train_group.npz"])
        name = os.path.splitext(os.path.basename(args.set))[0]
        add(os.path.dirname(os.path.abspath(args.set)), f"{rfeats}/train_manifest", [name + ".jsonl", name + ".stats.json"])
        log(f"train: {args.set}: {len(videos)} videos")
    if "eval" in args.groups:
        add_dir(args.feats, rfeats, "vsibench")
        add(args.feats, rfeats, ["stats_eval_group.npz"])
        add_dir(args.hf_home, rhf, "vsibench")
    if "vsc" in args.groups:
        add_dir(args.feats, rfeats, "vsc")
        add(args.feats, rfeats, ["vsc_cuts.json"])
        add_dir(args.hf_home, rhf, "cambrians_vsc")
    return trees, missing


def remote_sizes(pod, rdir):
    q = shlex.quote(rdir)
    out = kexec(pod, f"mkdir -p {q} && cd {q} && find . -type f -printf '%s %P\\n'")
    sizes = {}
    for line in out.splitlines():
        size, rel = line.split(" ", 1)
        sizes[rel] = int(size)
    return sizes


def md5_local(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def md5_remote(pod, rdir, rels):
    out = kexec(pod, f"cd {shlex.quote(rdir)} && xargs -0 -n 16 -P 8 md5sum --", stdin=("\0".join(rels) + "\0").encode(),
                timeout=7200)
    return {line[34:]: line[:32] for line in out.splitlines()}


def send(pod, root, rdir, rels, nbytes):
    """Stream one chunk; returns an error message or None."""
    with tempfile.NamedTemporaryFile("w", suffix=".lst", delete=False) as f:
        f.write("\0".join(rels) + "\0")
    try:
        tar = subprocess.Popen(["tar", "-C", root, "--dereference", "--null", "-T", f.name, "-cf", "-"], stdout=subprocess.PIPE)
        kc = subprocess.Popen(["kubectl", "exec", "-i", pod, "--", "tar", "-xf", "-", "-C", rdir], stdin=tar.stdout)
        tar.stdout.close()
        try:
            kc.wait(timeout=max(600, nbytes / 2e6))
        except subprocess.TimeoutExpired:
            kc.kill()
            tar.kill()
            kc.wait()
            tar.wait()
            return "timeout"
        tar.wait()
        if tar.returncode or kc.returncode:
            return f"tar exit {tar.returncode}, kubectl exit {kc.returncode}"
        return None
    finally:
        os.unlink(f.name)


def upload(args, todo):
    """todo: {(root, rdir): [rel]}; uploads in chunks of <= --chunk_gb on --streams parallel streams."""
    chunks = []
    for (root, rdir), rels in todo.items():
        cur, cur_bytes = [], 0
        for rel in rels:
            size = os.path.getsize(os.path.join(root, rel))
            if cur and cur_bytes + size > args.chunk_gb * 2**30:
                chunks.append((root, rdir, cur, cur_bytes))
                cur, cur_bytes = [], 0
            cur.append(rel)
            cur_bytes += size
        if cur:
            chunks.append((root, rdir, cur, cur_bytes))
    total = sum(c[3] for c in chunks)
    log(f"uploading {sum(len(c[2]) for c in chunks)} files, {total / 1e9:.2f} GB in {len(chunks)} chunks on {args.streams} streams")
    done, lock, t0 = [0], threading.Lock(), time.time()

    def run(i, chunk):
        root, rdir, rels, nbytes = chunk
        t = time.time()
        err = send(args.pod, root, rdir, rels, nbytes)
        with lock:
            done[0] += nbytes
            log(f"chunk {i + 1}/{len(chunks)} -> {rdir}: {len(rels)} files, {nbytes / 1e9:.2f} GB in {time.time() - t:.0f} s"
                f"{' FAILED: ' + err if err else ''}; {done[0] / 1e9:.1f}/{total / 1e9:.1f} GB, "
                f"{done[0] / 1e6 / max(time.time() - t0, 1e-3):.0f} MB/s")

    with cf.ThreadPoolExecutor(args.streams) as pool:
        list(pool.map(run, range(len(chunks)), chunks))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=f"{FEATS}/train_manifest/headroom_pack4_10000.jsonl")
    ap.add_argument("--groups", default="train,eval", help="comma list of train, eval, vsc")
    ap.add_argument("--pod", default="1001200-lz1bhc-2gpu-b200")
    ap.add_argument("--dest", default="/data/lz1bhc/data")
    ap.add_argument("--videos", default=os.path.expanduser("~/datasets/VSI-590K"), help="local DATA_PATH of the set")
    ap.add_argument("--feats", default=FEATS)
    ap.add_argument("--hf_home", default=os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")))
    ap.add_argument("--streams", type=int, default=3)
    ap.add_argument("--chunk_gb", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=3, help="re-uploads of files that differ after the first upload")
    ap.add_argument("--md5", action="store_true", help="also compare md5 checksums of all files")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    args.groups = set(args.groups.split(","))
    args.dest = args.dest.rstrip("/")
    assert args.groups <= {"train", "eval", "vsc"}, args.groups

    trees, missing = plan(args)
    for (root, rdir), rels in trees.items():
        nbytes = sum(os.path.getsize(os.path.join(root, r)) for r in rels)
        log(f"{root} -> {args.pod}:{rdir}: {len(rels)} files, {nbytes / 1e9:.2f} GB")
    if missing:
        log(f"{len(missing)} files missing locally (not uploaded), e.g. {missing[:3]}")

    local_md5, verified = {}, set()
    for rnd in range(args.rounds + 1):
        bad = {}
        for (root, rdir), rels in trees.items():
            remote = remote_sizes(args.pod, rdir)
            diff = [r for r in rels if remote.get(r) != os.path.getsize(os.path.join(root, r))]
            if diff:
                bad[(root, rdir)] = diff
        if not bad and args.md5 and not args.dry_run:
            for (root, rdir), rels in trees.items():
                rels = [r for r in rels if (rdir, r) not in verified]
                if not rels:
                    continue
                log(f"md5 of {len(rels)} files in {rdir}")
                with cf.ThreadPoolExecutor(8) as pool:
                    local_md5.update(zip(((rdir, r) for r in rels), pool.map(md5_local, (os.path.join(root, r) for r in rels))))
                remote = md5_remote(args.pod, rdir, rels)
                diff = [r for r in rels if remote.get(r) != local_md5[(rdir, r)]]
                verified.update((rdir, r) for r in rels if r not in diff)
                if diff:
                    bad[(root, rdir)] = diff
        n_bad = sum(map(len, bad.values()))
        if args.dry_run or not n_bad or rnd == args.rounds:
            break
        if rnd:
            log(f"round {rnd}: {n_bad} files differ on the pod, e.g. {next(iter(bad.values()))[:3]}: uploading them again")
        upload(args, bad)

    n_files = sum(map(len, trees.values()))
    if args.dry_run:
        log(f"dry run: {n_bad}/{n_files} files to upload, "
            f"{sum(os.path.getsize(os.path.join(root, r)) for (root, _), rels in bad.items() for r in rels) / 1e9:.2f} GB")
        return
    check = "size and md5" if args.md5 else "size"
    if n_bad:
        log(f"FAILED: {n_bad}/{n_files} files still differ on the pod ({check})")
    else:
        log(f"OK: all {n_files} files on {args.pod}:{args.dest} match ({check})")
    if missing:
        log(f"INCOMPLETE: {len(missing)} files missing locally, e.g. {missing[:3]}")
    sys.exit(1 if n_bad or missing else 0)


if __name__ == "__main__":
    main()
