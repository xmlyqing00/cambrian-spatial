"""Multi-turn ("packed") VSI-590K training set with a chosen question-type mix, for the VGGT-17 vs control comparison.

  python scripts/vggt17/build_packed_set.py            # -> <out_dir>/headroom_pack4_10000.jsonl (+ .stats.json)
  python scripts/vggt17/build_packed_set.py --feat_dir /data/dataset/vggt17_feats/train --num_qas 400 --tag smoke
      # only videos whose VGGT features already exist (smoke tests)

QAs come from the first --num_videos manifest videos (those run_precompute_train.sh computes VGGT features for) and are
packed into --num_qas / --pack conversations of --pack turns about one video, with a question-type mix close to --mix.
"headroom" favours the types where the base model still has room on its own training QAs
(output/vggt17/loss_probe/summary.json) and on VSI-Bench: ~70% numeric, ~25% relative distance / count / appearance
order, and a few saturated relative direction / size QAs against forgetting them. A type with fewer QAs than its share
(room size: one per video) gives all it has and the other types are scaled up.

No two turns of a conversation are about a common object with the same type or the related absolute / relative type
(e.g. the absolute and the relative direction of the same objects), so an earlier answer never gives a later one away.
Videos get conversations in proportion to their share of the mix (at least one; videos that cannot fill one are left
out), each QA is used at most once, and turns prefer distinct types. `<image>` and the "These are frames of a video."
preamble stay in the first turn only; turn order is random.
"""

import argparse
import collections
import json
import os
import random
import re

PREAMBLE = "<image>\nThese are frames of a video.\n"
MIXES = {
    "headroom": {
        "absolute_distance_object": 0.20, "absolute_size_object": 0.14, "absolute_direction_object": 0.14,
        "absolute_count": 0.12, "absolute_size_room": 0.10,
        "relative_distance_object": 0.14, "relative_count": 0.07, "appearance_order": 0.05,
        "relative_direction_object": 0.03, "relative_size_object": 0.01,
    },
}
# objects a question names (targets, references and answer options). QAs where the vocabulary finds another number are
# left out, so the conflict check never misses an object
NUM_OBJECTS = {
    "absolute_size_room": 0, "absolute_count": 1, "absolute_size_object": 1, "absolute_distance_object": 2,
    "relative_count": 2, "relative_size_object": 2, "absolute_direction_object": 3, "relative_direction_object": 3,
    "appearance_order": 4, "relative_distance_object": 5,
}
# absolute and relative questions about the same quantity
RELATED = {frozenset(p) for p in [
    ("absolute_direction_object", "relative_direction_object"), ("absolute_size_object", "relative_size_object"),
    ("absolute_count", "relative_count"), ("absolute_distance_object", "relative_distance_object")]}
# types whose questions about different object pairs / triples are independent (bed-tv vs bed-table distance)
PAIRWISE = {"absolute_distance_object", "absolute_direction_object", "relative_direction_object"}


def object_pattern(qas):
    """Object categories: the answer options of the relative distance / size and appearance order questions."""
    vocab = set()
    for d in qas:
        opts = re.findall(r"^[A-Z]\. (.+)$", d["conversations"][0]["value"], re.M)
        if d["question_type"] in ("relative_distance_object", "relative_size_object"):
            vocab.update(o.strip().lower() for o in opts)
        elif d["question_type"] == "appearance_order":
            vocab.update(x.strip().lower() for o in opts for x in o.split(","))
    words = sorted(vocab, key=len, reverse=True)  # longest first: a "sofa chair" is not a "chair"
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")(?:s|es)?\b", re.I)


def conflict(q, r):
    """True if the QAs must not share a conversation: the same type and a common object (the same object set for
    PAIRWISE types), or related types and a common object."""
    if q["t"] == r["t"]:
        return q["objs"] == r["objs"] if q["t"] in PAIRWISE else not q["objs"].isdisjoint(r["objs"])
    return frozenset((q["t"], r["t"])) in RELATED and not q["objs"].isdisjoint(r["objs"])


def targets(mix, avail, n):
    """n QAs split by the weights of mix (largest remainder); a type with fewer available QAs than its share gets them all
    and the rest is split among the others."""
    target, w = {}, dict(mix)
    while True:
        left = n - sum(target.values())
        raw = {t: left * x / sum(w.values()) for t, x in w.items()}
        short = [t for t in w if raw[t] > avail[t]]
        if not short:
            break
        for t in short:
            target[t] = avail[t]
            del w[t]
    fl = {t: int(r) for t, r in raw.items()}
    for t in sorted(raw, key=lambda t: fl[t] - raw[t])[: left - sum(fl.values())]:
        fl[t] += 1
    return {**target, **fl}


def spread(n, avail, rng):
    """n items over the keys of avail as evenly as possible (round robin, random order), at most avail[key] each."""
    keys = list(avail)
    rng.shuffle(keys)
    k = dict.fromkeys(keys, 0)
    while n > 0:
        open_ = [v for v in keys if k[v] < avail[v]]
        if not open_:
            break
        for v in open_[:n]:
            k[v] += 1
        n -= min(n, len(open_))
    return k


def assign(order, convs, pack, placed, distinct, quota=None):
    """Put each QA not yet placed (in order) into the least-filled conversation it fits: room left, no conflict and, if
    distinct, a new type; with quota, at most quota[t] QAs of type t in all conversations."""
    n = collections.Counter(q["t"] for cv in convs for q in cv)
    for q in order:
        if id(q) in placed or (quota is not None and n[q["t"]] >= quota[q["t"]]):
            continue
        fits = [cv for cv in convs if len(cv) < pack and not any(conflict(q, r) for r in cv)
                and not (distinct and any(r["t"] == q["t"] for r in cv))]
        if fits:
            min(fits, key=len).append(q)
            placed.add(id(q))
            n[q["t"]] += 1


def pack_video(qas, plan, c, pack, rank, rng, tries=20):
    """Up to c full conversations from one video's QAs: plan[t] QAs of each type first, then others in rank order, each
    stage with distinct types first. Best of a few random orders: most conversations, then fewest QAs off the plan."""
    best_key, best = None, []
    for _ in range(tries):
        order = qas[:]
        rng.shuffle(order)
        order.sort(key=lambda q: -len(q["objs"]))  # most constrained first
        convs, placed = [[] for _ in range(c)], set()
        for distinct in (True, False):
            assign(order, convs, pack, placed, distinct, plan)
        off = sorted((q for q in order if id(q) not in placed), key=rank)
        for distinct in (True, False):
            assign(off, convs, pack, placed, distinct)
        full = [cv for cv in convs if len(cv) == pack]
        n = collections.Counter(q["t"] for cv in full for q in cv)
        key = (len(full), -sum(max(0, n[t] - plan[t]) for t in n))
        if best_key is None or key > best_key:
            best_key, best = key, full
        if key == (c, 0):
            break
    return best


def capacity(qas, pack, rng, tries=10):
    """Lower bound on the number of full conversations a video's QAs fill (first fit, best of a few orders)."""
    best = 0
    for _ in range(tries):
        order = qas[:]
        rng.shuffle(order)
        order.sort(key=lambda q: -len(q["objs"]))
        convs = []
        for q in order:
            cv = next((cv for cv in convs if len(cv) < pack and not any(conflict(q, r) for r in cv)), None)
            if cv is None:
                convs.append([q])
            else:
                cv.append(q)
        best = max(best, sum(len(cv) == pack for cv in convs))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/data/dataset/VSI-590K/vsi_590k.jsonl")
    ap.add_argument("--manifest", default="/data/dataset/vggt17_feats/train_manifest/videos.txt")
    ap.add_argument("--num_videos", type=int, default=1000, help="first N manifest videos")
    ap.add_argument("--feat_dir", help="only videos with VGGT features here")
    ap.add_argument("--num_qas", type=int, default=10000)
    ap.add_argument("--pack", type=int, default=4, help="turns per conversation")
    ap.add_argument("--mix", default="headroom", choices=sorted(MIXES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/data/dataset/vggt17_feats/train_manifest")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.num_qas % a.pack:
        raise SystemExit(f"--num_qas {a.num_qas} is not a multiple of --pack {a.pack}")
    rng = random.Random(a.seed)
    mix, num_convs = MIXES[a.mix], a.num_qas // a.pack

    manifest = [l.strip() for l in open(a.manifest) if l.strip()]
    videos = manifest[: a.num_videos]
    if a.feat_dir:
        videos = [v for v in videos if os.path.exists(os.path.join(a.feat_dir, os.path.splitext(v)[0] + ".npz"))]
    videos, scans = set(videos), set(manifest)
    scan_qas = [d for d in map(json.loads, open(a.jsonl)) if d.get("video") in scans]
    pat = object_pattern(scan_qas)
    items = collections.defaultdict(lambda: collections.defaultdict(list))  # video -> type -> QAs
    skipped = collections.Counter()
    for d in scan_qas:
        t = d["question_type"]
        if d["video"] not in videos or t not in mix:
            continue
        h, g = d["conversations"]
        objs = frozenset(m.lower() for m in pat.findall(h["value"]))
        if len(objs) != NUM_OBJECTS[t] or not h["value"].startswith(PREAMBLE) or (h["from"], g["from"]) != ("human", "gpt"):
            skipped[t] += 1
            continue
        items[d["video"]][t].append({"d": d, "objs": objs, "t": t})
    qas_of = {v: [q for t in mix for q in items[v][t]] for v in sorted(items)}
    cap = {v: capacity(qas, a.pack, rng) for v, qas in qas_of.items()}
    vids = [v for v in qas_of if cap[v]]

    # 1. per-type targets over the usable videos, each type spread evenly over the videos that have it
    target = targets(mix, {t: sum(len(items[v][t]) for v in vids) for t in mix}, a.num_qas)
    k = {v: collections.Counter() for v in vids}
    for t in mix:
        for v, n in spread(target[t], {v: len(items[v][t]) for v in vids if items[v][t]}, rng).items():
            k[v][t] = n
    # 2. conversations per video: its share / pack, at least one, at most its capacity, num_convs in all
    share = {v: sum(k[v].values()) / a.pack for v in vids}
    c = {v: min(cap[v], max(1, round(share[v]))) for v in vids}
    while sum(c.values()) > num_convs:
        pool = [v for v in vids if c[v] > 1] or [v for v in vids if c[v] > 0]
        c[max(pool, key=lambda v: (c[v] - share[v], rng.random()))] -= 1
    while sum(c.values()) < num_convs:
        open_ = [v for v in vids if c[v] < cap[v]]
        if not open_:
            raise SystemExit(f"{len(vids)} videos hold only {sum(cap.values())} conversations of {a.pack}, asked for {num_convs}")
        c[max(open_, key=lambda v: (share[v] - c[v], rng.random()))] += 1
    # 3. plan pack * c QAs per video: add the types furthest below target, remove those furthest above
    deficit = {t: target[t] - sum(k[v][t] for v in vids) for t in mix}
    order = rng.sample(vids, len(vids))
    for v in order:
        while sum(k[v].values()) < a.pack * c[v]:
            t = max((t for t in mix if k[v][t] < len(items[v][t])), key=lambda t: (k[v][t] < c[v], deficit[t], rng.random()))
            k[v][t] += 1
            deficit[t] -= 1
        while sum(k[v].values()) > a.pack * c[v]:
            t = max((t for t in mix if k[v][t] > 0), key=lambda t: (k[v][t] > c[v], -deficit[t], rng.random()))
            k[v][t] -= 1
            deficit[t] += 1
    # 4. swap planned QAs of over-target types for under-target ones in the same video (first at most c per type)
    for strict in (True, False):
        moved = True
        while moved:
            moved = False
            for t in sorted((t for t in mix if deficit[t] > 0), key=lambda t: -deficit[t]):
                for u in sorted((u for u in mix if deficit[u] < 0), key=lambda u: deficit[u]):
                    for v in order:
                        if deficit[t] <= 0 or deficit[u] >= 0:
                            break
                        if k[v][u] > 0 and k[v][t] < len(items[v][t]) and (not strict or k[v][t] < c[v]):
                            k[v][u] -= 1
                            k[v][t] += 1
                            deficit[u] += 1
                            deficit[t] -= 1
                            moved = True
    # 5. build each video's conversations, the plan first; then fill missing conversations from videos with QAs left
    got, left = collections.Counter(), collections.Counter()
    for v in vids:
        left.update(k[v])
    # off-plan QAs: the types whose projected count (built + still planned) is relatively furthest below target first
    rank = lambda q: (got[q["t"]] + left[q["t"]] - target[q["t"]]) / max(1, target[q["t"]])
    convs = []
    for v in order:
        for cv in pack_video(qas_of[v], k[v], c[v], a.pack, rank, rng):
            convs.append((v, cv))
            got.update(q["t"] for q in cv)
        left.subtract(k[v])
    off_plan = sum(max(0, got[t] - target[t]) for t in mix)
    used = {id(q) for _, cv in convs for q in cv}
    topped_up = 0
    while len(convs) < num_convs:
        added = False
        for v in rng.sample(vids, len(vids)):
            if len(convs) == num_convs:
                break
            for cv in pack_video([q for q in qas_of[v] if id(q) not in used], collections.Counter(), 1, a.pack, rank, rng, 5):
                convs.append((v, cv))
                used.update(id(q) for q in cv)
                got.update(q["t"] for q in cv)
                topped_up += 1
                added = True
        if not added:
            raise SystemExit(f"only {len(convs)} conversations of {a.pack} fit in {len(vids)} videos, asked for {num_convs}")
    # 6. repair the mix: swap turns of over-target types for unused QAs of under-target types of the same video
    unused = {v: [q for q in qas_of[v] if id(q) not in used] for v in vids}
    swaps, swapped = 0, True
    while swapped:
        swapped = False
        for v, cv in convs:
            for i, q in enumerate(cv):
                if got[q["t"]] <= target[q["t"]]:
                    continue
                rest = cv[:i] + cv[i + 1:]
                cands = [r for r in unused[v] if got[r["t"]] < target[r["t"]] and not any(conflict(r, o) for o in rest)]
                if cands:
                    r = max(cands, key=lambda r: (target[r["t"]] - got[r["t"]], all(o["t"] != r["t"] for o in rest), rng.random()))
                    cv[i] = r
                    unused[v].remove(r)
                    unused[v].append(q)
                    got[q["t"]] -= 1
                    got[r["t"]] += 1
                    swaps += 1
                    swapped = True
    # 7. replace whole conversations of over-target types by new ones from the unused QAs of any video, while that
    #    lowers the mix error; every video keeps at least one conversation and gets at most max(c)
    err = lambda cnt: sum(abs(cnt[t] - target[t]) for t in mix)

    def delta(rem, add):
        cnt = got.copy()
        cnt.subtract(q["t"] for q in rem)
        cnt.update(q["t"] for q in add)
        return err(cnt) - err(got)

    def candidate(w):
        need = collections.Counter({t: max(0, target[t] - got[t]) for t in mix})
        full = pack_video(unused[w], need, 1, a.pack, rank, rng, 5)
        return full[0] if full else None

    n_of, most = collections.Counter(v for v, _ in convs), max(c.values())
    cand, fresh, replaced = {w: candidate(w) for w in vids}, True, 0
    while True:
        # among equal gains, move conversations from videos with many to videos with few
        victims = sorted((i for i, (v, _) in enumerate(convs) if n_of[v] > 1),
                         key=lambda i: (delta(convs[i][1], []), -n_of[convs[i][0]]))[:20]
        adds = sorted((w for w in vids if cand[w] and n_of[w] < most), key=lambda w: (delta([], cand[w]), n_of[w]))[:20]
        best = min(((delta(convs[i][1], cand[w]), n_of[w] - n_of[convs[i][0]], i, w) for i in victims for w in adds),
                   key=lambda x: x[:2], default=None)
        if best is None or best[0] >= 0:
            if fresh:
                break
            cand, fresh = {w: candidate(w) for w in vids}, True
            continue
        *_, i, w = best
        (v, cv), new = convs[i], cand[w]
        unused[v] += cv
        unused[w] = [q for q in unused[w] if all(q is not r for r in new)]
        convs[i] = (w, new)
        got.subtract(q["t"] for q in cv)
        got.update(q["t"] for q in new)
        n_of[v] -= 1
        n_of[w] += 1
        cand[v], cand[w], fresh = candidate(v), candidate(w), False
        replaced += 1

    rng.shuffle(convs)
    name = f"{a.mix}_pack{a.pack}_{a.num_qas}{'_' + a.tag if a.tag else ''}"
    os.makedirs(a.out_dir, exist_ok=True)
    dst = os.path.join(a.out_dir, name + ".jsonl")
    seen = set()
    with open(dst, "w") as f:
        for v, cv in convs:
            rng.shuffle(cv)
            assert len(cv) == a.pack and not any(conflict(x, y) for i, x in enumerate(cv) for y in cv[i + 1:])
            assert not seen & {id(q) for q in cv}
            seen.update(id(q) for q in cv)
            conv = []
            for j, q in enumerate(cv):
                h, g = q["d"]["conversations"]
                conv += [{"from": "human", "value": h["value"] if j == 0 else h["value"][len(PREAMBLE):]},
                         {"from": "gpt", "value": g["value"]}]
            f.write(json.dumps({"video": v, "question_type": "packed", "question_types": [q["t"] for q in cv],
                                "conversations": conv}) + "\n")

    got = collections.Counter(q["t"] for _, cv in convs for q in cv)
    stats = {
        "args": vars(a), "videos": len({v for v, _ in convs}), "videos_with_qas": len(qas_of),
        "videos_too_small_for_a_conversation": sum(n == 0 for n in cap.values()),
        "qas_off_plan": off_plan, "conversations_topped_up": topped_up, "mix_repair_swaps": swaps,
        "conversations_replaced": replaced,
        "conversations": len(convs), "qas": sum(got.values()),
        "conversations_per_video": dict(sorted(collections.Counter(collections.Counter(v for v, _ in convs).values()).items())),
        "distinct_types_per_conversation": dict(sorted(collections.Counter(len({q["t"] for q in cv}) for _, cv in convs).items())),
        "types": {t: {"target": target[t], "got": got[t], "pct": round(100 * got[t] / a.num_qas, 1),
                      "available": sum(len(items[v][t]) for v in vids)} for t in mix},
        "skipped_qas_object_count_mismatch": dict(skipped),
    }
    with open(os.path.join(a.out_dir, name + ".stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, indent=1))
    print(dst)


if __name__ == "__main__":
    main()
