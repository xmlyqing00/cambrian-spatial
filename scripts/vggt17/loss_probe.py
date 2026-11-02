"""How well does the base Cambrian-P already fit its own training data (VSI-590K), per question type?

Cambrian-P was trained on VSI-590K, so fine-tuning on VSI-590K QAs only gives signal where the model does not fit them yet.
For a seeded sample of --per_type QAs per question type from the VGGT-17 training-manifest videos (no VSI-Bench scenes),
with the eval wrapper's model (fp16, sdpa, 128 frames, CAM tokens) and the train.py dataset items (= eval inputs, see
test_train_data.py), this records:
  - teacher-forced cross-entropy of the answer (answer tokens + <|im_end|>, i.e. the training loss on the answer) and
    exact match (argmax = target at every answer position);
  - the greedy answer as eval decodes it (max 16 new tokens, eos, repetition penalty of generation_config.json), scored
    as VSI-Bench does: accuracy for multiple choice, MRA (.5:.95:.05) for numbers.

  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD/lmms-eval:$PWD python scripts/vggt17/loss_probe.py --out output/vggt17/loss_probe
  # -> <out>/probe.jsonl (sampled QAs), probe_results.jsonl (one line per QA, appended; reruns resume), summary.json
"""

import argparse
import collections
import json
import os
import random
import time

import numpy as np
import torch
from transformers import GenerationConfig

from test_train_data import REPO, build_train_dataset  # scripts/vggt17 is sys.path[0]; also puts REPO on sys.path

from cambrianp.constants import IGNORE_INDEX  # noqa: E402
from lmms_eval.models.simple.cambrian_p import CambrianP  # noqa: E402
from lmms_eval.tasks.vsibench.utils import exact_match, fuzzy_matching, mean_relative_accuracy, to_float  # noqa: E402

# VSI-590K question type -> VSI-Bench task of the same kind
BENCH = {"relative_direction_object": "object_rel_direction", "relative_distance_object": "object_rel_distance",
         "absolute_distance_object": "object_abs_distance", "absolute_size_object": "object_size_estimation",
         "absolute_count": "object_counting", "absolute_size_room": "room_size_estimation", "appearance_order": "obj_appearance_order"}


def sample_qas(jsonl, videos_txt, per_type, seed, dst):
    with open(videos_txt) as f:
        videos = set(f.read().split())
    pools = collections.defaultdict(list)
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            if d.get("video") in videos:
                pools[d["question_type"]].append(d)
    rng = random.Random(seed)
    picked = []
    for qtype in sorted(pools):
        for k, d in enumerate(rng.sample(pools[qtype], min(per_type, len(pools[qtype])))):
            picked.append(dict(d, probe_id=f"{qtype}/{k}"))
    with open(dst, "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in picked)
    print(f"{dst}: {len(picked)} QAs, {len({d['video'] for d in picked})} videos; pool sizes "
          f"{ {t: len(p) for t, p in sorted(pools.items())} }", flush=True)


def score(question, answer, pred):
    """VSI-Bench scoring: accuracy on the option letter for multiple choice, MRA for numbers."""
    if "Options:" in question and len(answer) == 1 and answer.isalpha():
        return "accuracy", exact_match(fuzzy_matching(pred), answer)
    if to_float(answer) is not None:
        p = to_float(fuzzy_matching(pred))
        return "MRA", 0.0 if p is None else float(mean_relative_accuracy(p, to_float(answer), start=.5, end=.95, interval=.05))
    return "accuracy", exact_match(fuzzy_matching(pred), answer)


class Items(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        try:
            return i, self.dataset._get_item(i)  # not __getitem__, which swaps in another sample on errors
        except Exception as e:  # noqa: BLE001
            return i, f"{type(e).__name__}: {e}"


def identity(x):
    return x


@torch.inference_mode()
def probe(model, tokenizer, item, gen):
    ids, labels = item["input_ids"], item["labels"]
    im_start, im_end = tokenizer.convert_tokens_to_ids("<|im_start|>"), tokenizer.convert_tokens_to_ids("<|im_end|>")
    n = (ids == im_start).nonzero()[-1].item() + 3  # eval prompt = everything up to "<|im_start|>assistant\n"
    assert tokenizer.decode(ids[n - 3:n]) == "<|im_start|>assistant\n"
    assert (labels[n:] != IGNORE_INDEX).all() and torch.equal(labels[n:], ids[n:]) and labels[-2:].tolist() == [im_end, 198]
    tgt = labels[n:-1].cuda()  # answer tokens + <|im_end|>; the final "\n" is outside the answer
    m = len(tgt)
    prompt = ids[:n][None].cuda()
    pixels, size, kind = item["image"][0]
    assert kind == "video"
    _, pos, mask, _, embeds, _, _ = model.prepare_inputs_labels_for_multimodal(
        prompt, None, torch.ones_like(prompt, dtype=torch.bool), None, None, [pixels.half().cuda()], ["video"], image_sizes=[size])
    seq = embeds.shape[1]
    embed = model.get_model().embed_tokens
    out = model.model(inputs_embeds=embeds, attention_mask=mask, position_ids=pos, use_cache=True, return_dict=True)
    past = out.past_key_values  # legacy tuple: later calls build new caches from it, it is never modified
    first = model.lm_head(out.last_hidden_state[0, -1]).float()

    # Teacher forcing: the gold answer after the prompt.
    cont = model.model(inputs_embeds=embed(tgt[None, :-1]), past_key_values=past, use_cache=True, return_dict=True,
                       position_ids=torch.arange(seq, seq + m - 1, device="cuda")[None])
    logits = torch.cat([first[None], model.lm_head(cont.last_hidden_state[0]).float()])
    del cont
    ce = -logits.log_softmax(-1)[torch.arange(m), tgt]
    p_first = first.softmax(-1)
    top = p_first.topk(3)

    # Greedy decoding as model.generate() in the eval wrapper (temperature 0, do_sample False).
    gen_ids, step_logits = [], first
    for step in range(gen.max_new_tokens):
        if gen_ids and gen.repetition_penalty != 1.0:
            seen = torch.tensor(sorted(set(gen_ids)), device="cuda")
            s = step_logits[seen]
            step_logits = step_logits.clone()
            step_logits[seen] = torch.where(s < 0, s * gen.repetition_penalty, s / gen.repetition_penalty)
        nxt = int(step_logits.argmax())
        gen_ids.append(nxt)
        if nxt in gen.eos or len(gen_ids) == gen.max_new_tokens:
            break
        o = model.model(inputs_embeds=embed(torch.tensor([[nxt]], device="cuda")), past_key_values=past, use_cache=True,
                        position_ids=torch.tensor([[seq + step]], device="cuda"), return_dict=True)
        past = o.past_key_values
        step_logits = model.lm_head(o.last_hidden_state[0, -1]).float()
    return dict(
        prompt_len=n, seq_len=seq, n_target=m, ce=ce.mean().item(), ce_first=ce[0].item(), ce_tokens=[round(x, 4) for x in ce.tolist()],
        tf_em=bool((logits.argmax(-1) == tgt).all()), p_first_gold=p_first[tgt[0]].item(),
        top_first=[[tokenizer.decode([int(j)]), round(float(p), 4)] for p, j in zip(top.values, top.indices)],
        pred=tokenizer.decode(gen_ids, skip_special_tokens=True).strip())


def summarize(rows):
    def stats(rs):
        ce = np.array([r["ce"] for r in rs])
        metric = collections.Counter(r["metric"] for r in rs).most_common(1)[0][0]
        return dict(n=len(rs), bench=BENCH.get(rs[0]["question_type"], "-") if len({r["question_type"] for r in rs}) == 1 else "-",
                    metric=metric, score=100 * float(np.mean([r["score"] for r in rs])), ce_mean=float(ce.mean()),
                    ce_median=float(np.median(ce)), ce_first_mean=float(np.mean([r["ce_first"] for r in rs])),
                    tf_em=100 * float(np.mean([r["tf_em"] for r in rs])), frac_ce_gt_0_1=float((ce > 0.1).mean()))
    by_type = collections.defaultdict(list)
    by_source = collections.defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
        by_source[r["video"].split("/")[0]].append(r)
    return dict(by_type={t: stats(rs) for t, rs in sorted(by_type.items())},
                by_source={s: dict(n=len(rs), ce_mean=float(np.mean([r["ce"] for r in rs])),
                                   score=100 * float(np.mean([r["score"] for r in rs]))) for s, rs in sorted(by_source.items())},
                all=dict(n=len(rows), ce_mean=float(np.mean([r["ce"] for r in rows])),
                         score_mean_over_types=float(np.mean([stats(rs)["score"] for rs in by_type.values()]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "output/vggt17/loss_probe"))
    ap.add_argument("--jsonl", default="/data/dataset/VSI-590K/vsi_590k.jsonl")
    ap.add_argument("--videos", default="/data/dataset/vggt17_feats/train_manifest/videos.txt")
    ap.add_argument("--data_path", default=os.path.expanduser("~/datasets/VSI-590K"))
    ap.add_argument("--ckpt", default=os.path.join(REPO, "ckpts/Cambrian-P-7B"))
    ap.add_argument("--per_type", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--vt_chunk", type=int, default=32, help="vision tower frames per forward (bit-identical to unchunked)")
    ap.add_argument("--check_generate", type=int, default=0, help="for the first N QAs, assert pred = the eval wrapper's generate()")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    qa_file, res_file = os.path.join(a.out, "probe.jsonl"), os.path.join(a.out, "probe_results.jsonl")
    if not os.path.exists(qa_file):
        sample_qas(a.jsonl, a.videos, a.per_type, a.seed, qa_file)

    dataset, tokenizer, _ = build_train_dataset(qa_file, a.data_path, a.ckpt)
    done = set()
    if os.path.exists(res_file):
        with open(res_file) as f:
            done = {json.loads(line)["probe_id"] for line in f}
    todo = [i for i, d in enumerate(dataset.list_data_dict) if d["probe_id"] not in done]
    print(f"{len(dataset)} QAs, {len(done)} done, {len(todo)} to go", flush=True)

    lm = CambrianP(pretrained=a.ckpt, conv_template="qwen_1_5", model_name="llava_qwen", max_frames_num=128, use_camera_tokens=True,
                   camera_tokens_mode="camera_tokens", camera_tokens_place="append_to_frame", query_mode="query_after_image")
    model = lm.model
    lm._config.mm_spatial_pool_stride, lm._config.mm_spatial_pool_mode = lm.mm_spatial_pool_stride, lm.mm_spatial_pool_mode
    model.config.vision_tower_chunk_size = a.vt_chunk
    gc = GenerationConfig.from_pretrained(a.ckpt)
    gen = argparse.Namespace(max_new_tokens=16, repetition_penalty=gc.repetition_penalty,  # lmms-eval vsibench gen_kwargs
                             eos={tokenizer.convert_tokens_to_ids("<|im_end|>"), *np.atleast_1d(gc.eos_token_id).tolist()})
    print(f"greedy decoding: max_new_tokens {gen.max_new_tokens}, repetition_penalty {gen.repetition_penalty}, eos {sorted(gen.eos)}", flush=True)

    loader = torch.utils.data.DataLoader(Items(dataset, todo), batch_size=None, num_workers=a.workers, collate_fn=identity, prefetch_factor=2)
    t0 = time.time()
    with open(res_file, "a") as f:
        for k, (i, item) in enumerate(loader):
            d = dataset.list_data_dict[i]
            question, answer = d["conversations"][0]["value"], d["conversations"][1]["value"]
            row = dict(probe_id=d["probe_id"], question_type=d["question_type"], video=d["video"], answer=answer)
            if isinstance(item, str):
                row.update(error=item)
            else:
                row.update(probe(model, tokenizer, item, gen))
                row["metric"], row["score"] = score(question, answer, row["pred"])
                if k < a.check_generate:
                    n = row["prompt_len"]
                    pixels, size, _ = item["image"][0]
                    prompt = item["input_ids"][:n][None].cuda()
                    ref = lm._run_inference_step(prompt, torch.ones_like(prompt, dtype=torch.bool), [pixels.half().cuda()], [size], "video",
                                                 dict(max_new_tokens=16, temperature=0, top_p=1.0, num_beams=1, do_sample=False), None)[0]
                    assert ref == row["pred"], f"{d['probe_id']}: probe greedy {row['pred']!r} != eval generate {ref!r}"
                    print(f"check_generate {d['probe_id']}: pred {ref!r} = eval generate()", flush=True)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if k % 20 == 0 or k == len(todo) - 1:
                print(f"[{k + 1}/{len(todo)}] {(time.time() - t0) / (k + 1):.1f} s/QA, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB | "
                      f"{row['question_type']}: gold {answer!r} pred {row.get('pred')!r} ce {row.get('ce', float('nan')):.3f} {row.get('error', '')}", flush=True)

    with open(res_file) as f:
        rows = [json.loads(line) for line in f]
    errors = [r for r in rows if "error" in r]
    summary = summarize([r for r in rows if "error" not in r])
    summary["errors"] = len(errors)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n{'question type':28s} {'VSI-Bench task':22s} {'n':>3s} {'metric':>8s} {'score':>6s} {'CE mean':>8s} {'CE med':>7s} "
          f"{'CE 1st':>7s} {'TF-EM':>6s} {'CE>0.1':>6s}")
    for t, s in summary["by_type"].items():
        print(f"{t:28s} {s['bench']:22s} {s['n']:3d} {s['metric']:>8s} {s['score']:6.1f} {s['ce_mean']:8.4f} {s['ce_median']:7.4f} "
              f"{s['ce_first_mean']:7.4f} {s['tf_em']:6.1f} {s['frac_ce_gt_0_1']:6.2f}")
    print(f"by source: {json.dumps(summary['by_source'])}\nall: {json.dumps(summary['all'])}, errors {len(errors)}")


if __name__ == "__main__":
    main()
