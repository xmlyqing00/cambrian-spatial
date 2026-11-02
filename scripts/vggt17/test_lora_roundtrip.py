"""The lmms-eval Cambrian-P wrapper loads a VGGT-17 LoRA adapter exactly: merged LLM weights = base + B A * alpha / r (to fp16
rounding), and the trained VGGT projector (weights and feature stats) replaces the freshly initialised one.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/lmms-eval:$PWD python scripts/vggt17/test_lora_roundtrip.py --lora output/vggt17/smoke/vggt
"""

import argparse
import json
import os

import torch
from safetensors import safe_open

from cambrianp.constants import DEFAULT_IMAGE_TOKEN
from lmms_eval.api.instance import Instance
from lmms_eval.models.simple.cambrian_p import CambrianP

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(REPO, "ckpts/Cambrian-P-7B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--vggt_feat_dir", default="/data/dataset/vggt17_feats/vsibench")
    ap.add_argument("--smoke_jsonl", help="also generate a short answer to the first sample using the real evaluation wrapper")
    ap.add_argument("--data_path", default=os.path.expanduser("~/datasets/VSI-590K"))
    a = ap.parse_args()

    adapter = torch.load(os.path.join(a.lora, "adapter_model.bin"), map_location="cpu", weights_only=True)
    with open(os.path.join(a.lora, "adapter_config.json")) as f:
        scale = (cfg := json.load(f))["lora_alpha"] / cfg["r"]
    use_vggt = any("vggt_projector." in k for k in adapter)
    lm = CambrianP(
        pretrained=a.ckpt,
        conv_template="qwen_1_5",
        model_name="llava_qwen",
        max_frames_num=128,
        use_camera_tokens=True,
        camera_tokens_mode="camera_tokens",
        camera_tokens_place="append_to_frame",
        query_mode="query_after_image",
        vggt_feat_dir=a.vggt_feat_dir if use_vggt else None,
        lora_path=os.path.abspath(a.lora),
    )
    model = lm.model
    assert not any("lora_" in n for n, _ in model.named_parameters()), "LoRA layers left unmerged"

    if use_vggt:
        proj = model.get_model().vggt_projector.state_dict()
        names = sorted(k.split("vggt_projector.")[1] for k in adapter if "vggt_projector." in k)
        assert names == sorted(proj), (names, sorted(proj))
        for name in names:
            assert torch.equal(proj[name].float().cpu(), adapter[f"base_model.model.model.vggt_projector.{name}"].float()), name
        with open(os.path.join(a.lora, "config.json")) as f:
            saved_config = json.load(f)
        projector = model.get_model().vggt_projector
        for attribute, key, default in (
            ("gather_mode", "vggt_gather_mode", "none"),
            ("gather_radius", "vggt_gather_radius", 0.15),
            ("gather_scope", "vggt_gather_scope", "all"),
            ("gather_far_gap", "vggt_gather_far_gap", 32),
        ):
            assert getattr(projector, attribute) == saved_config.get(key, default), key
        print(f"VGGT projector = adapter ({', '.join(names)}); output-layer rms {proj['mlp.2.weight'].float().pow(2).mean().sqrt():.3e}")

    with open(os.path.join(a.ckpt, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    n_layers = model.config.num_hidden_layers
    for layer in (0, n_layers // 2, n_layers - 1):
        for mod in ("self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj"):
            key = f"model.layers.{layer}.{mod}"
            with safe_open(os.path.join(a.ckpt, weight_map[f"{key}.weight"]), "pt") as f:
                base = f.get_tensor(f"{key}.weight").float()
            delta = adapter[f"base_model.model.{key}.lora_B.weight"].float() @ adapter[f"base_model.model.{key}.lora_A.weight"].float() * scale
            merged = model.get_submodule(key).weight.float().cpu()
            err = (merged - (base + delta)).abs().max().item()
            tol = (base + delta).abs().max().item() * 2**-10  # fp16 rounding of the merged weight
            assert delta.abs().max() > 0, f"{key}: zero LoRA delta"
            assert err <= tol, f"{key}: merged weight off by {err:.2e} (tol {tol:.2e}, |delta| max {delta.abs().max():.2e})"
            print(f"{key}: merged = base + BA*{scale:g} (max err {err:.1e} <= {tol:.1e}; |delta| max {delta.abs().max():.1e})")
    if a.smoke_jsonl:
        with open(a.smoke_jsonl) as f:
            sample = json.loads(next(line for line in f if line.strip()))
        context = sample["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        model.config.vision_tower_chunk_size = 32
        lm.task_dict = {"adapter_smoke": {"val": [sample]}}
        request = Instance(
            request_type="generate_until",
            arguments=(context, {"max_new_tokens": 4}, lambda doc: [os.path.join(a.data_path, doc["video"])], 0, "adapter_smoke", "val"),
            idx=0,
            metadata={"task": "adapter_smoke", "doc_id": 0, "repeats": 1},
        )
        prediction = lm.generate_until([request])
        assert len(prediction) == 1 and prediction[0].strip(), f"Empty generation: {prediction}"
        if use_vggt and projector.gather_mode == "lookat":
            assert projector.last_gather_stats is not None, "No look-at gathering during evaluation"
            print(f"Reloaded-adapter gathering: {projector.last_gather_stats}")
        print(f"Reloaded-adapter generation on {sample['video']}: {prediction}")
        print(f"Evaluation peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("ALL OK")


if __name__ == "__main__":
    main()
