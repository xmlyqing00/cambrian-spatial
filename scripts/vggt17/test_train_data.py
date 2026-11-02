"""Train/eval input parity for VGGT-17 fine-tuning (CPU only, no model).

For each sample of a training jsonl, the train.py dataset item must match what the lmms-eval CambrianP wrapper feeds the
model for the same video and question: prompt token ids, frame pixels (decord linspace + pad-to-square + SigLIP
preprocessing) and VGGT tokens (selected by the eval frame indices). Also checks that only the answers are supervised
(every assistant turn of packed multi-turn samples, whose first turn is checked against eval).

  CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python scripts/vggt17/test_train_data.py --jsonl J --vggt_feat_dir D [--n 8]
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import transformers
from decord import VideoReader, cpu
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from cambrianp import conversation as conversation_lib  # noqa: E402
from cambrianp.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX  # noqa: E402
from cambrianp.mm_utils import expand2square, tokenizer_image_token  # noqa: E402
from cambrianp.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor  # noqa: E402
from cambrianp.model.multimodal_projector.vggt_token import load_vggt_tokens, vggt_feature_path  # noqa: E402
from cambrianp.train.train import DataArguments, ModelArguments, TrainingArguments, make_supervised_data_module  # noqa: E402


def build_train_dataset(jsonl, data_path, ckpt, vggt_feat_dir=None, frames=128, gather_mode="none"):
    """The train.py dataset with the data-side arguments of scripts/vggt17/run_train.sh; returns (dataset, tokenizer, data_args)."""
    args = [
        "--output_dir", "/tmp/vggt17_test_train_data", "--model_name_or_path", ckpt, "--version", "qwen_1_5",
        "--data_path", data_path, "--json_file", jsonl, "--lazy_preprocess", "True",
        "--frames_upbound", str(frames), "--force_sample", "True", "--image_aspect_ratio", "anyres_max_9",
        "--mm_use_im_start_end", "False", "--mm_use_im_patch_token", "False",
        "--camera_tokens_mode", "camera_tokens", "--camera_tokens_place", "append_to_frame", "--load_rec_model", "True",
        "--enable_camera", "True", "--enable_depth", "False", "--enable_point", "False",
        "--video_camera_tokens", "True", "--video_pad_to_square", "True",
    ]
    if vggt_feat_dir is not None:
        args += ["--use_vggt_token", "True", "--vggt_feat_dir", vggt_feat_dir, "--vggt_token_source", "group", "--vggt_gather_mode", gather_mode]
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args)
    # As in train(): data_args fields set from the other argument groups and the vision tower.
    data_args.force_no_bp_vqa = training_args.force_no_bp_vqa
    data_args.camera_tokens_place = model_args.camera_tokens_place
    data_args.vggt_gather_mode = model_args.vggt_gather_mode
    data_args.enable_camera = model_args.enable_camera
    data_args.enable_point = model_args.enable_point
    data_args.enable_depth = model_args.enable_depth
    data_args.image_processor = SigLipImageProcessor()
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt, model_max_length=32768, padding_side="right")
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    return make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)["train_dataset"], tokenizer, data_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--vggt_feat_dir", required=True)
    ap.add_argument("--data_path", default=os.path.expanduser("~/datasets/VSI-590K"))
    ap.add_argument("--ckpt", default=os.path.join(REPO, "ckpts/Cambrian-P-7B"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gather_mode", default="none", choices=("none", "lookat"))
    a = ap.parse_args()

    dataset, tokenizer, data_args = build_train_dataset(a.jsonl, a.data_path, a.ckpt, a.vggt_feat_dir, gather_mode=a.gather_mode)

    proc = data_args.image_processor
    background = tuple(int(x * 255) for x in proc.image_mean)
    for i in range(min(a.n, len(dataset))):
        entry = dataset.list_data_dict[i]
        item = dataset._get_item(i)  # not __getitem__, which swaps in other samples on errors
        video = os.path.join(a.data_path, entry["video"])

        # Prompt as built by the eval wrapper: "<image>\n" + question, qwen_1_5 template, generation prompt.
        context = entry["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        conv = conversation_lib.conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
        conv.append_message(conv.roles[1], None)
        eval_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        ids, labels = item["input_ids"], item["labels"]
        n = len(eval_ids)
        assert torch.equal(ids[:n], eval_ids), f"{i}: prompt differs from eval\n{tokenizer.decode(ids[:n].clamp_min(0))}"
        assert (ids == IMAGE_TOKEN_INDEX).sum() == 1
        # preprocess_qwen (LLaVA-OneVision convention) also supervises "\n", <|im_start|> and <|im_end|> in the prompt, and the
        # whole assistant turn, whose "<|im_start|>assistant\n" header ends the eval prompt.
        structural = {198, tokenizer.convert_tokens_to_ids("<|im_start|>"), tokenizer.convert_tokens_to_ids("<|im_end|>")}
        assert tokenizer.decode(labels[n - 3:n]) == "<|im_start|>assistant\n", f"{i}: assistant header not supervised"
        structural.add(labels[n - 2].item())
        prompt_supervised = set(labels[:n][labels[:n] != IGNORE_INDEX].tolist())
        assert prompt_supervised <= structural, f"{i}: prompt tokens supervised: {tokenizer.decode(list(prompt_supervised - structural))!r}"
        target = tokenizer.decode(labels[n:][labels[n:] != IGNORE_INDEX])
        answer = entry["conversations"][1]["value"]
        assert target.startswith(answer + "<|im_end|>"), f"{i}: supervised {target!r}, answer {answer!r}"
        # Every assistant turn (all turns of packed multi-turn samples) is supervised whole: "<|im_start|>assistant\n",
        # the answer and <|im_end|>; outside them only structural ids.
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        starts = [p for p in range(len(ids) - 2) if ids[p:p + 3].tolist() == header]
        answers = [m["value"] for m in entry["conversations"][1::2]]
        assert len(starts) == len(answers) and starts[0] == n - 3, f"{i}: {len(starts)} assistant turns, {len(answers)} answers"
        in_turn = torch.zeros(len(ids), dtype=torch.bool)
        for p, ans in zip(starts, answers):
            end = p + 3 + (ids[p + 3:] == im_end).nonzero()[0].item()
            assert torch.equal(labels[p:end + 1], ids[p:end + 1]), f"{i}: assistant turn at {p} not fully supervised"
            assert tokenizer.decode(ids[p + 3:end]) == ans, f"{i}: turn at {p} is {tokenizer.decode(ids[p + 3:end])!r}, answer {ans!r}"
            in_turn[p:end + 1] = True
        outside = set(labels[~in_turn & (labels != IGNORE_INDEX)].tolist())
        assert outside <= structural, f"{i}: non-answer tokens supervised: {tokenizer.decode(list(outside - structural))!r}"

        # Frames as loaded and preprocessed by the eval wrapper.
        vr = VideoReader(video, ctx=cpu(0))
        frame_idx = np.linspace(0, len(vr) - 1, 128, dtype=int).tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [expand2square(Image.fromarray(f, mode="RGB"), background) for f in frames]
        eval_pixels = proc.preprocess(frames, return_tensors="pt")["pixel_values"]
        pixels, _, kind = item["image"][0]
        assert kind == "video" and pixels.shape == eval_pixels.shape == (128, 3, 384, 384), (kind, pixels.shape)
        pix_diff = (pixels - eval_pixels).abs().max().item()
        assert pix_diff == 0, f"{i}: pixels differ from eval by {pix_diff}"

        eval_features = load_vggt_tokens(vggt_feature_path(a.vggt_feat_dir, video, video_root=a.data_path), "group", frame_idx=frame_idx, gather_mode=a.gather_mode)
        if isinstance(eval_features, dict):
            for key, value in eval_features.items():
                torch.testing.assert_close(item["vggt_feats"][key], torch.from_numpy(value), rtol=0, atol=0, equal_nan=True, msg=f"{i}: {key} differs from eval")
            train_tokens = item["vggt_feats"]["tokens"]
        else:
            assert torch.equal(item["vggt_feats"], torch.from_numpy(eval_features)), f"{i}: VGGT tokens differ from eval"
            train_tokens = item["vggt_feats"]
        assert item["bp_rec"] is True and "rec_views" not in item

        print(f"[{i}] {entry['video']} {entry.get('question_type')}: prompt {n} tok (= eval, {len(prompt_supervised)} structural ids supervised), "
              f"{len(answers)} supervised answer turn(s) {answers}, "
              f"pixels (128,3,384,384) = eval, VGGT {tuple(train_tokens.shape)} {train_tokens.dtype} = eval, gather={a.gather_mode}")
    print("ALL OK")


if __name__ == "__main__":
    main()
