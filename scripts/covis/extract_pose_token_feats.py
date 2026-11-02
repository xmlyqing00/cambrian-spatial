#!/usr/bin/env python
"""Dump frozen Cambrian-P per-frame features for co-visibility probing (milestone M0).

Loads the checkpoint through the lmms-eval `cambrian_p` wrapper (same loading, frame sampling and
preprocessing as VSI-Bench evaluation with use_camera_tokens=true, append_to_frame), runs one
prefill forward pass, and saves one .npz per video with:

  frame_idx [N]          decoded frame indices (must match the DA3 label file)
  layers    [L]          LLM layers the features below were taken from (0 = input embeddings)
  cam_h     [L,N,C]      hidden state of each frame's camera token (fp16)
  img_h     [L,N,C]      mean-pooled 8x8 image tokens of each frame (fp16, newline tokens excluded)
  cam_proj  [N,2048]     rec-projector output for the camera token (camera-head input, fp16)
  cam_trunk [N,2048]     camera-head trunk output at the last refinement iteration (fp16)
  pose_enc  [N,9]        predicted pose encoding (absT_quaR_FoV, relative to frame 0)

With append_to_frame the camera tokens precede the question in a causal LLM, so these features do
not depend on the prompt text.

--no_camera_tokens: for checkpoints without camera tokens / pose head (e.g. the base models
Cambrian-S-7B and Cambrian-S-7B-S3, loaded through the same cast-and-remap path). Only frame_idx,
layers, img_h and seq_len are saved.

Run in the `cambrianp` env with PYTHONPATH=$PWD/lmms-eval:$PWD, e.g.
  python scripts/covis/extract_pose_token_feats.py --video_list list.txt --video_root ROOT --out_dir OUT
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

PROMPT = "These are frames of a video."


def build_input_ids(tokenizer, conv_template):
    from cambrianp.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from cambrianp.conversation import conv_templates
    from cambrianp.mm_utils import tokenizer_image_token

    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + PROMPT)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[None]


def image_tokens_without_camera(h, input_ids, n_tokens, n_frames, grid):
    """Per-frame 8x8 image tokens [N, grid*grid, C] for sequences without camera tokens.

    The video occupies the IMAGE_TOKEN position of the prompt, frame-major, as grid rows of
    (grid image tokens + 1 newline token).
    """
    from cambrianp.constants import IMAGE_TOKEN_INDEX

    pre = int((input_ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0, 0])
    n_img = n_tokens - (input_ids.shape[1] - 1)
    assert n_img == n_frames * grid * (grid + 1), f"unexpected video token layout: {n_img} tokens, {n_frames} frames"
    return h[0, pre : pre + n_img].view(n_frames, grid, grid + 1, -1)[:, :, :grid].reshape(n_frames, grid * grid, -1)


@torch.inference_mode()
def extract(lm, input_ids, path, num_frames, layers, use_camera_tokens=True):
    from lmms_eval.models.simple.cambrian_p import expand2square
    from cambrianp.model.reconstructor.vggt_reconstructor import VGGTReconstructor

    model = lm.model
    proc = lm._image_processor
    t0 = time.time()
    frames = lm.load_video(path, num_frames)
    bg = tuple(int(x * 255) for x in proc.image_mean)
    pil = [expand2square(Image.fromarray(f, mode="RGB"), bg) for f in frames]
    pix = proc.preprocess(pil, return_tensors="pt")["pixel_values"].half().cuda()
    t_dec = time.time() - t0

    ids = input_ids.cuda()
    (_, position_ids, attention_mask, _, embeds, _, info) = model.prepare_inputs_labels_for_multimodal(
        ids, None, torch.ones_like(ids, dtype=torch.bool), None, None, [pix], ["video"],
        image_sizes=list(frames.shape[1:3]),
    )
    out = model.get_model()(
        inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids,
        output_hidden_states=True, return_dict=True, use_cache=False,
    )
    hs = out.hidden_states
    grid = int(round(model.config.mm_img_tok_num ** 0.5))
    f16 = lambda t: t.float().cpu().numpy().astype(np.float16)
    frame_idx = np.linspace(0, lm._last_total_frames - 1, num_frames, dtype=int).astype(np.int32)
    if not use_camera_tokens:
        img_h = [
            image_tokens_without_camera(hs[li], ids, embeds.shape[1], len(frames), grid).float().mean(1)
            for li in layers
        ]
        res = dict(
            frame_idx=frame_idx,
            layers=np.array(layers, np.int32),
            img_h=np.stack([f16(x) for x in img_h]),
            seq_len=np.int32(embeds.shape[1]),
        )
        return res, dict(decode_s=t_dec, total_s=time.time() - t0, seq=embeds.shape[1])

    cam_h, img_h = [], []
    for li in layers:
        tok = VGGTReconstructor.prepare_rec_tokens_for_single_layer(
            info, hs, layer_idx=li, grid_size=grid, need_camera_token=True
        )[0]  # [N, 1 + grid*grid, C], camera token first
        cam_h.append(tok[:, 0])
        img_h.append(tok[:, 1:].float().mean(1))

    rec = model.get_model().rec_head
    proj_out, trunk_out = [], []
    hooks = [
        rec.projector_list[-1].register_forward_hook(lambda m, i, o: proj_out.append(o)),
        rec.camera_head.trunk.register_forward_hook(lambda m, i, o: trunk_out.append(o)),
    ]
    try:
        preds = rec(out, info_dict_list=info, batch={"images": None})
    finally:
        for h in hooks:
            h.remove()

    res = dict(
        frame_idx=frame_idx,
        layers=np.array(layers, np.int32),
        cam_h=np.stack([f16(x) for x in cam_h]),
        img_h=np.stack([f16(x) for x in img_h]),
        cam_proj=f16(proj_out[-1][0, :, 0]),
        cam_trunk=f16(trunk_out[-1][0]),
        pose_enc=preds["pose_enc"][0].float().cpu().numpy(),
        seq_len=np.int32(embeds.shape[1]),
    )
    return res, dict(decode_s=t_dec, total_s=time.time() - t0, seq=embeds.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_list", required=True, help="text file: one video path per line (relative to --video_root)")
    ap.add_argument("--video_root", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="ckpts/Cambrian-P-7B")
    ap.add_argument("--conv_template", default="qwen_1_5")
    ap.add_argument("--num_frames", type=int, default=128)
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28", help="LLM hidden-state indices (0 = embeddings)")
    ap.add_argument("--shard", type=str, default="0/1", help="i/n: process every n-th video starting at i")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_camera_tokens", action="store_true", help="checkpoint without camera tokens (base Cambrian-S)")
    args = ap.parse_args()
    use_cam = not args.no_camera_tokens

    from decord import VideoReader, cpu
    from lmms_eval.models.simple.cambrian_p import CambrianP

    lm = CambrianP(
        pretrained=os.path.abspath(args.ckpt), conv_template=args.conv_template, model_name="llava_qwen",
        max_frames_num=args.num_frames, use_camera_tokens=use_cam, camera_tokens_mode="camera_tokens",
        camera_tokens_place="append_to_frame", query_mode="query_after_image",
    )
    lm._config.mm_spatial_pool_stride = lm.mm_spatial_pool_stride
    lm._config.mm_spatial_pool_mode = lm.mm_spatial_pool_mode
    input_ids = build_input_ids(lm.tokenizer, args.conv_template)
    layers = [int(x) for x in args.layers.split(",")]

    vids = [l.strip() for l in open(args.video_list) if l.strip()]
    si, sn = map(int, args.shard.split("/"))
    vids = vids[si::sn]
    if args.limit:
        vids = vids[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    for k, rel in enumerate(vids):
        key = rel.replace("/", "__").rsplit(".", 1)[0]
        dst = os.path.join(args.out_dir, key + ".npz")
        if os.path.exists(dst):
            continue
        path = os.path.join(args.video_root, rel)
        try:
            lm._last_total_frames = len(VideoReader(path, ctx=cpu(0)))
            res, info = extract(lm, input_ids, path, args.num_frames, layers, use_camera_tokens=use_cam)
        except Exception as e:  # keep going on corrupt videos, but report them
            print(f"[{k+1}/{len(vids)}] FAIL {rel}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        tmp = dst + ".tmp.npz"
        np.savez(tmp, video=rel, **res)
        os.replace(tmp, dst)
        print(
            f"[{k+1}/{len(vids)}] {rel} seq={info['seq']} dec={info['decode_s']:.1f}s tot={info['total_s']:.1f}s "
            f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}G",
            flush=True,
        )


if __name__ == "__main__":
    main()
