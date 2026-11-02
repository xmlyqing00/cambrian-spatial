"""Checks the VGGT-17 token hook in eval mode, through the lmms-eval Cambrian-P wrapper:
sequence layout [img 72, CAM, VGGT] per frame, rec-token slicing, and that the token reaches the LLM.

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/lmms-eval:$PWD python scripts/vggt17/test_vggt_hook.py
"""

import os

import numpy as np
import torch
from PIL import Image

from cambrianp.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from cambrianp.conversation import conv_templates
from cambrianp.mm_utils import tokenizer_image_token
from cambrianp.model.reconstructor.vggt_reconstructor import VGGTReconstructor
from lmms_eval.models.simple.cambrian_p import CambrianP, expand2square

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO = os.path.expanduser("~/.cache/huggingface/vsibench/scannet/scene0011_00.mp4")
FEATS = "/data/dataset/vggt17_feats"

lm = CambrianP(pretrained=f"{REPO}/ckpts/Cambrian-P-7B", conv_template="qwen_1_5", model_name="llava_qwen", max_frames_num=128,
               use_camera_tokens=True, camera_tokens_mode="camera_tokens", camera_tokens_place="append_to_frame", query_mode="query_after_image",
               vggt_feat_dir=f"{FEATS}/vsibench", vggt_token_stats=f"{FEATS}/stats_eval_group.npz")
model, proc = lm.model, lm._image_processor
proj = model.get_model().vggt_projector

lm._config.mm_spatial_pool_stride, lm._config.mm_spatial_pool_mode = lm.mm_spatial_pool_stride, lm.mm_spatial_pool_mode
frames = lm.load_video([VIDEO], 128)
pix = proc.preprocess([expand2square(Image.fromarray(f), tuple(int(x * 255) for x in proc.image_mean)) for f in frames], return_tensors="pt")["pixel_values"].half().cuda()
feats = lm._load_vggt_feats(VIDEO, len(frames))
conv = conv_templates["qwen_1_5"].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nHow many chairs are in this room? Answer with a single number.")
conv.append_message(conv.roles[1], None)
input_ids = tokenizer_image_token(conv.get_prompt(), lm.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
pad_id = lm.tokenizer.pad_token_id if lm.tokenizer.pad_token_id is not None else lm.tokenizer.eos_token_id


def prepare(vggt):
    _, pos, mask, _, embeds, _, info = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, input_ids.ne(pad_id), None, None, [pix], ["video"], image_sizes=list(frames.shape[1:3]),
        vggt_feats=[feats] if vggt else None)
    return pos, mask, embeds, info


def run_llm(pos, mask, embeds, keep=False):
    """Prefill as in generate(); logits for the last position only (full-sequence logits would cost ~9 GB)."""
    out = model.model(inputs_embeds=embeds, attention_mask=mask, position_ids=pos, output_hidden_states=keep, return_dict=True)
    return (out if keep else None), model.lm_head(out.last_hidden_state[0, -1]).float()


with torch.inference_mode():
    pos0, mask0, emb0, info0 = prepare(False)
    pos1, mask1, emb1, info1 = prepare(True)
    i0, i1 = info0[0], info1[0]
    pre, F = i1["pre_image_token_length"], 128
    assert emb1.shape[1] - emb0.shape[1] == F, (emb0.shape, emb1.shape)
    assert i0["image_token_length"] == F * 73 and i1["image_token_length"] == F * 74 and i1["vggt_token_count"] == 1
    img0, img1 = emb0[0, pre:pre + F * 73].view(F, 73, -1), emb1[0, pre:pre + F * 74].view(F, 74, -1)
    assert torch.equal(img1[:, :73], img0), "image + CAM tokens changed"
    assert torch.equal(emb1[0, :pre], emb0[0, :pre]) and torch.equal(emb1[0, pre + F * 74:], emb0[0, pre + F * 73:]), "text changed"
    assert torch.equal(img1[:, 73], proj(feats).to(img1.dtype)), "VGGT token should be the projector output"
    print(f"layout OK: {emb0.shape[1]} -> {emb1.shape[1]} tokens, [img 72, CAM, VGGT] x {F} frames, "
          f"fresh VGGT token L2 norm median {img1[:, 73].float().norm(dim=-1).median().item():.2f}")

    out1, logit1 = run_llm(pos1, mask1, emb1, keep=True)
    rec = VGGTReconstructor.prepare_rec_tokens_for_single_layer(info1, out1.hidden_states, layer_idx=-1, grid_size=8)
    h = out1.hidden_states[-1][0, pre:pre + F * 74].view(F, 74, -1)
    assert rec.shape[:3] == (1, F, 65)
    assert torch.equal(rec[0, :, 0], h[:, 72]), "camera token slice"
    assert torch.equal(rec[0, :, 1:], h[:, :72].reshape(F, 8, 9, -1)[:, :, :8].reshape(F, 64, -1)), "image token slice"
    model.get_model().rec_head(out1, info_dict_list=info1, batch={"images": torch.randn(1, 32, 3, 192, 192, dtype=torch.bfloat16, device="cuda")})
    print("rec slicing OK: CAM = position 72 of each frame, VGGT token excluded; rec_head forward OK")
    del out1, rec, h
    torch.cuda.empty_cache()

    _, logit0 = run_llm(pos0, mask0, emb0)
    d_init = (logit1 - logit0).abs().max().item()
    torch.nn.init.zeros_(proj.mlp[2].weight)
    torch.nn.init.zeros_(proj.mlp[2].bias)
    pos2, mask2, emb2, _ = prepare(True)
    _, logit2 = run_llm(pos2, mask2, emb2)
    d_zero = (logit2 - logit0).abs().max().item()
    tok = lambda l: repr(lm.tokenizer.decode(l.argmax()))
    print(f"last-token logits max|diff| vs no-token: fresh projector {d_init:.3f}, zero token {d_zero:.3f}; "
          f"argmax {tok(logit0)} / {tok(logit1)} / {tok(logit2)}")
    assert d_init > d_zero, "a non-zero VGGT token should move the logits more than a zero token"
print("ALL OK")
