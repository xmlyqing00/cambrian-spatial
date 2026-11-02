"""train.py plus per-phase CUDA memory prints (debug aid: TRAIN_PY=scripts/vggt17/train_memprobe.py scripts/vggt17/run_train.sh ...).

Prints the vision-tower peak and the LLM input dtype for every forward; the trainer's per-step peak_mem_gb then covers the
LLM forward/backward and optimizer step only, since the vision probe resets the peak counter.
"""

import torch

from cambrianp.model.llava_arch import LlavaMetaForCausalLM
from cambrianp.train.train import train

GB = 2**30
_encode_images = LlavaMetaForCausalLM.encode_images
_prepare = LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal


def encode_images(self, images):
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = _encode_images(self, images)
    print(f"[memprobe] vision tower {tuple(images.shape)}: {before / GB:.2f} GB before, peak {torch.cuda.max_memory_allocated() / GB:.2f} GB", flush=True)
    torch.cuda.reset_peak_memory_stats()
    return out


def prepare_inputs_labels_for_multimodal(self, *args, **kwargs):
    out = _prepare(self, *args, **kwargs)
    if out[4] is not None:
        print(f"[memprobe] inputs_embeds {out[4].dtype} {tuple(out[4].shape)}, {torch.cuda.memory_allocated() / GB:.2f} GB allocated", flush=True)
        projector = getattr(self.get_model(), "vggt_projector", None)
        if hasattr(projector, "modules_to_save"):
            projector = projector.modules_to_save["default"]
        if getattr(projector, "last_gather_stats", None) is not None:
            print(f"[lookat] training={self.training} {projector.last_gather_stats}", flush=True)
    return out


LlavaMetaForCausalLM.encode_images = encode_images
LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal

if __name__ == "__main__":
    train()
