"""Look-at regression tests, including an optional CUDA precision check: python -m unittest scripts.vggt17.test_lookat -v."""

import builtins
import copy
import importlib.util
import os
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from cambrianp.constants import IGNORE_INDEX
from cambrianp.datasets.vqa_dataloading_utils import video_scene_key
from cambrianp.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
from cambrianp.model.llava_arch import LlavaMetaForCausalLM
from cambrianp.model.multimodal_projector.vggt_token import VGGTFeatureError, VGGTTokenProjector, check_vggt_features, load_vggt_tokens, lookat_weights
from cambrianp.model.reconstructor.vggt_reconstructor import VGGTReconstructor
from scripts.vggt17 import precompute_vggt
from scripts.vggt17.build_train_manifest import split_samples
from vggt.utils.rotation import mat_to_quat


def geometry(points, segments=None, valid=None):
    n = len(points)
    return dict(
        lookat=torch.tensor(points, dtype=torch.float32),
        lookat_depth=torch.full((n,), 2.0),
        seg_id=torch.zeros(n, dtype=torch.long) if segments is None else torch.tensor(segments),
        valid=torch.ones(n, dtype=torch.bool) if valid is None else torch.tensor(valid),
        frame_idx=torch.arange(n),
    )


def weights(g, **kwargs):
    return lookat_weights(g["lookat"], g["lookat_depth"], g["seg_id"], g["valid"], g["frame_idx"], **kwargs)


def mixed_ddp_worker(rank, store):
    from torch import distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.checkpoint import checkpoint

    class MixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projector = VGGTTokenProjector(4, 8, gather_mode="lookat")
            self.projector.reset_parameters()
            self.decoder = torch.nn.Linear(8, 1)

        def forward(self, video, scope):
            self.projector.gather_scope = scope
            features = torch.arange(32, dtype=torch.float32).reshape(4, 8)
            if video:
                features = self.projector(features[:, :4], geometry([[0, 0, 2]] * 4), features)
            return checkpoint(self.decoder, features, use_reentrant=False).square().mean()

    torch.set_num_threads(1)
    torch.manual_seed(42)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{store}")
    try:
        model = DistributedDataParallel(MixedModel(), find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        initial = model.module.projector.mlp[0].weight.detach().clone()
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            video = step == 3 or (step < 2 and rank == step)
            model(video, "all" if step == 3 else "self").backward()
            optimizer.step()
        actual = model.module.projector.mlp[0].weight.detach()
        assert torch.isfinite(actual).all() and not torch.equal(actual, initial)
        gathered = [torch.empty_like(actual) for _ in range(2)]
        dist.all_gather(gathered, actual)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class LookAtTests(unittest.TestCase):
    def test_nonzero_checkpointing_does_not_require_deepspeed(self):
        original_import = builtins.__import__
        original_spec = importlib.util.find_spec

        def without_deepspeed(name, *args, **kwargs):
            if name == "deepspeed" or name.startswith("deepspeed."):
                raise AssertionError(f"Unexpected DeepSpeed import: {name}")
            return original_import(name, *args, **kwargs)

        def find_spec(name, *args, **kwargs):
            return None if name == "deepspeed" else original_spec(name, *args, **kwargs)

        with patch("builtins.__import__", without_deepspeed), patch("importlib.util.find_spec", find_spec):
            from accelerate import Accelerator
            from cambrianp.train import train, llava_trainer

            accelerator = Accelerator(cpu=True)
            model = torch.nn.Linear(4, 4)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            initial = model.weight.detach().clone()
            model, optimizer = accelerator.prepare(model, optimizer)
            accelerator.backward(model(torch.ones(2, 4)).square().mean())
            optimizer.step()
            param = accelerator.unwrap_model(model).weight
            self.assertFalse(torch.equal(param, initial))
            states = [
                train.maybe_zero_3(param),
                llava_trainer.maybe_zero_3(param),
                train.get_peft_state_maybe_zero_3([("model.lora_A.weight", param)], "none")["model.lora_A.weight"],
                train.get_peft_state_non_lora_maybe_zero_3([("model.vggt_projector.weight", param)])["model.vggt_projector.weight"],
                llava_trainer.get_mm_adapter_state_maybe_zero_3([("model.mm_projector.weight", param)], ["mm_projector"])["model.mm_projector.weight"],
            ]
            for state in states:
                torch.testing.assert_close(state, param.detach())
                self.assertEqual(state.device.type, "cpu")
                self.assertFalse(state.requires_grad)
                self.assertNotEqual(state.data_ptr(), param.data_ptr())

    def test_zero3_checkpointing_still_gathers_partitioned_parameters(self):
        from cambrianp.train import train, llava_trainer

        deepspeed = ModuleType("deepspeed")
        gathered = MagicMock()
        deepspeed.zero = SimpleNamespace(GatheredParameters=gathered)
        partition = ModuleType("deepspeed.runtime.zero.partition_parameters")
        partition.ZeroParamStatus = SimpleNamespace(NOT_AVAILABLE=1)
        param = torch.nn.Parameter(torch.ones(4))
        param.ds_id, param.ds_status = 0, partition.ZeroParamStatus.NOT_AVAILABLE
        with patch.dict("sys.modules", {"deepspeed": deepspeed, partition.__name__: partition}):
            for clone in (train.maybe_zero_3, llava_trainer.maybe_zero_3):
                gathered.reset_mock()
                state = clone(param, ignore_status=True)
                gathered.assert_called_once_with([param])
                gathered.return_value.__enter__.assert_called_once()
                torch.testing.assert_close(state, param.detach())
                self.assertFalse(state.requires_grad)

    def test_ddp_handles_image_only_and_video_ranks_with_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(mixed_ddp_worker, args=(os.path.join(tmp, "store"),), nprocs=2)

    def test_training_cli_supports_nonreentrant_ddp_checkpointing(self):
        from transformers import HfArgumentParser
        from cambrianp.train.train import TrainingArguments

        args = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses([
            "--output_dir", "/tmp/vggt_checkpointing_argument_test",
            "--ddp_find_unused_parameters", "True",
            "--gradient_checkpointing_use_reentrant", "False",
        ])[0]
        self.assertTrue(args.ddp_find_unused_parameters)
        self.assertEqual(args.gradient_checkpointing_kwargs, {"use_reentrant": False})

    def test_revisits_are_selected_by_target_not_time(self):
        g = geometry([[0, 0, 2], [10, 0, 2], [20, 0, 2], [30, 0, 2], [0.05, 0, 2]])
        w = weights(g, radius=0.1, far_gap=3)
        self.assertGreater(w[0, 4], 0)
        self.assertEqual(w[0, 1], 0)
        torch.testing.assert_close(w.sum(-1), torch.ones(5))
        self.assertGreater(weights(g, scope="far", far_gap=3)[0, 4], 0)
        self.assertEqual(weights(g, scope="near", far_gap=3)[0, 4], 0)
        transformed = copy.deepcopy(g)
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transformed["lookat"] = 3 * g["lookat"] @ rotation.T + torch.tensor([10.0, -5.0, 8.0])
        transformed["lookat_depth"] *= 3
        torch.testing.assert_close(weights(transformed, radius=0.1, far_gap=3), w, atol=2e-6, rtol=2e-6)
        g["frame_idx"][4] = g["frame_idx"][0]
        self.assertEqual(weights(g)[0, 4], 0, "duplicate samples are not revisits")

    def test_clip_boundaries_and_invalid_geometry_fall_back_to_self(self):
        g = geometry([[0, 0, 2], [0, 0, 2], [float("nan"), 0, 2], [0, 0, 2]], segments=[0, 1, 0, 0])
        w = weights(g)
        self.assertEqual(w[0, 1], 0)
        self.assertEqual(w[0, 2], 0)
        self.assertGreater(w[0, 3], 0)
        self.assertTrue(torch.isfinite(w).all())
        g["valid"].fill_(False)
        torch.testing.assert_close(weights(g), torch.eye(4), atol=0, rtol=0)

    def test_visual_content_and_geometry_gates_train_without_extra_tokens(self):
        torch.manual_seed(11)
        p = VGGTTokenProjector(4, 4, gather_mode="lookat", gather_far_gap=3)
        p.reset_parameters()
        g = geometry([[0, 0, 2], [10, 0, 2], [20, 0, 2], [30, 0, 2], [0, 0, 2]])
        x = torch.ones(5, 4)
        visual = torch.zeros(5, 4)
        visual[-1] = 10
        out = p(x, geometry=g, visual_features=visual)
        self.assertEqual(out.shape, (5, 4))
        self.assertFalse(torch.equal(out[0], p.mlp(x)[0]), "matching views must contribute visual content even if camera features are identical")
        out[0].sum().backward()
        self.assertGreater(abs(p.visual_gather_logit.grad.item()), 0)
        p.zero_grad(set_to_none=True)
        x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        p(x, geometry=g, visual_features=visual).square().sum().backward()
        self.assertGreater(abs(p.gather_logit.grad.item()), 0)
        self.assertGreater(p.mlp[2].weight.grad.norm().item(), 0)
        p.gather_scope = "self"
        torch.testing.assert_close(p(x, geometry=g, visual_features=visual), p.mlp(x), atol=0, rtol=0)

    def test_low_memory_checkpoint_loading_initializes_new_gatherer(self):
        config = LlavaQwenConfig(vocab_size=23, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
        with tempfile.TemporaryDirectory() as tmp:
            LlavaQwenForCausalLM(config).save_pretrained(tmp)
            config.use_vggt_token = True
            config.vggt_token_dim = 4
            config.vggt_gather_mode = "lookat"
            loaded = LlavaQwenForCausalLM.from_pretrained(tmp, config=config, low_cpu_mem_usage=True, device_map="cpu", torch_dtype=torch.float16)
            p = loaded.get_model().vggt_projector.float()
            p.reset_parameters()
            result = p(torch.ones(5, 4), geometry([[0, 0, 2]] * 5), torch.randn(5, 8))
            self.assertTrue(torch.isfinite(result).all())
            self.assertGreater(p.last_gather_stats["neighbors"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA precision check")
    def test_self_and_invalid_fallback_are_exact_with_cuda_tf32(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            p = VGGTTokenProjector(2048, 64, gather_mode="lookat").cuda()
            p.reset_parameters()
            baseline = copy.deepcopy(p)
            baseline.gather_mode = "none"
            x, visual = torch.randn(128, 2048, device="cuda"), torch.randn(128, 64, device="cuda")
            g = {k: v.cuda() for k, v in geometry([[0, 0, 2]] * 128).items()}
            for scope in ("self", "all"):
                p.gather_scope = scope
                g["valid"].fill_(scope == "self")
                for amp in (False, True):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        expected = baseline(x)
                        actual = p(x, g, visual).to(expected.dtype)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    def test_injection_preserves_images_cameras_labels_and_budget(self):
        frames, dim = 3, 4
        p = VGGTTokenProjector(dim, dim, gather_mode="lookat")
        p.reset_parameters()
        host = SimpleNamespace(get_model=lambda: SimpleNamespace(vggt_projector=p), config=SimpleNamespace(mm_img_tok_num=4, mm_newline_position="grid"))
        feat = geometry([[0, 0, 2]] * frames)
        feat["tokens"] = torch.randn(frames, dim)
        base = torch.arange(frames * 7 * dim, dtype=torch.float32).reshape(frames, 7, dim)
        for place in ("append_to_frame", "prepend_to_frame"):
            prefix, suffix = torch.randn(2, dim), torch.randn(1, dim)
            segments = [prefix, base.reshape(-1, dim).clone(), suffix]
            labels = [torch.full((len(x),), IGNORE_INDEX) for x in segments]
            info = dict(pre_image_token_length=2, image_token_length=frames * 7, num_frames=frames, camera_tokens_place=place)
            LlavaMetaForCausalLM.inject_vggt_tokens(host, feat, frames, segments, labels, info)
            torch.testing.assert_close(segments[1].reshape(frames, 8, dim)[:, :7], base, atol=0, rtol=0)
            self.assertEqual(info["image_token_length"], frames * 8)
            self.assertTrue(labels[1].eq(IGNORE_INDEX).all())
            rec = VGGTReconstructor.prepare_rec_tokens_for_single_layer([info], (torch.cat(segments)[None],), grid_size=2)
            cam = base[:, -1] if place == "append_to_frame" else base[:, 0]
            images = base[:, :6] if place == "append_to_frame" else base[:, 1:]
            images = images.reshape(frames, 2, 3, dim)[:, :, :2].reshape(frames, 4, dim)
            torch.testing.assert_close(rec[0, :, 0], cam)
            torch.testing.assert_close(rec[0, :, 1:], images)

    def test_loader_requires_shared_scene_geometry_and_matching_anchors(self):
        n = 5
        pose = np.zeros((n, 9), np.float32)
        pose[:, 3:7] = mat_to_quat(torch.eye(3)).numpy()
        pose[:, 7:] = 1
        pose[:, 0] = -np.arange(n)
        points = np.stack([np.arange(n), np.zeros(n), np.full(n, 2)], -1).astype(np.float32)
        data = dict(anchors=np.linspace(0, 99, n, dtype=np.int32), n_frames=100, fps=30, group_tok=np.ones((n, 17, 4), np.float16), group_pose=np.zeros((n, 17, 9), np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.npz")
            np.savez(path, **data)
            self.assertEqual(load_vggt_tokens(path).shape, (n, 4))
            with self.assertRaisesRegex(VGGTFeatureError, "scene_only"):
                check_vggt_features(path, gather_mode="lookat", num_anchors=n)
            data.update(scene_pose=pose, scene_lookat=points, seg_id=np.zeros(n, np.int16))
            np.savez(path, **data)
            g = load_vggt_tokens(path, gather_mode="lookat")
            np.testing.assert_allclose(g["lookat_depth"], 2)
            self.assertTrue(g["valid"].all())
            data["scene_lookat"][2, 0] = np.nan
            np.savez(path, **data)
            self.assertFalse(load_vggt_tokens(path, gather_mode="lookat")["valid"][2])
            with self.assertRaises(VGGTFeatureError):
                load_vggt_tokens(path, frame_idx=data["anchors"] + 1, gather_mode="lookat")
            with self.assertRaises(VGGTFeatureError):
                check_vggt_features(path, gather_mode="lookat", num_anchors=n + 1)

    def test_split_keeps_rooms_and_all_answer_turns_together(self):
        rows = []
        for src in ("scannet", "arkitscenes"):
            for i in range(8):
                for scan in range(2):
                    video = f"{src}/scene{i}_{scan:02d}.mp4" if src == "scannet" else f"{src}/video{i}.mp4"
                    rows.append(dict(video=video, conversations=[dict(value=str(j)) for j in range(8)]))
        train, val = split_samples(rows, 0.25, 42, set())
        self.assertFalse({video_scene_key(d["video"]) for d in train} & {video_scene_key(d["video"]) for d in val})
        self.assertEqual(len(train) + len(val), len(rows))
        self.assertTrue(all(len(d["conversations"]) == 8 for d in train + val))
        self.assertEqual((train, val), split_samples(rows, 0.25, 42, set()))
        with self.assertRaisesRegex(ValueError, "benchmark"):
            split_samples(rows, 0.25, 42, {video_scene_key(rows[0]["video"])})

    def test_validation_sparse_loss_matches_dense_masked_loss(self):
        torch.manual_seed(5)
        model = LlavaQwenForCausalLM(LlavaQwenConfig(vocab_size=23, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
        ids = torch.randint(0, 23, (2, 7))
        labels = ids.clone()
        labels[:, :4] = IGNORE_INDEX
        for training in (True, False):
            model.train(training)
            model.config.lm_loss_labels_only = False
            dense = model.llm_forward(input_ids=ids, labels=labels, use_cache=False).loss
            model.zero_grad(set_to_none=True)
            dense.backward()
            grads = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            model.config.lm_loss_labels_only = True
            sparse = model.llm_forward(input_ids=ids, labels=labels, use_cache=False)
            torch.testing.assert_close(sparse.loss, dense)
            sparse.loss.backward()
            for k, p in model.named_parameters():
                if k in grads:
                    torch.testing.assert_close(p.grad, grads[k], rtol=1e-5, atol=2e-7)

    def test_scene_only_merge_preserves_groups_and_rejects_misalignment(self):
        anchors, stride, bounds, segments, group_idx, valid = precompute_vggt.plan(1000, 30, [])
        cached = dict(
            anchors=anchors,
            stride=stride,
            n_frames=1000,
            fps=30,
            seg_bounds=bounds,
            seg_id=segments,
            group_idx=group_idx,
            group_valid=valid,
            hw=np.array([518, 392]),
            group_tok=np.ones((128, 17, 4), np.float16),
            group_pose=np.ones((128, 17, 9), np.float32),
        )
        scene = {k: v for k, v in cached.items() if k not in ("group_tok", "group_pose", "hw")}
        scene.update(scene_pose=np.zeros((128, 9), np.float32), scene_lookat=np.ones((128, 3), np.float32), scene_depth=np.ones((128, 2, 2), np.float16), scene_conf=np.ones((128, 2, 2)), scene_long_side=448, scene_hw=np.array([448, 336]))
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "scannet", "video.npz")
            os.makedirs(os.path.dirname(dst))
            np.savez(dst, **cached)
            argv = ["precompute", "--root", tmp, "--out", tmp, "--videos", "scannet/video.mp4", "--scene_only"]
            with patch("sys.argv", argv), patch.object(precompute_vggt, "process", return_value=scene), patch.object(precompute_vggt.VGGT, "from_pretrained"), patch("torch.cuda.reset_peak_memory_stats"), patch(
                "torch.cuda.max_memory_allocated", return_value=0
            ):
                precompute_vggt.main()
            with np.load(dst) as z:
                for key in ("group_tok", "group_pose", "hw"):
                    np.testing.assert_array_equal(z[key], cached[key])
            np.savez(dst, **cached)
            bad = dict(scene, anchors=anchors + 1)
            with patch("sys.argv", argv), patch.object(precompute_vggt, "process", return_value=bad), patch.object(precompute_vggt.VGGT, "from_pretrained"), patch("torch.cuda.reset_peak_memory_stats"), patch(
                "torch.cuda.empty_cache"
            ), self.assertRaises(SystemExit):
                precompute_vggt.main()
            with np.load(dst) as z:
                self.assertNotIn("scene_lookat", z.files)
                np.testing.assert_array_equal(z["group_tok"], cached["group_tok"])


if __name__ == "__main__":
    unittest.main()
