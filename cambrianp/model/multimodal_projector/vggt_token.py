"""VGGT-17 tokens: one precomputed VGGT camera token per sampled frame, appended after the frame's
image (and CAM) tokens. Features come from scripts/vggt17/precompute_vggt.py (one npz per video)."""

import math
import os
import zipfile

import numpy as np
import torch
import torch.nn as nn

VGGT_TOKEN_SOURCES = ("group", "scene")
VGGT_GATHER_MODES = ("none", "lookat")
VGGT_GATHER_SCOPES = ("all", "near", "far", "self")


class VGGTFeatureError(ValueError):
    pass


def vggt_feature_path(feat_dir, video_path, video_root=None):
    """Preserve dataset-relative subfolders; video_root resolves absolute video paths.

    Without a root, absolute paths retain the legacy parent/stem lookup used by flat benchmark caches.
    """
    video_path = os.path.normpath(video_path)
    if os.path.isabs(video_path):
        if video_root is not None:
            video_path = os.path.relpath(video_path, video_root)
        else:
            video_path = os.path.join(os.path.basename(os.path.dirname(video_path)), os.path.basename(video_path))
    if video_path == os.pardir or video_path.startswith(os.pardir + os.sep):
        raise VGGTFeatureError(f"{video_path}: video path is outside the dataset root")
    return os.path.join(feat_dir, os.path.splitext(video_path)[0] + ".npz")


def _check_metadata(z, npz_path, source, gather_mode, num_anchors=None):
    required = {"anchors", "n_frames", "fps", "group_tok" if source == "group" else "scene_tok"}
    if gather_mode == "lookat":
        required.update(("scene_pose", "scene_lookat", "seg_id"))
    missing = required.difference(z.files)
    if missing:
        raise VGGTFeatureError(f"{npz_path}: missing {sorted(missing)}; look-at gathering needs shared-frame geometry (precompute_vggt.py --scene_only)")
    anchors, total, fps = z["anchors"], int(z["n_frames"]), float(z["fps"])
    if total <= 0 or not math.isfinite(fps) or fps <= 0 or anchors.ndim != 1 or not len(anchors):
        raise VGGTFeatureError(f"{npz_path}: invalid video/anchor metadata")
    if num_anchors is not None and len(anchors) != num_anchors:
        raise VGGTFeatureError(f"{npz_path}: {len(anchors)} feature anchors, expected {num_anchors}")
    return anchors, total, fps


def check_vggt_features(npz_path, source="group", gather_mode="none", num_anchors=None):
    """Cheap preflight: check keys and sampling without reading the large group-token array."""
    try:
        with np.load(npz_path) as z:
            anchors, total, _ = _check_metadata(z, npz_path, source, gather_mode, num_anchors)
            if not np.array_equal(anchors, np.linspace(0, total - 1, len(anchors), dtype=int)):
                raise VGGTFeatureError(f"{npz_path}: anchors are not uniform linspace samples")
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as e:
        raise VGGTFeatureError(str(e)) from e


def load_vggt_tokens(npz_path, source="group", frame_idx=None, n_frames=None, video_time=None, gather_mode="none"):
    """Return [num_anchors, dim] float16 tokens, checking that the npz anchors are the frames fed to the LLM.

    source="group": the anchor's own token from its 17-frame group pass (group_tok[:, 0]).
    source="scene": the token from the single pass over all anchors of the clip (scene_tok).
    video_time: duration (s) of the decoded video, len(vr) / fps as in cambrianp.utils.process_video_with_decord.
    gather_mode="lookat": return tokens plus shared-frame looked-at points, ray depths, segment IDs and validity.
    """
    if source not in VGGT_TOKEN_SOURCES:
        raise ValueError(f"vggt token source must be one of {VGGT_TOKEN_SOURCES}, got {source}")
    if gather_mode not in VGGT_GATHER_MODES:
        raise ValueError(f"vggt gather mode must be one of {VGGT_GATHER_MODES}, got {gather_mode}")
    try:
        with np.load(npz_path) as z:
            anchors, total, fps = _check_metadata(z, npz_path, source, gather_mode)
            if n_frames is not None and total != n_frames:
                raise VGGTFeatureError(f"{npz_path}: features were computed on {total} frames, video has {n_frames}")
            if video_time is not None and abs(total / fps - video_time) > 1e-3:
                raise VGGTFeatureError(f"{npz_path}: features were computed on a {total / fps:.3f} s video, got {video_time:.3f} s")
            expected = np.asarray(frame_idx) if frame_idx is not None else np.linspace(0, total - 1, len(anchors), dtype=int)
            if not np.array_equal(anchors, expected):
                raise VGGTFeatureError(f"{npz_path}: feature anchors differ from the sampled frame indices")
            tok = np.ascontiguousarray(z["group_tok"][:, 0] if source == "group" else z["scene_tok"])
            if tok.ndim != 2 or len(tok) != len(anchors) or not np.isfinite(tok).all():
                raise VGGTFeatureError(f"{npz_path}: invalid or non-finite VGGT tokens")
            if gather_mode == "none":
                return tok
            pose, gaze, segments = z["scene_pose"], z["scene_lookat"], z["seg_id"]
            if pose.shape != (len(anchors), 9) or gaze.shape != (len(anchors), 3) or segments.shape != anchors.shape:
                raise VGGTFeatureError(f"{npz_path}: geometry does not match the feature anchors")
            # Independent 17-frame group poses have different origins/scales; only scene_pose is comparable across anchors.
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri

            with torch.no_grad():
                pe = torch.from_numpy(pose).float()
                ext, _ = pose_encoding_to_extri_intri(pe[None], build_intrinsics=False)
                R, t = ext[0, :, :, :3], ext[0, :, :, 3]
                centre = -(R.transpose(1, 2) @ t[..., None])[..., 0]
                points = torch.from_numpy(gaze).float()
                depth = ((points - centre) * R[:, 2, :]).sum(-1)
                valid = torch.isfinite(pe).all(-1) & pe[:, 3:7].norm(dim=-1).gt(1e-6)
                valid &= torch.isfinite(points).all(-1) & torch.isfinite(depth) & depth.gt(0)
            return dict(tokens=tok, lookat=points.numpy(), lookat_depth=depth.numpy(), valid=valid.numpy(), seg_id=segments.astype(np.int64), frame_idx=anchors.astype(np.int64))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as e:
        raise VGGTFeatureError(str(e)) from e


@torch.no_grad()
def lookat_weights(lookat, depth, seg_id, valid=None, frame_idx=None, radius=0.15, scope="all", far_gap=32):
    """Pool views looking within radius * median ray depth of the same 3D point, in each clip's coordinate frame."""
    if not math.isfinite(radius) or radius <= 0 or scope not in VGGT_GATHER_SCOPES or far_gap < 1:
        raise ValueError("look-at gathering needs a positive finite radius, a valid scope and far_gap >= 1")
    n = len(lookat)
    if not n or lookat.shape != (n, 3) or depth.shape != (n,) or seg_id.shape != (n,):
        raise ValueError("look-at points, depths and segment IDs must have matching frame dimensions")
    if (valid is not None and valid.shape != (n,)) or (frame_idx is not None and frame_idx.shape != (n,)):
        raise ValueError("look-at validity and frame indices must match the frames")
    valid_points = torch.isfinite(lookat).all(-1) & torch.isfinite(depth) & depth.gt(0) & seg_id.ge(0)
    if valid is not None:
        valid_points &= valid
    points = torch.nan_to_num(lookat.float(), nan=0.0, posinf=0.0, neginf=0.0)
    scales = torch.ones(n, device=points.device)
    for segment in torch.unique(seg_id):
        members = seg_id.eq(segment)
        good = members & valid_points
        if good.any():
            scales[members] = depth[good].float().median()
    distance = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
    normalized = distance / (radius * scales[:, None]).clamp_min(torch.finfo(torch.float32).tiny)
    allowed = valid_points[:, None] & valid_points[None] & seg_id[:, None].eq(seg_id[None]) & normalized.le(1)
    idx = torch.arange(n, device=points.device)
    gap = (idx[:, None] - idx[None]).abs()
    if scope == "near":
        allowed &= gap.lt(far_gap)
    elif scope == "far":
        allowed &= gap.ge(far_gap)
    elif scope == "self":
        allowed &= gap.eq(0)
    if frame_idx is not None:
        allowed &= frame_idx[:, None].ne(frame_idx[None])
    weights = torch.where(allowed, torch.exp(-0.5 * normalized.square()), 0.0)
    # Invalid geometry and unmatched views must reproduce the original anchor feature, never an empty/NaN pool.
    weights.diagonal().fill_(1.0)
    return weights / weights.sum(-1, keepdim=True)


class VGGTTokenProjector(nn.Module):
    """Project VGGT context; optionally gather geometry and central visual content from matching looked-at places."""

    def __init__(self, in_dim, hidden_size, gather_mode="none", gather_radius=0.15, gather_scope="all", gather_far_gap=32):
        super().__init__()
        if gather_mode not in VGGT_GATHER_MODES:
            raise ValueError(f"invalid VGGT gather mode: {gather_mode}")
        if not math.isfinite(gather_radius) or gather_radius <= 0 or gather_scope not in VGGT_GATHER_SCOPES or gather_far_gap < 1:
            raise ValueError("invalid look-at gathering parameters")
        self.gather_mode, self.gather_radius = gather_mode, gather_radius
        self.gather_scope, self.gather_far_gap = gather_scope, gather_far_gap
        self.last_gather_stats = None
        if gather_mode == "lookat":
            # The installed low-memory HF loader cannot allocate missing zero-dimensional parameters.
            self.gather_logit = nn.Parameter(torch.tensor([math.log(0.1 / 0.9)]))
            self.visual_gather_logit = nn.Parameter(torch.tensor([math.log(0.1 / 0.9)]))
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))

    @torch.no_grad()
    def reset_parameters(self):
        # Also the buffers: HF from_pretrained leaves keys missing from the checkpoint as torch.empty.
        self.feat_mean.zero_()
        self.feat_std.fill_(1.0)
        if self.gather_mode == "lookat":
            self.gather_logit.fill_(math.log(0.1 / 0.9))
            self.visual_gather_logit.fill_(math.log(0.1 / 0.9))
        # Not zero-init: the LLM's input RMSNorm (eps 1e-6) maps a token of per-dim rms r to x / sqrt(r^2 + 1e-6), so near 0
        # (r ~ 1e-3) tiny updates swing the token's normalized direction and its gradient is amplified up to 1000x, which made
        # training blow up. At std 0.02 the token starts at image-token scale (L2 norm ~33 vs ~45) in a VGGT-dependent direction.
        nn.init.trunc_normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.trunc_normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    @torch.no_grad()
    def load_stats(self, stats_path):
        with np.load(stats_path) as z:
            mean, std = torch.from_numpy(z["mean"]).float(), torch.from_numpy(z["std"]).float()
        if mean.shape != self.feat_mean.shape or std.shape != self.feat_std.shape:
            raise ValueError(f"{stats_path}: stats shape {tuple(mean.shape)} != projector input {tuple(self.feat_mean.shape)}")
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(1e-4))

    def forward(self, x, geometry=None, visual_features=None):
        x = (x.float() - self.feat_mean.float()) / self.feat_std.float()
        if self.gather_mode == "lookat":
            if geometry is None:
                raise VGGTFeatureError("look-at gathering requires shared-frame geometry")
            if visual_features is None or visual_features.shape != (len(x), self.mlp[2].out_features):
                raise ValueError("look-at gathering requires one central visual descriptor per anchor")
            with torch.autocast(device_type=x.device.type, enabled=False):
                weights = lookat_weights(
                    geometry["lookat"].float(),
                    geometry["lookat_depth"].float(),
                    geometry["seg_id"],
                    geometry["valid"],
                    geometry["frame_idx"],
                    self.gather_radius,
                    self.gather_scope,
                    self.gather_far_gap,
                )
                # Exclude self terms so unmatched views stay exactly unchanged even with CUDA TF32.
                weights.diagonal().zero_()
                neighbor_mass = weights.sum(-1, keepdim=True)
                gate = self.gather_logit.float().sigmoid()
                x = x + gate * (weights @ x - neighbor_mass * x)
                gap = (torch.arange(len(x), device=x.device)[:, None] - torch.arange(len(x), device=x.device)[None]).abs()
                neighbors = weights.gt(0) & gap.gt(0)
                self.last_gather_stats = dict(
                    neighbors=float(neighbors.sum(-1).float().mean()),
                    far_neighbors=float((neighbors & gap.ge(self.gather_far_gap)).sum(-1).float().mean()),
                    gate=float(gate.detach()),
                    visual_gate=float(self.visual_gather_logit.detach().float().sigmoid()),
                )
        projected = self.mlp(x.to(self.mlp[0].weight.dtype))
        if self.gather_mode == "lookat":
            with torch.autocast(device_type=x.device.type, enabled=False):
                visual = visual_features.float()
                projected = projected.float() + self.visual_gather_logit.float().sigmoid() * (weights @ visual - neighbor_mass * visual)
        return projected
