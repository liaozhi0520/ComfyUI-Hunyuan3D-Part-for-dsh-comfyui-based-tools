"""ComfyUI-native Conditioner for Hunyuan3D-Part.

Consolidates conditioner/condioner_release.py + conditioner/part_encoders.py
into a single flat file with operations= parameter and no torch.autocast.
"""

import torch
import torch.nn as nn

from .vae import VolumeDecoderShapeVAE
from ..sonata.extractor import SonataFeatureExtractor


def _nearest_index_chunked(query, reference, budget=256 * 2**20):
    """Exact argmin of ``cdist(query, reference, p=2)`` over the reference point
    dimension, computed in chunks so the ``(..., Q, R)`` distance matrix is never
    materialised in full (peak VRAM ~= ``budget`` instead of O(Q*R)).

    Equivalent to an unchunked distance matrix + ``argmin(dim=-1)`` for both 2-D
    and batched 3-D inputs, tie-breaking included: the strict ``<`` keeps the
    earliest index, exactly like ``argmin`` does.
    """
    squeeze = query.dim() == 2
    if squeeze:
        query = query.unsqueeze(0)
    if reference.dim() == 2:
        reference = reference.unsqueeze(0)

    total = reference.shape[-2]
    batch = max(query.shape[0], reference.shape[0])
    per_point = max(1, batch * query.shape[-2] * reference.element_size())
    step = max(1, min(total, int(budget // per_point)))

    best_val = best_idx = None
    for start in range(0, total, step):
        dist = torch.cdist(query, reference[..., start:start + step, :], p=2)
        val, idx = torch.min(dist, dim=-1)
        idx = idx + start
        if best_val is None:
            best_val, best_idx = val, idx
        else:
            better = val < best_val
            best_val = torch.where(better, val, best_val)
            best_idx = torch.where(better, idx, best_idx)
        del dist, val, idx
    return best_idx.squeeze(0) if squeeze else best_idx


# --------------------------------------------------------------------------- #
#  PartEncoder
# --------------------------------------------------------------------------- #

class PartEncoder(nn.Module):
    def __init__(self, use_local=True, local_global_feat_dim=None,
                 local_geo_cfg=None, local_feat_type="latents",
                 num_tokens_cond=2048,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.local_global_feat_dim = local_global_feat_dim
        self.local_feat_type = local_feat_type
        self.num_tokens_cond = num_tokens_cond
        self.use_local = use_local

        if use_local:
            if local_geo_cfg is None:
                raise ValueError("local_geo_cfg must be provided when use_local is True")
            # Build VolumeDecoderShapeVAE directly from params dict
            self.local_encoder = VolumeDecoderShapeVAE(
                **local_geo_cfg, dtype=dtype, device=device, operations=operations
            )
            if self.local_global_feat_dim is not None:
                # Determine input dim from geo cfg
                if self.local_feat_type == "latents":
                    in_dim = local_geo_cfg.get("embed_dim", 64)
                else:
                    in_dim = local_geo_cfg.get("width", 1024)
                self.local_out_layer = operations.Linear(
                    in_dim, self.local_global_feat_dim, bias=True,
                    dtype=dtype, device=device,
                )

    def forward(self, part_surface_inbbox, object_surface, return_local_pc_info=False):
        if self.use_local:
            if self.local_feat_type == "latents":
                local_features, local_pc_infos = self.local_encoder.encode(
                    part_surface_inbbox, sample_posterior=True, return_pc_info=True
                )
            elif self.local_feat_type == "latents_shape":
                local_features, local_pc_infos = self.local_encoder.encode_shape(
                    part_surface_inbbox, return_pc_info=True
                )
            elif self.local_feat_type == "miche-point-query-structural-vae":
                local_features, local_pc_infos = self.local_encoder.encode(
                    part_surface_inbbox, sample_posterior=True, return_pc_info=True
                )
                local_features = self.local_encoder(local_features)
            else:
                raise ValueError(f"local_feat_type {self.local_feat_type} not supported")

            geo_features = (
                self.local_out_layer(local_features)
                if hasattr(self, "local_out_layer")
                else local_features
            )

        if return_local_pc_info:
            return geo_features, local_pc_infos
        return geo_features


# --------------------------------------------------------------------------- #
#  Conditioner
# --------------------------------------------------------------------------- #

class Conditioner(nn.Module):
    def __init__(
        self,
        use_image=False,
        use_geo=True,
        use_obj=True,
        use_seg_feat=False,
        # Direct params instead of OmegaConf configs
        geo_encoder_params=None,
        geo_output_dim=None,
        obj_encoder_params=None,
        obj_output_dim=None,
        seg_feat_encoder_params=None,
        seg_feat_output_dim=None,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.use_image = use_image
        self.use_obj = use_obj
        self.use_geo = use_geo
        self.use_seg_feat = use_seg_feat

        if use_geo and geo_encoder_params is not None:
            self.geo_encoder = PartEncoder(
                **geo_encoder_params,
                dtype=dtype, device=device, operations=operations,
            )
            if geo_output_dim is not None:
                self.geo_out_proj = operations.Linear(
                    1024 + 512, geo_output_dim,
                    dtype=dtype, device=device,
                )

        if use_obj and obj_encoder_params is not None:
            self.obj_encoder = VolumeDecoderShapeVAE(
                **obj_encoder_params,
                dtype=dtype, device=device, operations=operations,
            )
            if obj_output_dim is not None:
                self.obj_out_proj = operations.Linear(
                    1024 + 512, obj_output_dim,
                    dtype=dtype, device=device,
                )

        if use_seg_feat and seg_feat_encoder_params is not None:
            self.seg_feat_encoder = SonataFeatureExtractor(**seg_feat_encoder_params)
            if seg_feat_output_dim is not None:
                self.seg_feat_outproj = operations.Linear(
                    512, seg_feat_output_dim,
                    dtype=dtype, device=device,
                )

    def forward(self, part_surface_inbbox, object_surface, precomputed_sonata_features=None):
        bz = part_surface_inbbox.shape[0]
        context = {}

        # geo_cond
        if self.use_geo:
            context["geo_cond"], local_pc_infos = self.geo_encoder(
                part_surface_inbbox, object_surface, return_local_pc_info=True,
            )

        # obj cond
        if self.use_obj:
            with torch.no_grad():
                context["obj_cond"], global_pc_infos = self.obj_encoder.encode_shape(
                    object_surface, return_pc_info=True
                )

        # seg feat cond (NO torch.autocast - explicit dtype casting instead)
        if self.use_seg_feat:
            num_parts = part_surface_inbbox.shape[0]

            if precomputed_sonata_features is not None:
                precomputed_points = precomputed_sonata_features['points']
                precomputed_feats = precomputed_sonata_features['features']

                obj_points = object_surface[:1, ..., :3].to(dtype=precomputed_points.dtype)

                with torch.no_grad():
                    nearest_indices = _nearest_index_chunked(
                        obj_points[0], precomputed_points
                    )
                    point_feat = precomputed_feats[nearest_indices].unsqueeze(0)

                print(f"[Conditioner] Using precomputed Sonata features ({precomputed_feats.shape[0]} -> {point_feat.shape[1]} points)")
            else:
                with torch.no_grad():
                    point, normal = (
                        object_surface[:1, ..., :3].float(),
                        object_surface[:1, ..., 3:6].float(),
                    )
                    point_feat = self.seg_feat_encoder(point, normal)

            # local feat
            if self.use_obj:
                nearest_global_matches = _nearest_index_chunked(
                    global_pc_infos[0], object_surface[..., :3]
                )
                global_point_feats = point_feat.expand(num_parts, -1, -1).gather(
                    1,
                    nearest_global_matches.unsqueeze(-1).expand(-1, -1, point_feat.size(-1)),
                )
                context["obj_cond"] = torch.concat(
                    [context["obj_cond"], global_point_feats], dim=-1
                ).to(dtype=self.obj_out_proj.weight.dtype)
                if hasattr(self, "obj_out_proj"):
                    context["obj_cond"] = self.obj_out_proj(context["obj_cond"])

            if self.use_geo:
                nearest_local_matches = _nearest_index_chunked(
                    local_pc_infos[0], object_surface[..., :3]
                )
                local_point_feats = point_feat.expand(num_parts, -1, -1).gather(
                    1,
                    nearest_local_matches.unsqueeze(-1).expand(-1, -1, point_feat.size(-1)),
                )
                context["geo_cond"] = torch.concat(
                    [context["geo_cond"], local_point_feats], dim=-1,
                ).to(dtype=self.geo_out_proj.weight.dtype)
                if hasattr(self, "geo_out_proj"):
                    context["geo_cond"] = self.geo_out_proj(context["geo_cond"])

        return context
