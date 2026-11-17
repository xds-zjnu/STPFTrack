from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.ostrack.utils import combine_tokens, recover_tokens
from stpf import (FieldModulation, FieldUpdate, check_field_tensor,
                  compute_template_search_similarity, generate_center_gaussian_field,
                  is_etfm_enabled, logits_to_prob, prob_to_logits,
                  log_etfm_dense_token_notice_once)


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

        self.etfm_enabled = False
        self.etfm_cfg = None
        self.etfm_modulation_layers = set()
        self.etfm_update_layers = set()
        self.field_modulations = None
        self.field_updates = None

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # self.cls_token = None
        # self.pos_embed = None

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

        self.setup_etfm(cfg)

    def setup_etfm(self, cfg):
        self.etfm_enabled = is_etfm_enabled(cfg)
        self.etfm_cfg = cfg.MODEL.ETFM if self.etfm_enabled else None
        if not self.etfm_enabled:
            return

        if self.add_cls_token:
            raise ValueError("ETFM currently expects OSTrack token order [template_tokens, search_tokens] without cls token.")

        self.etfm_modulation_layers = set(int(i) for i in self.etfm_cfg.MODULATION_LAYERS)
        self.etfm_update_layers = set(int(i) for i in self.etfm_cfg.UPDATE_LAYERS)
        self.field_modulations = nn.ModuleDict({
            str(i): FieldModulation(self.embed_dim,
                                    bottleneck=self.etfm_cfg.BOTTLENECK,
                                    scale_init=self.etfm_cfg.MOD_SCALE_INIT)
            for i in self.etfm_modulation_layers
        })
        self.field_updates = nn.ModuleDict({
            str(i): FieldUpdate(scale_init=self.etfm_cfg.UPDATE_SCALE_INIT)
            for i in self.etfm_update_layers
        })
        log_etfm_dense_token_notice_once()

    def _init_or_check_target_field(self, target_field, batch_size, feat_h, feat_w, device, dtype):
        if target_field is None:
            target_field = generate_center_gaussian_field(batch_size, feat_h, feat_w, device, dtype=dtype,
                                                          sigma_ratio=self.etfm_cfg.FIELD_SIGMA)

        check_field_tensor("target_field", target_field, batch_size, feat_h, feat_w)
        target_field = target_field.to(device=device, dtype=dtype)
        if bool(getattr(self.etfm_cfg, "USE_FIELD_LOGITS", True)):
            return prob_to_logits(target_field)
        return target_field

    def _init_or_check_target_field_logits(self, target_field_logits, target_field,
                                           batch_size, feat_h, feat_w, device, dtype):
        if bool(getattr(self.etfm_cfg, "USE_FIELD_LOGITS", True)) and target_field_logits is not None:
            check_field_tensor("target_field_logits", target_field_logits, batch_size, feat_h, feat_w)
            return target_field_logits.to(device=device, dtype=dtype)
        return self._init_or_check_target_field(target_field, batch_size, feat_h, feat_w, device, dtype)

    def _init_or_check_box_prior_map(self, box_prior_map, batch_size, feat_h, feat_w, device, dtype):
        if box_prior_map is None:
            if not bool(getattr(self.etfm_cfg, "USE_BOX_PRIOR_IN_UPDATE", True)):
                return torch.zeros(batch_size, 1, feat_h, feat_w, device=device, dtype=dtype)
            return generate_center_gaussian_field(batch_size, feat_h, feat_w, device, dtype=dtype,
                                                  sigma_ratio=self.etfm_cfg.FIELD_SIGMA)
        check_field_tensor("box_prior_map", box_prior_map, batch_size, feat_h, feat_w)
        return box_prior_map.to(device=device, dtype=dtype)

    def _field_state_to_prob(self, field_state):
        if bool(getattr(self.etfm_cfg, "USE_FIELD_LOGITS", True)):
            return logits_to_prob(field_state)
        return field_state

    def _log_field_nan_context(self, field_state, sim_map, box_prior_map):
        def _stats(tensor):
            tensor = tensor.detach()
            return (float(tensor.min()), float(tensor.max()), float(tensor.mean()))

        field_prob = self._field_state_to_prob(field_state)
        print("ETFM NaN context:",
              f"field_state min/max/mean={_stats(field_state)}",
              f"field_prob min/max/mean={_stats(field_prob)}",
              f"sim_map min/max/mean={_stats(sim_map)}",
              f"box_prior_map min/max/mean={_stats(box_prior_map)}")

    def _apply_etfm_after_block(self, block_idx, tokens, lens_z, lens_x, feat_h, feat_w,
                                field_state, box_prior_map, field_outputs):
        if not self.etfm_enabled or block_idx not in self.etfm_modulation_layers | self.etfm_update_layers:
            return tokens, field_state
        if self.cat_mode != "direct":
            raise ValueError("ETFM first implementation supports direct token concatenation only.")

        template_tokens = tokens[:, :lens_z]
        search_tokens = tokens[:, lens_z:lens_z + lens_x]
        if search_tokens.shape[1] != feat_h * feat_w:
            raise ValueError(
                f"ETFM requires dense search tokens, but got Lx={search_tokens.shape[1]}, "
                f"Hf*Wf={feat_h * feat_w}. Please disable candidate elimination."
            )

        if block_idx in self.etfm_modulation_layers:
            search_tokens = self.field_modulations[str(block_idx)](
                search_tokens,
                field_state,
                use_centered=bool(getattr(self.etfm_cfg, "USE_CENTERED_FIELD_MODULATION", True)),
                field_is_logits=bool(getattr(self.etfm_cfg, "USE_FIELD_LOGITS", True)))

        if block_idx in self.etfm_update_layers:
            sim_map = compute_template_search_similarity(template_tokens, search_tokens, feat_h, feat_w)
            field_prob = self._field_state_to_prob(field_state)
            if sim_map.shape != field_prob.shape:
                raise ValueError(f"sim_map shape {tuple(sim_map.shape)} does not match "
                                 f"field shape {tuple(field_state.shape)}")
            if box_prior_map.shape != field_state.shape:
                raise ValueError(f"box_prior_map shape {tuple(box_prior_map.shape)} does not match "
                                 f"field shape {tuple(field_state.shape)}")
            field_state = self.field_updates[str(block_idx)](
                field_state, sim_map, box_prior_map,
                use_logits=bool(getattr(self.etfm_cfg, "USE_FIELD_LOGITS", True)))
            if torch.isnan(field_state).any():
                self._log_field_nan_context(field_state, sim_map, box_prior_map)
                raise ValueError("ETFM field state contains NaN after FieldUpdate")
            field_outputs.append(self._field_state_to_prob(field_state))

        tokens = torch.cat([template_tokens, search_tokens], dim=1)
        return tokens, field_state

    def forward_features(self, z, x, target_field=None, target_field_logits=None,
                         box_prior_map=None, return_field=False):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        feat_h, feat_w = H // self.patch_size, W // self.patch_size
        field_state = None
        field_outputs = []
        if self.etfm_enabled:
            field_state = self._init_or_check_target_field_logits(
                target_field_logits, target_field, B, feat_h, feat_w, x.device, x.dtype)
            check_field_tensor("field_state", field_state, B, feat_h, feat_w)
            box_prior_map = self._init_or_check_box_prior_map(box_prior_map, B, feat_h, feat_w, x.device, x.dtype)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            x, field_state = self._apply_etfm_after_block(
                i, x, lens_z, lens_x, feat_h, feat_w, field_state, box_prior_map, field_outputs)

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        if self.etfm_enabled and return_field:
            if self.etfm_update_layers and len(field_outputs) == 0:
                raise ValueError("ETFM expected at least one updated target field, got none")
            target_field = self._field_state_to_prob(field_state)
            aux_dict["target_field_logits"] = field_state if bool(getattr(self.etfm_cfg, "USE_FIELD_LOGITS", True)) \
                else prob_to_logits(target_field)
            aux_dict["target_field"] = target_field
            aux_dict["target_field_list"] = field_outputs
        return self.norm(x), aux_dict

    def forward(self, z, x, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x,
                                            target_field=kwargs.get("target_field", None),
                                            target_field_logits=kwargs.get("target_field_logits", None),
                                            box_prior_map=kwargs.get("box_prior_map", None),
                                            return_field=kwargs.get("return_field", False))

        return x, aux_dict
