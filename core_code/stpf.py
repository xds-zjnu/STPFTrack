import torch
import torch.nn as nn
import torch.nn.functional as F

_ETFM_DENSE_TOKEN_NOTICE_PRINTED = False


def _as_bool(value):
    return bool(value) if value is not None else False


def is_etfm_enabled(cfg):
    return hasattr(cfg.MODEL, "ETFM") and _as_bool(cfg.MODEL.ETFM.ENABLE)


def log_etfm_dense_token_notice_once():
    global _ETFM_DENSE_TOKEN_NOTICE_PRINTED
    if _ETFM_DENSE_TOKEN_NOTICE_PRINTED:
        return
    print("ETFM requires dense search tokens to maintain the spatial target field. "
          "Candidate elimination is disabled in the first implementation.")
    _ETFM_DENSE_TOKEN_NOTICE_PRINTED = True


def generate_center_gaussian_field(batch_size, feat_h, feat_w, device, dtype=torch.float32, sigma_ratio=0.25):
    center = torch.tensor([[0.5, 0.5, 0.25, 0.25]], device=device, dtype=dtype).repeat(batch_size, 1)
    return generate_gaussian_field_from_box(center, feat_h, feat_w, search_size=1, box_format="cxcywh",
                                            sigma_ratio=sigma_ratio)


def prob_to_logits(p, eps=1e-6):
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def logits_to_prob(logits):
    return torch.sigmoid(logits)


def check_field_tensor(name, tensor, batch_size, feat_h, feat_w):
    expected_shape = (batch_size, 1, feat_h, feat_w)
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")


def generate_gaussian_field_from_box(box, feat_h, feat_w, search_size, box_format="xywh", sigma_ratio=0.25):
    """Create a spatial target field from normalized or pixel-space boxes.

    The field is patch-aligned with search tokens. It is used by ETFM as a spatial
    target prior and as the cross-frame state propagated during tracking.
    """
    if box is None:
        raise ValueError("box must be provided. Use generate_center_gaussian_field when no box is available.")

    if box.dim() != 2 or box.shape[-1] != 4:
        raise ValueError(f"box must have shape (B, 4), got {tuple(box.shape)}")

    box = box.float()
    device = box.device
    dtype = box.dtype

    if box_format == "xywh":
        cx = box[:, 0] + 0.5 * box[:, 2]
        cy = box[:, 1] + 0.5 * box[:, 3]
        bw = box[:, 2]
        bh = box[:, 3]
    elif box_format == "cxcywh":
        cx, cy, bw, bh = box.unbind(dim=-1)
    else:
        raise ValueError(f"Unsupported box_format={box_format}")

    # Training annotations in OSTrack are normalized xywh. Pixel boxes are also accepted.
    max_coord = torch.stack([cx.abs(), cy.abs(), bw.abs(), bh.abs()], dim=-1).amax()
    if max_coord > 2.0:
        search_size_tensor = torch.as_tensor(search_size, device=device, dtype=dtype).clamp_min(1.0)
        cx = cx / search_size_tensor
        cy = cy / search_size_tensor
        bw = bw / search_size_tensor
        bh = bh / search_size_tensor

    cx_feat = cx.clamp(0.0, 1.0) * (feat_w - 1)
    cy_feat = cy.clamp(0.0, 1.0) * (feat_h - 1)
    w_feat = bw.clamp_min(0.0) * feat_w
    h_feat = bh.clamp_min(0.0) * feat_h

    sigma_x = torch.maximum(w_feat * 0.5, torch.ones_like(w_feat))
    sigma_y = torch.maximum(h_feat * 0.5, torch.ones_like(h_feat))
    if sigma_ratio is not None and sigma_ratio > 0:
        max_sigma = float(sigma_ratio) * max(feat_h, feat_w)
        sigma_x = torch.maximum(sigma_x, torch.full_like(sigma_x, max_sigma))
        sigma_y = torch.maximum(sigma_y, torch.full_like(sigma_y, max_sigma))

    yy, xx = torch.meshgrid(
        torch.arange(feat_h, device=device, dtype=dtype),
        torch.arange(feat_w, device=device, dtype=dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    field = torch.exp(-(((xx - cx_feat[:, None, None]) ** 2) / (2 * sigma_x[:, None, None] ** 2 + 1e-6) +
                        ((yy - cy_feat[:, None, None]) ** 2) / (2 * sigma_y[:, None, None] ** 2 + 1e-6)))
    field = field.unsqueeze(1)
    field_min = field.amin(dim=(-2, -1), keepdim=True)
    field_max = field.amax(dim=(-2, -1), keepdim=True)
    return (field - field_min) / (field_max - field_min + 1e-6)


class FieldModulation(nn.Module):
    def __init__(self, dim, bottleneck=64, scale_init=1e-3):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, search_tokens, target_field, use_centered=True, field_is_logits=True):
        bsz, hw, channels = search_tokens.shape
        if target_field.dim() != 4 or target_field.shape[0] != bsz or target_field.shape[1] != 1:
            raise ValueError(f"target_field must have shape (B, 1, Hf, Wf), got {tuple(target_field.shape)}")

        feat_h, feat_w = target_field.shape[-2:]
        if hw != feat_h * feat_w:
            raise ValueError(
                f"ETFM requires dense search tokens, but got Lx={hw}, Hf*Wf={feat_h * feat_w}. "
                f"Please disable candidate elimination."
            )

        field_prob = logits_to_prob(target_field) if field_is_logits else target_field
        field = field_prob.flatten(2).transpose(1, 2)
        if field.shape[1] != hw:
            raise ValueError(f"FieldModulation shape mismatch: HW={hw}, field_HW={field.shape[1]}")
        if use_centered:
            field = 2.0 * field - 1.0
        delta = self.up(self.act(self.down(search_tokens)))
        return search_tokens + self.scale * field * delta


class FieldUpdate(nn.Module):
    def __init__(self, scale_init=1e-3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, target_field, sim_map, box_prior_map, use_logits=True):
        if target_field.shape != sim_map.shape or target_field.shape != box_prior_map.shape:
            raise ValueError(
                f"FieldUpdate expects matching shapes, got target={tuple(target_field.shape)}, "
                f"sim={tuple(sim_map.shape)}, prior={tuple(box_prior_map.shape)}"
            )

        if not use_logits:
            inp = torch.cat([target_field, sim_map, box_prior_map], dim=1)
            delta = self.conv(inp)
            updated = target_field + self.scale * delta
            return torch.clamp(updated, 0.0, 1.0)

        field_prob = logits_to_prob(target_field)
        inp = torch.cat([field_prob, sim_map, box_prior_map], dim=1)
        delta_logits = self.conv(inp)
        return target_field + self.scale * delta_logits


def compute_template_search_similarity(template_tokens, search_tokens, feat_h, feat_w):
    bsz, lx, _ = search_tokens.shape
    if lx != feat_h * feat_w:
        raise ValueError(
            f"ETFM requires dense search tokens, but got Lx={lx}, Hf*Wf={feat_h * feat_w}. "
            f"Please disable candidate elimination."
        )
    template_proto = template_tokens.mean(dim=1, keepdim=True)
    template_proto = F.normalize(template_proto, dim=-1)
    search_norm = F.normalize(search_tokens, dim=-1)
    sim = torch.matmul(search_norm, template_proto.transpose(1, 2))
    sim = sim.transpose(1, 2).reshape(bsz, 1, feat_h, feat_w)
    sim_min = sim.amin(dim=(-2, -1), keepdim=True)
    sim_max = sim.amax(dim=(-2, -1), keepdim=True)
    return (sim - sim_min) / (sim_max - sim_min + 1e-6)


def freeze_backbone_for_etfm(model, cfg):
    if not is_etfm_enabled(cfg):
        return

    if not _as_bool(cfg.MODEL.ETFM.FREEZE_BACKBONE):
        return

    backbone = model.backbone
    for _, param in backbone.named_parameters():
        param.requires_grad = False

    for module_name in ["field_modulations", "field_updates"]:
        module = getattr(backbone, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    train_head = _as_bool(cfg.MODEL.ETFM.TRAIN_HEAD)
    for param in model.box_head.parameters():
        param.requires_grad = train_head

    trainable = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
    print("ETFM trainable parameters:")
    for name, _ in trainable:
        print(name)
    print(f"ETFM trainable parameter count: {sum(count for _, count in trainable)}")
