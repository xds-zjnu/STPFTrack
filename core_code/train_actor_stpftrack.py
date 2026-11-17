from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import os
import torch.nn.functional as F
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from stpf import (generate_center_gaussian_field, generate_gaussian_field_from_box,
                  is_etfm_enabled, logits_to_prob, prob_to_logits)
from lib.train.data.etfm_visualization import visualize_etfm_fields


class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        if is_etfm_enabled(self.cfg):
            return self.forward_pass_etfm(data)

        # forward pass
        out_dict = self.forward_pass(data)
        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def _get_template_list(self, data):
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)
        if len(template_list) == 1:
            return template_list[0]
        return template_list

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = self._get_template_list(data)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_for_mask = template_list if torch.is_tensor(template_list) else template_list[0]
            box_mask_z = generate_mask_cond(self.cfg, template_for_mask.shape[0], template_for_mask.device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def _get_gt_field_init_prob(self, epoch):
        if not self.cfg.MODEL.ETFM.USE_GT_FIELD_FOR_TRAIN_INIT:
            return 0.0

        start = float(getattr(self.cfg.MODEL.ETFM, "GT_FIELD_INIT_PROB_START", 1.0))
        end = float(getattr(self.cfg.MODEL.ETFM, "GT_FIELD_INIT_PROB_END", start))
        decay_epochs = int(getattr(self.cfg.MODEL.ETFM, "GT_FIELD_INIT_DECAY_EPOCHS", 0))
        if decay_epochs <= 0 and start != end:
            decay_epochs = int(getattr(self.cfg.TRAIN, "EPOCH", 1))
        if decay_epochs <= 0:
            prob = start
        else:
            try:
                epoch_num = int(epoch)
            except (TypeError, ValueError):
                epoch_num = int(epoch.item())
            progress = min(max(epoch_num - 1, 0) / max(float(decay_epochs - 1), 1.0), 1.0)
            prob = start + progress * (end - start)
        return max(0.0, min(1.0, prob))

    def _should_use_gt_field_for_init(self, epoch, device):
        prob = self._get_gt_field_init_prob(epoch)
        if prob <= 0.0:
            return False, prob
        if prob >= 1.0:
            return True, prob
        return torch.rand((), device=device).item() < prob, prob

    def _field_size_from_search(self, search_img):
        feat_h = search_img.shape[-2] // self.cfg.MODEL.BACKBONE.STRIDE
        feat_w = search_img.shape[-1] // self.cfg.MODEL.BACKBONE.STRIDE
        return feat_h, feat_w

    def _gt_field_prob(self, gt_bbox, feat_h, feat_w):
        return generate_gaussian_field_from_box(
            gt_bbox, feat_h, feat_w, self.cfg.DATA.SEARCH.SIZE,
            box_format="xywh", sigma_ratio=self.cfg.MODEL.ETFM.FIELD_SIGMA)

    def _center_field_prob(self, batch_size, feat_h, feat_w, device, dtype):
        return generate_center_gaussian_field(
            batch_size, feat_h, feat_w, device, dtype=dtype, sigma_ratio=self.cfg.MODEL.ETFM.FIELD_SIGMA)

    def _box_prior_map(self, gt_bbox, feat_h, feat_w, search_img):
        if bool(getattr(self.cfg.MODEL.ETFM, "USE_BOX_PRIOR_IN_UPDATE", True)):
            return self._gt_field_prob(gt_bbox, feat_h, feat_w)
        return torch.zeros(search_img.shape[0], 1, feat_h, feat_w,
                           device=search_img.device, dtype=search_img.dtype)

    def _propagate_field_logits(self, out_dict):
        field_out_prob = out_dict["target_field"]
        feat_h, feat_w = field_out_prob.shape[-2:]

        if bool(getattr(self.cfg.MODEL.ETFM, "FIELD_PROPAGATION_MIX", True)):
            pred_box = out_dict["pred_boxes"].mean(dim=1)
            box_field_prob = generate_gaussian_field_from_box(
                pred_box, feat_h, feat_w, self.cfg.DATA.SEARCH.SIZE,
                box_format="cxcywh", sigma_ratio=self.cfg.MODEL.ETFM.FIELD_SIGMA)
            lambda_f = float(getattr(self.cfg.MODEL.ETFM, "FIELD_PROPAGATION_LAMBDA", 0.5))
            next_field_prob = lambda_f * field_out_prob + (1.0 - lambda_f) * box_field_prob
        else:
            next_field_prob = field_out_prob

        next_field_logits = prob_to_logits(next_field_prob.clamp(1e-6, 1.0 - 1e-6))
        if self.cfg.MODEL.ETFM.DETACH_PROPAGATION:
            return next_field_logits.detach()
        return next_field_logits

    def _propagate_field_state(self, out_dict):
        field_out_prob = out_dict["target_field"]
        feat_h, feat_w = field_out_prob.shape[-2:]

        if bool(getattr(self.cfg.MODEL.ETFM, "FIELD_PROPAGATION_MIX", True)):
            pred_box = out_dict["pred_boxes"].mean(dim=1)
            box_field_prob = generate_gaussian_field_from_box(
                pred_box, feat_h, feat_w, self.cfg.DATA.SEARCH.SIZE,
                box_format="cxcywh", sigma_ratio=self.cfg.MODEL.ETFM.FIELD_SIGMA)
            lambda_f = float(getattr(self.cfg.MODEL.ETFM, "FIELD_PROPAGATION_LAMBDA", 0.5))
            next_field_prob = lambda_f * field_out_prob + (1.0 - lambda_f) * box_field_prob
        else:
            next_field_prob = field_out_prob

        if bool(getattr(self.cfg.MODEL.ETFM, "USE_FIELD_LOGITS", True)):
            next_field_state = prob_to_logits(next_field_prob.clamp(1e-6, 1.0 - 1e-6))
        else:
            next_field_state = next_field_prob
        if self.cfg.MODEL.ETFM.DETACH_PROPAGATION:
            return next_field_state.detach()
        return next_field_state

    def _should_visualize_fields(self, data):
        if not bool(getattr(self.cfg.MODEL.ETFM, "VISUALIZE_FIELD", False)):
            return False
        interval = int(getattr(self.cfg.MODEL.ETFM, "VIS_INTERVAL", 500))
        if interval <= 0:
            return False
        iteration = int(data.get("iteration", 0))
        if iteration <= 0:
            return False
        settings = data.get("settings", None)
        if settings is not None and getattr(settings, "local_rank", -1) not in [-1, 0]:
            return False
        return iteration % interval == 0

    def _visualize_fields(self, data, frame_id, search_img, gt_bbox, out_dict, init_field_prob):
        if not self._should_visualize_fields(data):
            return
        try:
            settings = data.get("settings", None)
            workspace_dir = getattr(getattr(settings, "env", None), "workspace_dir", ".")
            vis_dir = getattr(self.cfg.MODEL.ETFM, "VIS_DIR", "field_vis")
            save_dir = vis_dir if os.path.isabs(vis_dir) else os.path.join(workspace_dir, vis_dir)
            epoch = int(data.get("epoch", 0))
            iteration = int(data.get("iteration", 0))
            pred_box = out_dict["pred_boxes"].mean(dim=1)
            max_samples = int(getattr(self.cfg.MODEL.ETFM, "VIS_MAX_SAMPLES", 4))
            num_samples = min(max_samples, search_img.shape[0])
            for sample_idx in range(num_samples):
                save_path = os.path.join(
                    save_dir, f"epoch{epoch:04d}_iter{iteration:06d}_frame{frame_id}_sample{sample_idx}.jpg")
                visualize_etfm_fields(
                    search_img[sample_idx].detach().cpu(),
                    gt_bbox[sample_idx].detach().cpu(),
                    pred_box[sample_idx].detach().cpu(),
                    init_field_prob[sample_idx:sample_idx + 1].detach().cpu(),
                    [f[sample_idx:sample_idx + 1].detach().cpu() for f in out_dict.get("target_field_list", [])],
                    out_dict["target_field"][sample_idx:sample_idx + 1].detach().cpu(),
                    save_path,
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD)
        except Exception as exc:
            print(f"WARNING: ETFM field visualization failed: {exc}")

    def forward_pass_etfm(self, data):
        assert len(data['template_images']) == 1

        template = self._get_template_list(data)
        num_search_frames = min(int(self.cfg.MODEL.ETFM.NUM_SEARCH_FRAMES), len(data['search_images']))
        previous_field_state = None
        total_loss = 0.0
        status_acc = {}
        field_requires_grad = []
        previous_requires_grad = []
        used_gt_field_init = 0.0
        gt_field_init_prob = self._get_gt_field_init_prob(data['epoch'])

        for frame_id in range(num_search_frames):
            search_img = data['search_images'][frame_id].view(-1, *data['search_images'].shape[2:])
            gt_bbox = data['search_anno'][frame_id]
            feat_h, feat_w = self._field_size_from_search(search_img)

            use_field_logits = bool(getattr(self.cfg.MODEL.ETFM, "USE_FIELD_LOGITS", True))
            if frame_id > 0 and self.cfg.MODEL.ETFM.PROPAGATE_FIELD and \
                    self.cfg.MODEL.ETFM.USE_PRED_FIELD_FOR_NEXT and previous_field_state is not None:
                init_field_prob = logits_to_prob(previous_field_state) if use_field_logits else previous_field_state
            else:
                use_gt_init, gt_field_init_prob = self._should_use_gt_field_for_init(data['epoch'], search_img.device)
                used_gt_field_init += float(use_gt_init)
                if use_gt_init:
                    init_field_prob = self._gt_field_prob(gt_bbox, feat_h, feat_w)
                else:
                    init_field_prob = self._center_field_prob(
                        search_img.shape[0], feat_h, feat_w, search_img.device, search_img.dtype)

            if use_field_logits:
                target_field = None
                target_field_logits = prob_to_logits(init_field_prob) if previous_field_state is None or \
                    not (frame_id > 0 and self.cfg.MODEL.ETFM.PROPAGATE_FIELD and
                         self.cfg.MODEL.ETFM.USE_PRED_FIELD_FOR_NEXT) else previous_field_state
            else:
                target_field = init_field_prob
                target_field_logits = None

            box_prior_map = self._box_prior_map(gt_bbox, feat_h, feat_w, search_img)

            out_dict = self.net(template=template,
                                search=search_img,
                                target_field=target_field,
                                target_field_logits=target_field_logits,
                                box_prior_map=box_prior_map,
                                return_field=True,
                                ce_template_mask=None,
                                ce_keep_rate=1.0,
                                return_last_attn=False)
            out_dict["init_field"] = init_field_prob

            loss_i, status_i = self.compute_losses(out_dict, data, gt_bbox=gt_bbox)
            total_loss = total_loss + loss_i
            for key, value in status_i.items():
                status_acc[key] = status_acc.get(key, 0.0) + float(value)

            field_out = out_dict["target_field"]
            field_requires_grad.append(float(field_out.requires_grad))
            if self.cfg.MODEL.ETFM.PROPAGATE_FIELD:
                previous_field_state = self._propagate_field_state(out_dict)
            else:
                previous_field_state = None
            previous_requires_grad.append(
                float(previous_field_state.requires_grad) if previous_field_state is not None else 0.0)
            self._visualize_fields(data, frame_id, search_img, gt_bbox, out_dict, init_field_prob)

        total_loss = total_loss / num_search_frames
        status = {key: value / num_search_frames for key, value in status_acc.items()}
        status["ETFM/num_search_frames"] = float(num_search_frames)
        status["ETFM/gt_field_init_prob"] = gt_field_init_prob
        status["ETFM/used_gt_field_init"] = used_gt_field_init / num_search_frames
        status["ETFM/field_0_requires_grad"] = field_requires_grad[0] if field_requires_grad else 0.0
        if previous_requires_grad:
            status["ETFM/previous_field_requires_grad"] = previous_requires_grad[-1]
        status["Loss/total"] = total_loss.item()
        return total_loss, status

    def _field_distribution(self, field):
        flat = field.flatten(1)
        return flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _field_center_of_mass(self, field):
        bsz, _, feat_h, feat_w = field.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, feat_h, device=field.device, dtype=field.dtype),
            torch.linspace(0.0, 1.0, feat_w, device=field.device, dtype=field.dtype),
            indexing="ij",
        )
        mass = field.sum(dim=(-2, -1)).clamp_min(1e-6)
        cx = (field[:, 0] * xx).sum(dim=(-2, -1)) / mass[:, 0]
        cy = (field[:, 0] * yy).sum(dim=(-2, -1)) / mass[:, 0]
        return torch.stack([cx, cy], dim=-1)

    def _compute_field_loss(self, target_field, gt_field):
        loss_type = self.cfg.MODEL.ETFM.FIELD_LOSS_TYPE
        if loss_type == "mse":
            return F.mse_loss(target_field, gt_field)
        if loss_type == "bce":
            return F.binary_cross_entropy(target_field.clamp(1e-6, 1 - 1e-6), gt_field)
        if loss_type == "soft_ce_center":
            pred_prob = self._field_distribution(target_field)
            gt_prob = self._field_distribution(gt_field)
            soft_ce = -(gt_prob * pred_prob.clamp_min(1e-6).log()).sum(dim=1).mean()
            pred_center = self._field_center_of_mass(target_field)
            gt_center = self._field_center_of_mass(gt_field)
            center_loss = F.mse_loss(pred_center, gt_center)
            center_weight = float(getattr(self.cfg.MODEL.ETFM, "FIELD_CENTER_LOSS_WEIGHT", 1.0))
            return soft_ce + center_weight * center_loss
        raise ValueError(f"Unsupported ETFM field loss type: {loss_type}")

    def _compute_deep_field_loss(self, pred_dict, gt_field):
        final_field = pred_dict["target_field"]
        all_fields = list(pred_dict.get("target_field_list", [])) \
            if bool(getattr(self.cfg.MODEL.ETFM, "USE_FIELD_DEEP_SUPERVISION", True)) else []
        if not all_fields:
            all_fields.append(final_field)

        field_losses = []
        for field_i in all_fields:
            if field_i.shape != gt_field.shape:
                raise ValueError(f"gt_field shape {tuple(gt_field.shape)} does not match "
                                 f"target_field shape {tuple(field_i.shape)}")
            if field_i.min() < -1e-6 or field_i.max() > 1.0 + 1e-6:
                raise ValueError(f"target_field probability is out of range: "
                                 f"min={float(field_i.min())}, max={float(field_i.max())}")
            field_losses.append(self._compute_field_loss(field_i, gt_field))

        field_loss = torch.stack(field_losses).mean()
        if torch.isnan(field_loss):
            print("ETFM field loss NaN:",
                  f"target min/max/mean={float(final_field.min())}/{float(final_field.max())}/{float(final_field.mean())}",
                  f"gt min/max/mean={float(gt_field.min())}/{float(gt_field.max())}/{float(gt_field.mean())}")
            raise ValueError("ETFM field_loss is NaN")
        return field_loss, len(all_fields)

    def compute_losses(self, pred_dict, gt_dict, gt_bbox=None, return_status=True):
        # gt gaussian map
        if gt_bbox is None:
            gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            heatmap_source = gt_dict['search_anno']
        else:
            heatmap_source = gt_bbox.unsqueeze(0)
        gt_gaussian_maps = generate_heatmap(heatmap_source, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        field_loss = torch.tensor(0.0, device=loss.device)
        field_supervision_count = 0
        if is_etfm_enabled(self.cfg) and self.cfg.MODEL.ETFM.USE_FIELD_LOSS and "target_field" in pred_dict:
            target_field = pred_dict["target_field"]
            feat_h, feat_w = target_field.shape[-2:]
            gt_field = generate_gaussian_field_from_box(gt_bbox, feat_h, feat_w, self.cfg.DATA.SEARCH.SIZE,
                                                        box_format="xywh",
                                                        sigma_ratio=self.cfg.MODEL.ETFM.FIELD_SIGMA)
            field_loss, field_supervision_count = self._compute_deep_field_loss(pred_dict, gt_field)
            deep_weight = float(getattr(self.cfg.MODEL.ETFM, "FIELD_DEEP_SUPERVISION_WEIGHT", 1.0))
            loss = loss + self.cfg.MODEL.ETFM.FIELD_LOSS_WEIGHT * deep_weight * field_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            target_field = pred_dict.get("target_field", None)
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/field": field_loss.item(),
                      "ETFM/field_supervision_count": float(field_supervision_count),
                      "IoU": mean_iou.item()}
            if target_field is not None:
                status["ETFM/field_min"] = target_field.detach().min().item()
                status["ETFM/field_max"] = target_field.detach().max().item()
                status["ETFM/field_mean"] = target_field.detach().mean().item()
            return loss, status
        else:
            return loss
