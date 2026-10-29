"""SOCLR-Segformer inference models; state-dict keys match released checkpoints.

SegFormer/MiT is provided by segmentation_models_pytorch. Original architecture:
https://github.com/NVlabs/SegFormer (retain its dependency license restrictions).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp

class SegFormer(smp.Segformer):
    pass

ARMS = ("selective_refine", "direction_gate", "context_verify")
class GuidedAggregate(nn.Module):
    def __init__(self, channels=128, inner=32, radius=2):
        super().__init__()
        self.radius = radius
        self.value = nn.Conv2d(channels, inner, 1, bias=False)
        self.output = nn.Conv2d(inner, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(math.log(math.expm1(.05))))
        self._grids = {}

    def _grid(self, x, direction, distance, sign):
        _, _, h, w = x.shape
        key = (h, w, x.device, x.dtype, direction, distance, sign)
        if key not in self._grids:
            yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                                    torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
            angle = direction * math.pi / 8
            xx = xx + sign * distance * math.cos(angle)
            yy = yy + sign * distance * math.sin(angle)
            self._grids[key] = torch.stack((2 * (xx + .5) / w - 1,
                                            2 * (yy + .5) / h - 1), -1)[None]
        return self._grids[key].expand(x.shape[0], -1, -1, -1)

    def forward(self, feature, direction_probability, position_probability):
        if feature.shape[1] != 128 or direction_probability.shape[1] != 8:
            raise ValueError("Stage2 channel mismatch")
        value = self.value(feature)
        pooled = torch.zeros_like(value)
        for direction in range(8):
            candidate = value
            for distance in range(1, self.radius + 1):
                positive = F.grid_sample(value, self._grid(value, direction, distance, 1),
                                         padding_mode="border", align_corners=False)
                negative = F.grid_sample(value, self._grid(value, direction, distance, -1),
                                         padding_mode="border", align_corners=False)
                candidate = candidate + (positive + negative) * .5
            pooled = pooled + direction_probability[:, direction:direction + 1] * F.gelu(candidate)
        pooled = pooled / (pooled.square().mean(dim=1, keepdim=True) + 1e-6).sqrt()
        return feature + F.softplus(self.alpha) * position_probability * self.output(pooled)

class DirectionPositionDetail(nn.Module):
    def __init__(self, encoder_weights="imagenet"):
        super().__init__()
        self.core = SegFormer(encoder_name="mit_b2", encoder_weights=encoder_weights,
                              in_channels=3, classes=1)
        self.direction_head = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
                                            nn.Conv2d(128, 8, 1))
        self.position_head = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
                                           nn.Conv2d(64, 1, 1))
        self.aggregate = GuidedAggregate()
        self.detail = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.detail[-1].weight)
        nn.init.zeros_(self.detail[-1].bias)
        self.last_direction_logits = None
        self.last_position_logits = None

    def forward(self, image):
        self.core.check_input_shape(image)
        features = list(self.core.encoder(image))
        if features[2].shape[1] != 64 or features[3].shape[1] != 128:
            raise RuntimeError(f"Unexpected encoder stage shapes: {[tuple(f.shape) for f in features]}")
        stage2 = features[3]
        normalized = F.layer_norm(stage2.permute(0, 2, 3, 1), (128,), eps=1e-6).permute(0, 3, 1, 2)
        self.last_direction_logits = self.direction_head(normalized)
        self.last_position_logits = self.position_head(normalized)
        features[3] = self.aggregate(stage2, self.last_direction_logits.softmax(1),
                                     self.last_position_logits.sigmoid())
        logits = self.core.segmentation_head(self.core.decoder(features))
        detail = F.interpolate(self.detail(features[2]), size=logits.shape[-2:],
                               mode="bilinear", align_corners=False)
        return logits + detail

class FullZoomModel(DirectionPositionDetail):
    def __init__(self, arm, encoder_weights="imagenet"):
        if arm != "full":
            raise ValueError(arm)
        super().__init__(encoder_weights=encoder_weights)
        self.arm = arm
        self.denoise = None
        self.multiscale = None

    def forward(self, image):
        self.core.check_input_shape(image)
        features = list(self.core.encoder(image))
        if features[2].shape[1] != 64 or features[3].shape[1] != 128 or features[4].shape[1] != 320:
            raise RuntimeError(f"Unexpected encoder stages: {[tuple(f.shape) for f in features]}")
        raw_stage1, stage2 = features[2], features[3]
        normalized = F.layer_norm(stage2.permute(0, 2, 3, 1), (128,), eps=1e-6).permute(0, 3, 1, 2)
        self.last_direction_logits = self.direction_head(normalized)
        self.last_position_logits = self.position_head(normalized)
        position = self.last_position_logits.sigmoid()
        if self.denoise is not None:
            features[2] = self.denoise(raw_stage1, position)
        features[3] = self.aggregate(stage2, self.last_direction_logits.softmax(1), position)
        if self.multiscale is not None:
            features[4] = self.multiscale(features[4])
        logits = self.core.segmentation_head(self.core.decoder(features))
        detail = F.interpolate(self.detail(raw_stage1), size=logits.shape[-2:],
                               mode="bilinear", align_corners=False)
        return logits + detail

class ExperimentalModel(FullZoomModel):
    def __init__(self, arm: str, encoder_weights="imagenet"):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__("full", encoder_weights=encoder_weights)
        self.arm = arm
        self.last_native_probability = None
        if arm == "selective_refine":
            self.native_project = nn.Conv2d(64, 16, 1)
            self.refine = nn.Sequential(nn.Conv2d(17, 16, 3, padding=1), nn.GELU(),
                                        nn.Conv2d(16, 1, 1))
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)
        elif arm == "direction_gate":
            self.direction_gate = nn.Conv2d(130, 1, 1)
            nn.init.zeros_(self.direction_gate.weight)
            nn.init.constant_(self.direction_gate.bias, math.log(9))  # sigmoid = 0.9
        else:
            self.context_reduce = nn.Conv2d(320, 16, 1)
            self.context_candidate = nn.Sequential(nn.Conv2d(65, 32, 3, padding=1), nn.GELU(),
                                                   nn.Conv2d(32, 1, 1))
            self.context_gate = nn.Sequential(nn.Conv2d(64 + 16 + 1, 32, 3, padding=1),
                                              nn.GELU(), nn.Conv2d(32, 1, 1))
            nn.init.zeros_(self.context_candidate[-1].weight)
            nn.init.zeros_(self.context_candidate[-1].bias)
            nn.init.zeros_(self.context_gate[-1].bias)

    def forward(self, image):
        self.core.check_input_shape(image)
        self.last_native_probability = None
        features = list(self.core.encoder(image))
        if [features[k].shape[1] for k in (2, 3, 4)] != [64, 128, 320]:
            raise RuntimeError("Unexpected encoder stage channels")
        stage1, stage2, stage3 = features[2], features[3], features[4]
        normalized = F.layer_norm(stage2.permute(0, 2, 3, 1), (128,), eps=1e-6)
        normalized = normalized.permute(0, 3, 1, 2)
        self.last_direction_logits = self.direction_head(normalized)
        self.last_position_logits = self.position_head(normalized)
        direction = self.last_direction_logits.softmax(1)
        position = self.last_position_logits.sigmoid()
        aggregated = self.aggregate(stage2, direction, position)
        if self.arm == "direction_gate":
            entropy = -(direction * direction.clamp_min(1e-8).log()).sum(1, keepdim=True)
            entropy = entropy / math.log(8)
            evidence = torch.cat((normalized, entropy, position), dim=1)
            confidence = self.direction_gate(evidence).sigmoid()
            features[3] = stage2 + confidence * (aggregated - stage2)
            self.last_gate_mean = confidence.detach().mean()
        else:
            features[3] = aggregated
            self.last_gate_mean = None
        logits = self.core.segmentation_head(self.core.decoder(features))
        detail = F.interpolate(self.detail(stage1), size=logits.shape[-2:],
                               mode="bilinear", align_corners=False)
        logits = logits + detail
        if self.arm == "selective_refine":
            if stage1.shape[-2:] != (120, 216) or logits.shape[-2:] != (480, 864):
                raise RuntimeError("Native refinement geometry mismatch")
            # All refinement decisions are made on the original 378x672 grid.
            native_coarse = F.interpolate(logits[:, :, :, :840].sigmoid(),
                                          size=(384, 672), mode="bilinear",
                                          align_corners=False)[:, :, :378]
            native_feature = F.interpolate(self.native_project(stage1[:, :, :, :210]),
                                           size=(384, 672), mode="bilinear",
                                           align_corners=False)[:, :, :378]
            correction = self.refine(torch.cat((native_feature, native_coarse.detach()), dim=1))
            uncertainty = 4 * native_coarse.detach() * (1 - native_coarse.detach())
            native_delta = uncertainty * correction
            self.last_native_probability = torch.sigmoid(
                torch.logit(native_coarse.clamp(1e-6, 1 - 1e-6)) + native_delta).detach()
            # Project the native correction to the zoomed training canvas so the
            # original frozen segmentation/direction/position losses stay intact.
            zoom_delta = F.interpolate(F.pad(native_delta, (0, 0, 0, 6)),
                                       size=(480, 840), mode="bilinear",
                                       align_corners=False)
            logits = logits + F.pad(zoom_delta, (0, 24))
        elif self.arm == "context_verify":
            coarse = F.interpolate(logits.detach().sigmoid(), size=stage1.shape[-2:],
                                   mode="bilinear", align_corners=False)
            candidate = self.context_candidate(torch.cat((stage1, coarse), dim=1))
            context = F.interpolate(self.context_reduce(stage3), size=stage1.shape[-2:],
                                    mode="bilinear", align_corners=False)
            confidence = self.context_gate(torch.cat((stage1, context, coarse), dim=1)).sigmoid()
            correction = F.interpolate(2 * confidence * candidate,
                                       size=logits.shape[-2:], mode="bilinear",
                                       align_corners=False)
            logits = logits + correction
            self.last_gate_mean = confidence.detach().mean()
        return logits

def build_model(arm, encoder_weights=None):
    if arm == "full": return FullZoomModel("full", encoder_weights=encoder_weights)
    if arm not in ("selective_refine", "direction_gate"):
        raise ValueError("Released arms: full, selective_refine, direction_gate")
    return ExperimentalModel(arm, encoder_weights=encoder_weights)
