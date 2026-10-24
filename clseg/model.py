"""Switchable segmentation components with explicit replacement paths."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class TinyEncoder(nn.Module):
    """Random small encoder for offline integration tests, not a research backbone."""
    channels = [16, 32, 64, 128]

    def __init__(self):
        super().__init__()
        blocks, previous = [], 3
        for i, channels in enumerate(self.channels):
            blocks.append(nn.Sequential(nn.Conv2d(previous, channels, 3, stride=4 if i == 0 else 2, padding=1), nn.GroupNorm(4, channels), nn.GELU()))
            previous = channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class SegFormerEncoder(nn.Module):
    def __init__(self, name, pretrained=True, architecture=None):
        super().__init__()
        from transformers import SegformerConfig, SegformerModel
        self.encoder = SegformerModel.from_pretrained(name) if pretrained else SegformerModel(SegformerConfig(**architecture))
        self.channels = self.encoder.config.hidden_sizes

    def forward(self, x):
        return list(self.encoder(x, output_hidden_states=True).hidden_states)


class Adapter(nn.Module):
    def __init__(self, channels, bottleneck):
        super().__init__()
        self.path = nn.Sequential(nn.Conv2d(channels, bottleneck, 1), nn.GELU(), nn.Conv2d(bottleneck, channels, 1))

    def forward(self, x):
        return x + self.path(x)


class PixelDecoder(nn.Module):
    def __init__(self, channels, width, multiscale):
        super().__init__()
        self.multiscale = multiscale
        self.projections = nn.ModuleList(nn.Conv2d(c, width, 1) for c in (channels if multiscale else channels[:1]))
        if multiscale:
            self.fusion = nn.Conv2d(width * len(channels), width, 1)

    def forward(self, features):
        projected = [F.interpolate(p(f), size=features[0].shape[-2:], mode="bilinear", align_corners=False) for p, f in zip(self.projections, features)]
        return self.fusion(torch.cat(projected, 1)) if self.multiscale else projected[0]


class SemanticMemory(nn.Module):
    def __init__(self, classes, width, cfg, identity, appearance):
        super().__init__()
        self.identity, self.appearance = identity, appearance
        self.threshold, self.temperature = cfg["similarity_threshold"], cfg["temperature"]
        self.register_buffer("active", torch.zeros(classes, dtype=torch.bool))
        if identity:
            self.register_buffer("identities", torch.zeros(classes, width))
            self.register_buffer("identity_sum", torch.zeros(classes, width))
            self.register_buffer("identity_count", torch.zeros(classes))
            self.register_buffer("consolidated", torch.zeros(classes, dtype=torch.bool))
        if appearance:
            self.register_buffer("appearances", torch.zeros(classes, cfg["slots"], width))
            self.register_buffer("appearance_sum", torch.zeros(classes, cfg["slots"], width))
            self.register_buffer("counts", torch.zeros(classes, cfg["slots"]))

    @torch.no_grad()
    def update(self, descriptors, labels, ignore):
        # Pool at descriptor resolution, weighting by exact annotated area. This
        # avoids losing thin/rare classes through nearest-neighbour downsampling.
        for descriptor, label in zip(descriptors.detach(), labels):
            for c in label.unique().tolist():
                if c == ignore:
                    continue
                weight = F.interpolate((label == c)[None, None].float(), size=descriptor.shape[-2:], mode="area")[0, 0]
                z = F.normalize((descriptor * weight).sum((1, 2)) / weight.sum().clamp_min(1e-8), dim=0)
                self.active[c] = True
                if self.identity and not self.consolidated[c]:
                    self.identity_sum[c] += z
                    self.identity_count[c] += 1
                    self.identities[c] = F.normalize(self.identity_sum[c], dim=0)
                if self.appearance:
                    occupied = self.counts[c] > 0
                    similarity = self.appearances[c] @ z
                    similarity[~occupied] = -2
                    nearest = int(similarity.argmax())
                    if occupied.any() and similarity[nearest] >= self.threshold:
                        k = nearest
                    elif (~occupied).any():
                        k = int((~occupied).nonzero()[0])
                    elif len(occupied) == 1:
                        k = 0
                    else:
                        pair_sim = self.appearances[c] @ self.appearances[c].T
                        pair_sim.fill_diagonal_(-2)
                        first, second = divmod(int(pair_sim.argmax()), len(occupied))
                        self.appearance_sum[c, first] += self.appearance_sum[c, second]
                        self.counts[c, first] += self.counts[c, second]
                        self.appearances[c, first] = F.normalize(self.appearance_sum[c, first], dim=0)
                        self.appearance_sum[c, second].zero_()
                        self.counts[c, second] = 0
                        k = second
                    self.appearance_sum[c, k] += z
                    self.counts[c, k] += 1
                    self.appearances[c, k] = F.normalize(self.appearance_sum[c, k], dim=0)

    @torch.no_grad()
    def consolidate(self):
        if self.identity:
            self.consolidated |= self.active

    def forward(self, descriptor):
        batch = descriptor.shape[0]
        pieces = []
        if self.identity:
            pieces.append(self.identities[None].expand(batch, -1, -1))
        if self.appearance:
            g = F.normalize(descriptor.mean((2, 3)), dim=1)
            scores = torch.einsum("bd,ckd->bck", g, self.appearances) / self.temperature
            scores = scores.masked_fill(self.counts[None] == 0, -1e4)
            weights = scores.softmax(-1) * (self.counts[None] > 0)
            pieces.append(torch.einsum("bck,ckd->bcd", weights, self.appearances))
        return torch.cat(pieces, -1)


def position_encoding(h, w, width, device, dtype):
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device), indexing="ij")
    freq = 10000 ** (-torch.arange(width // 4, device=device).float() / max(1, width // 4))
    x, y = x.flatten()[:, None] * freq * 2 * math.pi, y.flatten()[:, None] * freq * 2 * math.pi
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], -1).to(dtype)[None]


class Segmenter(nn.Module):
    def __init__(self, cfg, load_pretrained=True, encoder_config=None):
        super().__init__()
        self.cfg = cfg
        m = cfg["model"]
        flags = m["modules"]
        width, c = m["width"], len(cfg["dataset"]["classes"])
        encoder = m["encoder"]
        if encoder["name"] == "tiny":
            self.encoder = TinyEncoder()
        elif encoder["name"] == "segformer":
            if not load_pretrained and encoder_config is None:
                raise ValueError("Offline checkpoint construction requires its saved encoder config")
            self.encoder = SegFormerEncoder(encoder["pretrained"], load_pretrained, encoder_config)
        else:
            raise ValueError(f"Unknown encoder: {encoder['name']}")
        self.encoder_frozen = encoder["frozen"]
        if self.encoder_frozen:
            self.encoder.requires_grad_(False)
        channels = self.encoder.channels
        if flags["adapters"]:
            # Do not construct unused scale adapters in the single-scale ablation.
            self.adapters = nn.ModuleList(Adapter(ch, m["adapter_width"]) for ch in (channels if flags["multiscale_fusion"] else channels[:1]))
        self.pixel_decoder = PixelDecoder(channels, width, flags["multiscale_fusion"])
        self.has_memory = flags["identity_memory"] or flags["appearance_memory"]
        self.register_buffer("active_classes", torch.zeros(c, dtype=torch.bool))
        if flags["descriptor"]:
            with torch.random.fork_rng():
                torch.manual_seed(cfg["seed"])
                self.descriptor = nn.Conv2d(channels[0], width, 1, bias=False)
            self.descriptor.requires_grad_(False)
        if self.has_memory:
            self.memory = SemanticMemory(c, width, m["memory"], flags["identity_memory"], flags["appearance_memory"])
            self.query_projection = nn.Linear(width * (int(flags["identity_memory"]) + int(flags["appearance_memory"])), width)
        else:
            self.learned_queries = nn.Parameter(torch.randn(c, width) * .02)
        if flags["query_decoder"]:
            layer = nn.TransformerDecoderLayer(width, m["heads"], width * 4, dropout=0, batch_first=True, norm_first=True)
            self.query_decoder = nn.TransformerDecoder(layer, m["decoder_layers"])
        self.mask_head = nn.Linear(width, width)

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, x, labels=None, update_memory=False, return_features=False):
        if update_memory and (not self.training or labels is None):
            raise ValueError("Memory updates require training mode and verified labels")
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.encoder_frozen):
            raw = self.encoder(x)
        descriptor = None
        if self.has_memory:
            with torch.no_grad():
                descriptor = F.normalize(self.descriptor(raw[0].detach()).float(), dim=1)
                if update_memory:
                    self.memory.update(descriptor, labels, self.cfg["dataset"]["ignore_index"])
        if update_memory:
            with torch.no_grad():
                classes = labels.unique()
                classes = classes[classes != self.cfg["dataset"]["ignore_index"]]
                self.active_classes[classes] = True
        features = [adapter(feature) for adapter, feature in zip(self.adapters, raw)] if hasattr(self, "adapters") else raw
        pixels = self.pixel_decoder(features)
        queries = self.query_projection(self.memory(descriptor)) if self.has_memory else self.learned_queries[None].expand(x.shape[0], -1, -1)
        active = self.memory.active if self.has_memory else self.active_classes
        if hasattr(self, "query_decoder"):
            tokens = pixels.flatten(2).transpose(1, 2)
            tokens = tokens + position_encoding(*pixels.shape[-2:], pixels.shape[1], x.device, pixels.dtype)
            # At least one active class is required by the training/evaluation runner.
            padding = (~active)[None].expand(x.shape[0], -1) if active.any() else None
            queries = self.query_decoder(queries, tokens, tgt_key_padding_mask=padding)
        logits = torch.einsum("bcd,bdhw->bchw", self.mask_head(queries), pixels)
        logits = logits.masked_fill(~active[None, :, None, None], -1e4)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return (logits, pixels) if return_features else logits

    def consolidate(self):
        if self.has_memory:
            self.memory.consolidate()

    def module_report(self):
        return {"switches": self.cfg["model"]["modules"], "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "constructed_modules": [name for name, _ in self.named_children()]}
