"""Dataset-neutral discovery, paired transforms, stage masking and padded batches."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Record:
    image: str
    mask: str | None
    group: str
    sample_id: str


def discover(cfg):
    d = cfg["dataset"]
    if not d["root"] or not Path(d["root"]).is_dir():
        raise ValueError("Set dataset.root in your config to an existing dataset directory")
    root = Path(d["root"]).resolve()
    splits = {}
    for name, spec in d["splits"].items():
        if spec is None:
            continue
        rows = []
        if "manifest" in spec:
            with (root / spec["manifest"]).open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    rows.append(Record(str((root / r["image"]).resolve()), str((root / r["mask"]).resolve()) if r.get("mask") else None, r.get("group", ""), r.get("id", r.get("sample_id", r["image"]))))
        else:
            base = root / spec.get("image_base", "")
            for image in sorted(root.glob(spec["images"])):
                relative = image.relative_to(base)
                stem = image.stem
                suffix = spec.get("strip_suffix", "")
                if suffix and stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                mask = spec.get("masks")
                if mask:
                    mask = str((root / mask.format(relative=relative.as_posix(), parent=relative.parent.as_posix(), stem=stem, name=image.name)).resolve())
                rows.append(Record(str(image.resolve()), mask, relative.parent.as_posix(), image.relative_to(root).as_posix()))
        for r in rows:
            if not Path(r.image).is_file() or (r.mask and not Path(r.mask).is_file()):
                raise FileNotFoundError(f"Missing image/mask pair: {r}")
        if len({r.image for r in rows}) != len(rows) or len({r.sample_id for r in rows}) != len(rows):
            raise ValueError(f"Duplicate image or sample ID in {name}")
        splits[name] = rows
    part = d.get("partition")
    if part:
        source = splits.pop(part["source"])
        ratios = part["ratios"]
        if abs(sum(ratios.values()) - 1) > 1e-6 or any(v <= 0 for v in ratios.values()):
            raise ValueError("Partition ratios must be positive and sum to one")
        key = (lambda r: r.group) if part.get("by_group", True) else (lambda r: r.sample_id)
        groups = sorted({key(r) for r in source})
        if len(groups) < len(ratios):
            raise ValueError("Too few groups for the requested partition")
        random.Random(cfg["seed"]).shuffle(groups)
        start = 0
        for i, (name, ratio) in enumerate(ratios.items()):
            if name in splits:
                raise ValueError(f"Partition would overwrite existing split {name}")
            end = len(groups) if i == len(ratios) - 1 else start + round(len(groups) * ratio)
            chosen = set(groups[start:end])
            splits[name] = [r for r in source if key(r) in chosen]
            if not splits[name]:
                raise ValueError(f"Empty partition {name}; adjust ratios/group count")
            start = end
    seen_images, seen_masks = {}, {}
    for split, rows in splits.items():
        for r in rows:
            for value, seen in ((r.image, seen_images), (r.mask, seen_masks)):
                if value and value in seen and seen[value] != split:
                    raise ValueError(f"Split leakage: {value} occurs in {seen[value]} and {split}")
                if value:
                    seen[value] = split
    return splits


def read_pair(record, cfg):
    d = cfg["dataset"]
    with Image.open(record.image) as im:
        image = im.convert("RGB")
    if record.mask is None:
        return image, None
    with Image.open(record.mask) as im:
        raw = np.array(im)
    if raw.shape[:2] != (image.height, image.width):
        raise ValueError(f"Image/mask size mismatch: {record.sample_id}")
    mask = np.full(raw.shape[:2], d["ignore_index"], dtype=np.int64)
    if raw.ndim == 3:
        if not d.get("rgb_label_map"):
            raise ValueError("RGB masks require dataset.rgb_label_map")
        for color, target in d["rgb_label_map"].items():
            rgb = [int(v) for v in color.split(",")]
            mask[np.all(raw[..., :3] == rgb, axis=-1)] = target
    elif d.get("label_map"):
        for source, target in d["label_map"].items():
            mask[raw == int(source)] = target
    else:
        mask = raw.astype(np.int64)
        valid = (mask == d["ignore_index"]) | ((mask >= 0) & (mask < len(d["classes"])))
        if not valid.all():
            raise ValueError(f"Unmapped mask indices in {record.sample_id}: {np.unique(mask[~valid])}")
    return image, mask


class PairedTransforms:
    def __init__(self, specs, ignore):
        self.specs, self.ignore = specs, ignore
        allowed = {"resize", "random_resize", "random_crop", "center_crop", "random_horizontal_flip", "color_jitter", "normalize"}
        for s in specs:
            if s["name"] not in allowed:
                raise ValueError(f"Unknown transform {s['name']}")

    def __call__(self, image, mask):
        mask = Image.fromarray(mask.astype(np.int32)) if mask is not None else None
        mean, std = [0, 0, 0], [1, 1, 1]
        for s in self.specs:
            name = s["name"]
            if name in ("resize", "random_resize"):
                if name == "resize":
                    h, w = s["size"]
                else:
                    scale = random.uniform(*s["scale"])
                    h, w = max(1, round(image.height * scale)), max(1, round(image.width * scale))
                image = image.resize((w, h), Image.Resampling.BILINEAR)
                if mask is not None:
                    mask = mask.resize((w, h), Image.Resampling.NEAREST)
            elif name in ("random_crop", "center_crop"):
                h, w = s["size"]
                pad = (0, 0, max(0, w - image.width), max(0, h - image.height))
                image = ImageOps.expand(image, pad, fill=0)
                if mask is not None:
                    mask = ImageOps.expand(mask, pad, fill=self.ignore)
                x = random.randint(0, image.width - w) if name == "random_crop" else (image.width - w) // 2
                y = random.randint(0, image.height - h) if name == "random_crop" else (image.height - h) // 2
                image = image.crop((x, y, x + w, y + h))
                if mask is not None:
                    mask = mask.crop((x, y, x + w, y + h))
            elif name == "random_horizontal_flip" and random.random() < s.get("p", .5):
                image = ImageOps.mirror(image)
                if mask is not None:
                    mask = ImageOps.mirror(mask)
            elif name == "color_jitter":
                for field, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast), ("saturation", ImageEnhance.Color)):
                    strength = s.get(field, 0)
                    image = enhancer(image).enhance(random.uniform(max(0, 1 - strength), 1 + strength))
            elif name == "normalize":
                mean, std = s["mean"], s["std"]
        x = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 255
        x = (x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        y = torch.from_numpy(np.array(mask).astype(np.int64)) if mask is not None else torch.full((image.height, image.width), self.ignore, dtype=torch.long)
        return x, y


class SegDataset(Dataset):
    def __init__(self, records, cfg, split, classes=None, groups=None):
        self.records = [r for r in records if groups is None or r.group in groups]
        self.cfg, self.classes = cfg, classes
        self.split = split
        self.transforms = PairedTransforms(cfg["transforms"][split], cfg["dataset"]["ignore_index"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        image, mask = read_pair(r, self.cfg)
        if mask is not None and self.classes is not None and self.split == "train":
            mask = np.where(np.isin(mask, self.classes), mask, self.cfg["dataset"]["ignore_index"])
        x, y = self.transforms(image, mask)
        return {"image": x, "mask": y, "id": r.sample_id, "labelled": r.mask is not None}


def collate(samples, ignore=255):
    h = max(s["image"].shape[-2] for s in samples)
    w = max(s["image"].shape[-1] for s in samples)
    images = torch.zeros(len(samples), 3, h, w)
    masks = torch.full((len(samples), h, w), ignore, dtype=torch.long)
    sizes = []
    for i, s in enumerate(samples):
        sh, sw = s["mask"].shape
        images[i, :, :sh, :sw] = s["image"]
        masks[i, :sh, :sw] = s["mask"]
        sizes.append((sh, sw))
    return {"image": images, "mask": masks, "sizes": sizes, "id": [s["id"] for s in samples]}


def save_manifests(splits, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "mask", "group", "sample_id"])
            writer.writeheader()
            writer.writerows(asdict(r) for r in records)
