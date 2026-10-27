from copy import deepcopy
from pathlib import Path
import os
import yaml


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path, overrides=(), _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    seen.add(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(os.path.expandvars(f.read()))
    parent = raw.pop("extends", None)
    cfg = merge(load_config(path.parent / parent, _seen=seen), raw) if parent else raw
    for item in overrides:
        key, value = item.split("=", 1)
        cursor = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(value)
    return cfg


def validate(cfg):
    data, model = cfg["dataset"], cfg["model"]
    c = len(data["classes"])
    if c < 2 or len(set(data["classes"])) != c:
        raise ValueError("At least two unique class names are required")
    if data["ignore_index"] in range(c):
        raise ValueError("ignore_index must be outside the class indices")
    for target in data.get("label_map", {}).values():
        if target not in range(c) and target != data["ignore_index"]:
            raise ValueError(f"Invalid label mapping target {target}")
    for target in data.get("rgb_label_map", {}).values():
        if target not in range(c) and target != data["ignore_index"]:
            raise ValueError(f"Invalid RGB mapping target {target}")
    if cfg["metrics"]["boundary_ratio"] <= 0:
        raise ValueError("boundary_ratio must be positive")
    if type(cfg["metrics"].get("train_boundary", True)) is not bool:
        raise ValueError("metrics.train_boundary must be a boolean")
    loss = cfg["training"]["loss"]
    if not loss["cross_entropy"] and not loss["dice"]:
        raise ValueError("At least one loss must be enabled")
    if loss.get("class_weights") is not None and (len(loss["class_weights"]) != c or any(w <= 0 for w in loss["class_weights"])):
        raise ValueError("class_weights must contain a positive weight per class")
    if cfg["training"]["lr"] <= 0:
        raise ValueError("Learning rate must be positive")
    if model["memory"]["temperature"] <= 0:
        raise ValueError("Memory attention temperature must be positive")
    flags = model["modules"]
    expected = {"adapters", "multiscale_fusion", "descriptor", "identity_memory", "appearance_memory", "query_decoder"}
    if set(flags) != expected or any(type(v) is not bool for v in flags.values()):
        raise ValueError(f"modules must contain exactly these boolean switches: {sorted(expected)}")
    memory = flags["identity_memory"] or flags["appearance_memory"]
    if memory and not flags["descriptor"]:
        raise ValueError("Memory requires descriptor=true")
    if flags["descriptor"] and not memory:
        raise ValueError("Disable descriptor when both memory banks are off; it would be unused")
    if memory and not model["encoder"]["frozen"]:
        raise ValueError("Persistent memory requires a frozen encoder to keep descriptor coordinates stable")
    if model["width"] % model["heads"] or model["width"] % 4:
        raise ValueError("Model width must be divisible by attention heads and by four")
    if model["memory"]["slots"] < 1:
        raise ValueError("At least one appearance slot is required")
    if cfg["training"]["epochs"] < 1 or cfg["loader"]["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    for split in ("val", "test"):
        for t in cfg["transforms"][split]:
            if t["name"].startswith("random") or t["name"] == "color_jitter":
                raise ValueError(f"Stochastic {split} transform is forbidden: {t['name']}")
    names = []
    for stage in cfg["continual"]["stages"]:
        names.append(stage["name"])
        if stage.get("epochs", 1) < 1:
            raise ValueError("Stage epochs must be positive")
        for field in ("classes", "eval_classes"):
            ids = stage.get(field)
            if ids is not None and (not ids or any(i not in range(c) for i in ids)):
                raise ValueError(f"Invalid {field} in {stage['name']}")
    if len(set(names)) != len(names):
        raise ValueError("Stage names must be unique")
    return cfg
