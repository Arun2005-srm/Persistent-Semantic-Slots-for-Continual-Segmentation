from pathlib import Path
from functools import partial
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import yaml
from .data import SegDataset, collate, read_pair, save_manifests
from .metrics import SegMetrics, segmentation_loss, continual_scores
from .model import Segmenter
from .replay import ReplayBuffer
from .reports import write_json, epoch_reports, continual_report, tsne_report, save_triplet, metric_table


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def device_for(cfg):
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"])


def loader_for(dataset, cfg, shuffle=False, epoch_seed=None):
    generator = torch.Generator().manual_seed(cfg["seed"] if epoch_seed is None else epoch_seed)
    return DataLoader(dataset, batch_size=cfg["loader"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["loader"]["num_workers"], pin_memory=cfg["loader"]["pin_memory"],
                      worker_init_fn=seed_worker, generator=generator,
                      collate_fn=partial(collate, ignore=cfg["dataset"]["ignore_index"]))


def stages_for(cfg):
    return cfg["continual"]["stages"] or [{"name": "all", "classes": None}]


def dataset_for(cfg, splits, stage, split):
    split_name = stage.get(f"{split}_split", split)
    classes = stage.get("classes") if split == "train" else stage.get("eval_classes", stage.get("classes"))
    groups = stage.get(f"{split}_groups")
    if split_name not in splits:
        raise ValueError(f"Missing split: {split_name}")
    dataset = SegDataset(splits[split_name], cfg, split, classes, groups)
    if not len(dataset):
        raise ValueError(f"Empty {split} dataset for stage {stage['name']}")
    return dataset


def update_metrics(metrics, logits, labels, sizes):
    pred = logits.detach().argmax(1).cpu().numpy()
    truth = labels.detach().cpu().numpy()
    for p, y, (h, w) in zip(pred, truth, sizes):
        metrics.update(p[:h, :w], y[:h, :w])


@torch.no_grad()
def evaluate(model, dataset, cfg, device):
    model.eval()
    c, ignore = len(cfg["dataset"]["classes"]), cfg["dataset"]["ignore_index"]
    metrics = SegMetrics(c, ignore, cfg["metrics"]["boundary_ratio"], dataset.classes)
    weighted_loss, count = 0., 0
    for batch in loader_for(dataset, cfg):
        x, y = batch["image"].to(device), batch["mask"].to(device)
        logits = model(x)
        loss_target = y
        if dataset.classes is not None:
            scope = torch.as_tensor(dataset.classes, device=device)
            loss_target = y.masked_fill(~torch.isin(y, scope), ignore)
        loss = segmentation_loss(logits, loss_target, cfg, ignore)
        n = int((loss_target != ignore).sum())
        if loss is not None:
            weighted_loss += float(loss) * n
            count += n
        update_metrics(metrics, logits, y, batch["sizes"])
    result = metrics.compute()
    result["loss"] = weighted_loss / count if count else float("nan")
    result["labelled_pixels"] = count
    return result


def make_optimizer(model, cfg, epochs):
    t = cfg["training"]
    params = [p for p in model.parameters() if p.requires_grad]
    if t["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    elif t["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params, lr=t["lr"], weight_decay=t["weight_decay"], momentum=t["momentum"])
    else:
        raise ValueError(f"Unknown optimizer {t['optimizer']}")
    if t["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=t["min_lr"])
    elif t["scheduler"] == "polynomial":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(t["min_lr"] / t["lr"], (1 - min(epoch, epochs) / epochs) ** t["polynomial_power"]))
    elif t["scheduler"] == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    else:
        raise ValueError(f"Unknown scheduler {t['scheduler']}")
    return optimizer, scheduler


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def checkpoint(path, model, optimizer, scheduler, scaler, replay, cfg, history, matrix, scores, stage, epoch, best, completed):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "scaler": scaler.state_dict(), "replay": replay.state_dict(), "config": cfg, "history": history,
               "matrix": matrix, "scores": scores, "stage": stage, "epoch": epoch, "best": best,
               "stage_completed": completed, "rng": rng_state(),
               "encoder_config": model.encoder.encoder.config.to_dict() if hasattr(model.encoder, "encoder") else None}
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(cfg, splits, resume=None):
    seed_all(cfg["seed"])
    output = Path(cfg["output_dir"])
    if (output / "last.pt").exists() and not resume:
        raise FileExistsError(f"{output} already contains a run; choose another output_dir or --resume")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    save_manifests(splits, output / "manifests")
    if cfg["eda"].get("before_training", True) and not resume:
        from .eda import run_eda
        eda_summary = run_eda(cfg, splits, output)
        if any(v["failures"] for v in eda_summary.values()):
            raise ValueError("EDA found corrupt/malformed samples; inspect eda/*/metadata.json")
    device = device_for(cfg)
    state = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if state:
        old = dict(state["config"])
        new = dict(cfg)
        for key in ("device", "output_dir"):
            old.pop(key, None)
            new.pop(key, None)
        if old != new:
            raise ValueError("Resume requires the saved experiment config (device/output_dir may change)")
    model = Segmenter(cfg, load_pretrained=state is None, encoder_config=state.get("encoder_config") if state else None).to(device)
    write_json(output / "model_modules.json", model.module_report())
    write_json(output / "environment.json", {"torch": torch.__version__, "device": str(device), "cuda": torch.version.cuda,
                                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    replay_cfg = cfg["continual"]["replay"]
    replay = ReplayBuffer(replay_cfg["budget_bytes"], cfg["dataset"]["ignore_index"], cfg["seed"])
    history, matrix, scores = [], [], []
    if state:
        model.load_state_dict(state["model"])
        replay.load_state_dict(state["replay"])
        history, matrix, scores = state["history"], state["matrix"], state["scores"]
    stages = stages_for(cfg)
    ignore, classes = cfg["dataset"]["ignore_index"], cfg["dataset"]["classes"]
    for stage_index, stage in enumerate(stages):
        if state and (stage_index < state["stage"] or (stage_index == state["stage"] and state["stage_completed"])):
            continue
        train_set = dataset_for(cfg, splits, stage, "train")
        val_set = dataset_for(cfg, splits, stage, "val")
        if any(r.mask is None for r in train_set.records + val_set.records):
            raise ValueError("Training and validation require labelled records")
        epochs = stage.get("epochs", cfg["training"]["epochs"])
        optimizer, scheduler = make_optimizer(model, cfg, epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda")
        start, best = 0, -float("inf")
        if state and stage_index == state["stage"]:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start, best = state["epoch"] + 1, state["best"]
        if state:
            restore_rng(state["rng"])
            state = None
        for epoch in range(start, epochs):
            began = time.perf_counter()
            model.train()
            metrics = SegMetrics(len(classes), ignore, cfg["metrics"]["boundary_ratio"])
            weighted_loss, count, steps = 0., 0, 0
            for batch in loader_for(train_set, cfg, shuffle=True, epoch_seed=cfg["seed"] + stage_index * 100000 + epoch):
                x, y = batch["image"].to(device), batch["mask"].to(device)
                replay_samples = replay.sample(replay_cfg["batch_size"]) if replay_cfg["enabled"] else []
                optimizer.zero_grad(set_to_none=True)
                terms, weights = [], []
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    if (y != ignore).any():
                        logits = model(x, labels=y, update_memory=True)
                        loss = segmentation_loss(logits, y, cfg, ignore)
                        terms.append(loss)
                        weights.append(int((y != ignore).sum()))
                        update_metrics(metrics, logits, y, batch["sizes"])
                    if replay_samples:
                        rb = collate(replay_samples, ignore)
                        rx, ry = rb["image"].to(device), rb["mask"].to(device)
                        rlogits = model(rx) # Replay never updates descriptor memory.
                        rloss = segmentation_loss(rlogits, ry, cfg, ignore)
                        if rloss is not None:
                            terms.append(rloss)
                            weights.append(int((ry != ignore).sum()))
                            update_metrics(metrics, rlogits, ry, rb["sizes"])
                    if not terms:
                        continue
                    loss = sum(v * w for v, w in zip(terms, weights)) / sum(weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                weighted_loss += float(loss.detach()) * sum(weights)
                count += sum(weights)
                steps += 1
            if not steps:
                raise ValueError(f"Stage {stage['name']} has no supervised training pixels")
            train_result = metrics.compute()
            train_result["loss"] = weighted_loss / count
            val_result = evaluate(model, val_set, cfg, device)
            if not val_result["labelled_pixels"]:
                raise ValueError(f"Stage {stage['name']} has no valid validation pixels")
            history.append({"stage": stage["name"], "epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                            "seconds": time.perf_counter() - began, "train": train_result, "val": val_result})
            scheduler.step()
            fields = " ".join(f"{k}={val_result[k]:.4f}" for k in ("miou", "iou", "biou", "dice"))
            print(f"[{stage['name']}] ep[{epoch+1}/{epochs}] : train acc={train_result['accuracy']:.4f}/loss={train_result['loss']:.4f}, val acc={val_result['accuracy']:.4f}/loss={val_result['loss']:.4f}, {fields}", flush=True)
            print("TRAIN\n" + metric_table(train_result, classes) + "\nVAL\n" + metric_table(val_result, classes), flush=True)
            epoch_reports(history, output, classes)
            improved = val_result["miou"] > best
            best = max(best, val_result["miou"])
            args = (model, optimizer, scheduler, scaler, replay, cfg, history, matrix, scores, stage_index, epoch, best, False)
            checkpoint(output / "last.pt", *args)
            if improved:
                checkpoint(output / f"best_stage_{stage_index:02d}.pt", *args)
        model.consolidate()
        # Store only stage-visible labels; future classes never enter replay targets.
        if replay_cfg["enabled"]:
            for i in range(len(train_set)):
                sample = train_set[i]
                sample["id"] = f"stage{stage_index}:{sample['id']}"
                replay.add(sample)
        row = [float("nan")] * len(stages)
        for task_index in range(stage_index + 1):
            row[task_index] = evaluate(model, dataset_for(cfg, splits, stages[task_index], "val"), cfg, device)["miou"]
        matrix.append(row)
        scores.append(continual_scores(matrix))
        if len(stages) > 1:
            continual_report(matrix, scores, output, [s["name"] for s in stages])
        else:
            write_json(output / "continual_status.json", {"status": "not_applicable", "reason": "Forgetting requires at least two stages"})
        write_json(output / "replay_status.json", {"enabled": replay_cfg["enabled"], "serialized_storage_bytes": replay.bytes_used, "images": len(replay.entries), "budget_bytes": replay.budget})
        checkpoint(output / "last.pt", model, optimizer, scheduler, scaler, replay, cfg, history, matrix, scores, stage_index, epochs - 1, best, True)
        checkpoint(output / f"stage_{stage_index:02d}.pt", model, optimizer, scheduler, scaler, replay, cfg, history, matrix, scores, stage_index, epochs - 1, best, True)
        visualize(model, val_set, cfg, device, output / f"visuals_stage_{stage_index:02d}")
    return output


@torch.no_grad()
def visualize(model, dataset, cfg, device, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    features, labels = [], []
    rng = np.random.default_rng(cfg["seed"])
    reporting = cfg["reporting"]
    maximum = max(reporting["num_predictions"], reporting["tsne_max_images"] if reporting["tsne"] else 0)
    for index in range(min(len(dataset), maximum)):
        sample = dataset[index]
        logits, pixels = model(sample["image"][None].to(device), return_features=True)
        if index < reporting["num_predictions"]:
            original, original_mask = read_pair(dataset.records[index], cfg)
            # Predict on the configured transformed view; invert resize/normalize
            # by evaluating geometry-only original resize below only for resize paths.
            specs = dataset.transforms.specs
            cropped = any(s["name"] == "center_crop" for s in specs)
            if cropped:
                x = sample["image"].clone()
                norm = next((s for s in specs if s["name"] == "normalize"), None)
                if norm:
                    x = x * torch.tensor(norm["std"])[:, None, None] + torch.tensor(norm["mean"])[:, None, None]
                original = (x.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
                truth = sample["mask"].numpy() if dataset.records[index].mask else None
                pred = logits.argmax(1)[0].cpu().numpy()
                title = "Aligned evaluation crop"
            else:
                truth = original_mask
                pred = F.interpolate(logits, size=(original.height, original.width), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
                title = dataset.records[index].sample_id
            save_triplet(output / f"prediction_{index:04d}.png", original, truth, pred, cfg, title)
        if reporting["tsne"] and index < reporting["tsne_max_images"]:
            y = F.interpolate(sample["mask"][None, None].float(), size=pixels.shape[-2:], mode="nearest")[0, 0].long().flatten().numpy()
            f = pixels[0].flatten(1).T.float().cpu().numpy()
            valid = np.flatnonzero(y != cfg["dataset"]["ignore_index"])
            if dataset.classes is not None:
                valid = valid[np.isin(y[valid], dataset.classes)]
            chosen = rng.choice(valid, min(len(valid), reporting["tsne_points_per_image"]), replace=False)
            features.extend(f[chosen])
            labels.extend(y[chosen])
    if reporting["tsne"]:
        tsne_report(features, labels, cfg, output)


def test(cfg, splits, checkpoint_path, split="test", stage_index=None):
    device = device_for(cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Config controls data/reporting, but weights must match the saved architecture/taxonomy.
    if cfg["model"] != state["config"]["model"] or cfg["dataset"]["classes"] != state["config"]["dataset"]["classes"]:
        raise ValueError("Checkpoint architecture/taxonomy does not match config")
    model = Segmenter(cfg, load_pretrained=False, encoder_config=state.get("encoder_config")).to(device)
    model.load_state_dict(state["model"])
    stage = stages_for(cfg)[stage_index] if stage_index is not None else {"name": "all"}
    dataset = dataset_for(cfg, splits, stage, split)
    output = Path(cfg["output_dir"]) / f"evaluation_{split}"
    output.mkdir(parents=True, exist_ok=True)
    if any(r.mask is not None for r in dataset.records):
        result = evaluate(model, dataset, cfg, device)
        write_json(output / "metrics.json", result)
        (output / "per_class.txt").write_text(metric_table(result, cfg["dataset"]["classes"]), encoding="utf-8")
        print(metric_table(result, cfg["dataset"]["classes"]))
        print({k: result[k] for k in ("miou", "iou", "biou", "dice", "accuracy")})
    else:
        write_json(output / "metrics.json", {"status": "unavailable", "reason": "No ground truth: prediction export only"})
    visualize(model, dataset, cfg, device, output)
    if cfg["reporting"].get("export_masks", True):
        export_masks(model, dataset, cfg, device, output / "masks")
    return output


@torch.no_grad()
def export_masks(model, dataset, cfg, device, output):
    if any(s["name"] not in ("resize", "normalize") for s in dataset.transforms.specs):
        raise ValueError("Full-resolution mask export requires resize/normalize-only evaluation transforms; set reporting.export_masks=false for crop evaluation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    entries = []
    mapping = cfg["dataset"].get("label_map", {})
    inverse = {}
    for raw, target in mapping.items():
        if target not in range(len(cfg["dataset"]["classes"])):
            continue
        if target in inverse:
            inverse[target] = None
        else:
            inverse[target] = int(raw)
    raw_export = all(inverse.get(i) is not None for i in range(len(cfg["dataset"]["classes"])))
    for i, record in enumerate(dataset.records):
        sample = dataset[i]
        with Image.open(record.image) as original:
            size = (original.height, original.width)
        logits = model(sample["image"][None].to(device))
        prediction = F.interpolate(logits, size=size, mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
        dtype = np.uint8 if len(cfg["dataset"]["classes"]) <= 256 else np.uint16
        filename = f"{i:06d}_{Path(record.image).stem}.png"
        Image.fromarray(prediction.astype(dtype)).save(output / filename)
        entry = {"sample_id": record.sample_id, "source_image": record.image, "train_ids": filename}
        if raw_export:
            values = np.array([inverse[c] for c in range(len(cfg["dataset"]["classes"]))])
            raw = values[prediction]
            raw_dtype = np.uint8 if raw.min() >= 0 and raw.max() <= 255 else np.uint16
            raw_name = f"{i:06d}_{Path(record.image).stem}_rawIds.png"
            Image.fromarray(raw.astype(raw_dtype)).save(output / raw_name)
            entry["raw_ids"] = raw_name
        entries.append(entry)
    write_json(output / "manifest.json", {"classes": cfg["dataset"]["classes"], "predictions": entries})
