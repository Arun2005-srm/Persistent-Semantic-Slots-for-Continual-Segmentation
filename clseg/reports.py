from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def colorize(mask, cfg):
    classes = cfg["dataset"]["classes"]
    colors = cfg["dataset"].get("colors")
    if not colors or len(colors) != len(classes):
        colors = (plt.get_cmap("tab20")(np.arange(len(classes)) % 20)[:, :3] * 255).astype(np.uint8)
    output = np.full((*mask.shape, 3), 80, dtype=np.uint8)
    for c, color in enumerate(colors):
        output[mask == c] = color
    return output


def save_triplet(path, image, truth, prediction, cfg, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, value, name in zip(axes, [image, colorize(truth, cfg) if truth is not None else None, colorize(prediction, cfg)], ["Original image", "Ground truth", "Prediction"]):
        if value is not None:
            ax.imshow(value)
        else:
            ax.text(.5, .5, "Ground truth unavailable", ha="center", va="center")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def epoch_reports(history, output, classes):
    output = Path(output)
    write_json(output / "history.json", history)
    scalar_fields = ["loss", "accuracy", "miou", "iou", "biou", "dice", "macro_f1"]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["stage", "epoch"] + [f"{split}_{metric}" for split in ("train", "val") for metric in scalar_fields]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({"stage": row["stage"], "epoch": row["epoch"], **{f"{s}_{m}": clean(row[s][m]) for s in ("train", "val") for m in scalar_fields}})
    with (output / "per_class_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "epoch", "split", "class_index", "class", "IoU", "F1", "Boundary_IoU", "pixels"])
        for row in history:
            for split in ("train", "val"):
                for c, name in enumerate(classes):
                    writer.writerow([row["stage"], row["epoch"], split, c, name, *[clean(row[split][k][c]) for k in ("class_iou", "class_f1", "class_biou", "class_support")]])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("Training history — dashed lines separate continual stages")
    x = np.arange(1, len(history) + 1)
    for ax, metric in zip(axes.flat, ["loss", "accuracy", "iou", "miou", "biou", "dice"]):
        for split in ("train", "val"):
            ax.plot(x, [r[split][metric] for r in history], label=split)
        for i in range(1, len(history)):
            if history[i]["stage"] != history[i - 1]["stage"]:
                ax.axvline(i + .5, color="gray", ls="--", alpha=.5)
        ax.set(xlabel="Global epoch", ylabel=metric, title=metric)
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(output / "training_curves.png", dpi=140)
    plt.close(fig)


def continual_report(matrix, scores, output, names):
    output = Path(output)
    write_json(output / "continual.json", {"task_names": names, "miou_matrix": matrix, "scores": scores})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    r = np.asarray(matrix)
    image = axes[0].imshow(r, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[0].set(xlabel="Evaluated task", ylabel="After stage", title="Continual mIoU matrix")
    axes[0].set_xticks(range(len(names)), names, rotation=30, ha="right")
    axes[0].set_yticks(range(len(r)), names[:len(r)])
    fig.colorbar(image, ax=axes[0])
    axes[1].plot(range(1, len(scores) + 1), [s["forgetting"] if s["forgetting"] is not None else np.nan for s in scores], marker="o", label="Forgetting")
    axes[1].plot(range(1, len(scores) + 1), [s["backward_transfer"] if s["backward_transfer"] is not None else np.nan for s in scores], marker="o", label="Backward transfer")
    axes[1].set(xlabel="Completed stage", ylabel="mIoU difference", title="Retention across stages")
    axes[1].set_xticks(range(1, len(scores) + 1))
    axes[1].set_xlim(.8, len(scores) + .2)
    axes[1].legend()
    axes[1].grid(alpha=.2)
    fig.savefig(output / "forgetting_curve.png", dpi=140)
    plt.close(fig)


def tsne_report(features, labels, cfg, output):
    from sklearn.manifold import TSNE
    features, labels = np.asarray(features), np.asarray(labels)
    maximum = cfg["reporting"]["tsne_max_points"]
    rng = np.random.default_rng(cfg["seed"])
    indices = rng.choice(len(labels), min(len(labels), maximum), replace=False)
    features, labels = features[indices], labels[indices]
    if len(labels) < 4:
        write_json(Path(output) / "tsne_status.json", {"status": "skipped", "reason": "Fewer than four labelled feature samples"})
        return
    coordinates = TSNE(n_components=2, perplexity=min(30, max(2, (len(labels) - 1) / 3)), init="random", random_state=cfg["seed"], learning_rate="auto").fit_transform(features)
    np.savez_compressed(Path(output) / "tsne_samples.npz", features=features, labels=labels, coordinates=coordinates)
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for c in np.unique(labels):
        select = labels == c
        ax.scatter(*coordinates[select].T, s=8, label=cfg["dataset"]["classes"][int(c)], alpha=.7)
    ax.set(title="Pixel-decoder features: t-SNE (qualitative diagnostic)", xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), markerscale=2)
    fig.savefig(Path(output) / "tsne.png", dpi=150)
    plt.close(fig)


def metric_table(metrics, classes):
    def fmt(x):
        return "N/A" if x is None or not np.isfinite(x) else f"{x:.4f}"
    lines = [f"{'class':22} {'IoU':>9} {'F1/Dice':>9} {'BIoU':>9}"]
    for c, name in enumerate(classes):
        lines.append(f"{name:22} {fmt(metrics['class_iou'][c]):>9} {fmt(metrics['class_f1'][c]):>9} {fmt(metrics['class_biou'][c]):>9}")
    return "\n".join(lines)
