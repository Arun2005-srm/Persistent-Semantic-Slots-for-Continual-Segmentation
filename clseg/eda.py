"""Unaugmented, split-specific EDA; statistics are never fitted across splits."""
from pathlib import Path
from collections import Counter
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .data import read_pair
from .reports import write_json, colorize


def run_eda(cfg, splits, output):
    root = Path(output) / "eda"
    c, ignore = len(cfg["dataset"]["classes"]), cfg["dataset"]["ignore_index"]
    summary = {}
    for split, records in splits.items():
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        limit = cfg["eda"]["max_images"]
        # Seeded sample avoids systematic bias toward first city in sorted paths.
        selected = records
        if limit is not None and limit < len(records):
            rng = np.random.default_rng(cfg["seed"])
            selected = [records[i] for i in sorted(rng.choice(len(records), limit, replace=False))]
        pixels, presence = np.zeros(c, np.int64), np.zeros(c, np.int64)
        cooccur = np.zeros((c, c), np.int64)
        rgb_sum, rgb_sq = np.zeros(3), np.zeros(3)
        total, ignored, labelled_pixels, labelled_images = 0, 0, 0, 0
        widths, heights, rows, failures = [], [], [], []
        raw_counts, groups = Counter(), Counter()
        for i, record in enumerate(selected):
            try:
                image, mask = read_pair(record, cfg)
                rgb = np.array(image).astype(np.float64) / 255
                h, w = rgb.shape[:2]
                total += h * w
                rgb_sum += rgb.sum((0, 1))
                rgb_sq += (rgb * rgb).sum((0, 1))
                widths.append(w)
                heights.append(h)
                groups[record.group] += 1
                counts = np.zeros(c, np.int64)
                void = None
                if mask is not None:
                    valid = mask != ignore
                    labelled_images += 1
                    labelled_pixels += h * w
                    ignored += (~valid).sum()
                    void = float((~valid).mean())
                    counts = np.bincount(mask[valid], minlength=c)
                    pixels += counts
                    presence += counts > 0
                    cooccur += np.outer(counts > 0, counts > 0)
                    with Image.open(record.mask) as im:
                        raw = np.array(im)
                    if raw.ndim == 2:
                        ids, nums = np.unique(raw, return_counts=True)
                        raw_counts.update(dict(zip(map(str, ids), map(int, nums))))
                    else:
                        colors, nums = np.unique(raw[..., :3].reshape(-1, 3), axis=0, return_counts=True)
                        raw_counts.update({','.join(map(str, color)): int(n) for color, n in zip(colors, nums)})
                rows.append([record.sample_id, record.group, w, h, record.mask is not None, void, *counts.tolist()])
                if i < cfg["eda"]["example_images"]:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
                    axes[0].imshow(image)
                    axes[0].set_title("Original")
                    if mask is not None:
                        axes[1].imshow(colorize(mask, cfg))
                    axes[1].set_title("Mapped ground truth" if mask is not None else "Unlabelled")
                    for ax in axes:
                        ax.axis("off")
                    fig.savefig(directory / f"sample_{i:03d}.png", dpi=120)
                    plt.close(fig)
            except (ValueError, OSError) as error:
                failures.append({"id": record.sample_id, "error": str(error)})
        names = cfg["dataset"]["classes"]
        with (directory / "samples.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "group", "width", "height", "labelled", "ignored_fraction", *names])
            writer.writerows(rows)
        class_rows = [{"index": i, "name": name, "pixels": int(pixels[i]), "pixel_fraction": float(pixels[i] / max(1, pixels.sum())), "images_present": int(presence[i]), "image_fraction": float(presence[i] / max(1, labelled_images))} for i, name in enumerate(names)]
        with (directory / "classes.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(class_rows[0]))
            writer.writeheader()
            writer.writerows(class_rows)
        stats = {"total_records": len(records), "scanned": len(selected), "successful": len(rows), "sampled": len(selected) < len(records),
                 "labelled_images": labelled_images, "ignored_fraction": float(ignored / labelled_pixels) if labelled_pixels else None,
                 "rgb_mean": (rgb_sum / total).tolist() if total else None,
                 "rgb_std": np.sqrt(np.maximum(0, rgb_sq / total - (rgb_sum / total) ** 2)).tolist() if total else None,
                 "groups": dict(groups), "raw_label_counts": dict(raw_counts), "classes": class_rows, "failures": failures,
                 "class_cooccurrence": cooccur.tolist()}
        write_json(directory / "metadata.json", stats)
        summary[split] = {"images": len(records), "scanned": len(selected), "failures": len(failures), "labelled": labelled_images}
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
        for ax, counts, title in [(axes[0, 0], pixels, "Class pixel counts"), (axes[0, 1], presence, "Images containing each class")]:
            ax.bar(range(c), counts)
            ax.set_xticks(range(c), [f"{i}: {n}" for i, n in enumerate(names)], rotation=90, fontsize=8)
            ax.set_title(title)
            ax.set_yscale("symlog", linthresh=1)
        axes[1, 0].hist(widths, alpha=.6, label="width")
        axes[1, 0].hist(heights, alpha=.6, label="height")
        axes[1, 0].set_title("Image dimensions")
        axes[1, 0].legend()
        heatmap = axes[1, 1].imshow(cooccur, cmap="magma", vmin=0, vmax=max(1, int(cooccur.max())))
        fig.colorbar(heatmap, ax=axes[1, 1], label="Images")
        axes[1, 1].set_xticks(range(c))
        axes[1, 1].set_yticks(range(c))
        axes[1, 1].set(title="Class co-occurrence (images)", xlabel="Class index", ylabel="Class index")
        fig.savefig(directory / "distributions.png", dpi=140)
        plt.close(fig)
    write_json(root / "summary.json", summary)
    return summary
