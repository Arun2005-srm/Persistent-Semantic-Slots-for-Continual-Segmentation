"""Generate visibly synthetic data to check the pipeline; not benchmark evidence."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/smoke_data")
    args = parser.parse_args()
    root = Path(args.root)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    rng = np.random.default_rng(42)
    for split, count in (("train", 8), ("val", 4), ("test", 4)):
        for folder in ("images", "masks"):
            (root / folder / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            h, w = 48, 64
            yy, xx = np.mgrid[:h, :w]
            mask = np.zeros((h, w), np.uint8)
            shift = int(rng.integers(-4, 5))
            mask[8:32, 8 + shift:28 + shift] = 1
            mask[(xx - 46) ** 2 + (yy - (26 + shift)) ** 2 < 100] = 2
            colors = np.array([[30, 40, 60], [220, 80, 60], [60, 180, 100]])
            image = np.clip(colors[mask] + rng.normal(0, 12, (h, w, 3)), 0, 255).astype(np.uint8)
            mask[:2] = 255
            Image.fromarray(image).save(root / "images" / split / f"{i:03d}.png")
            Image.fromarray(mask).save(root / "masks" / split / f"{i:03d}.png")
    print(f"Synthetic dataset created at {root.resolve()}")


if __name__ == "__main__":
    main()
