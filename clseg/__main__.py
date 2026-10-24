import argparse
from pathlib import Path
import yaml
from .config import load_config, validate
from .data import discover, save_manifests


def main():
    parser = argparse.ArgumentParser(description="Config-driven continual semantic segmentation")
    parser.add_argument("command", choices=["validate", "eda", "train", "test"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--stage", type=int)
    args = parser.parse_args()
    cfg = validate(load_config(args.config, args.set))
    if args.command == "validate":
        print("Configuration is valid. Dataset files and pretrained weights have not been loaded.")
        print(yaml.safe_dump(cfg["model"]["modules"]))
        return
    splits = discover(cfg)
    if args.command == "eda":
        from .eda import run_eda
        save_manifests(splits, Path(cfg["output_dir"]) / "manifests")
        print(run_eda(cfg, splits, cfg["output_dir"]))
    elif args.command == "train":
        from .engine import train
        print(train(cfg, splits, args.resume))
    elif args.command == "test":
        if not args.checkpoint:
            parser.error("test requires --checkpoint")
        from .engine import test
        print(test(cfg, splits, args.checkpoint, args.split, args.stage))


if __name__ == "__main__":
    main()
