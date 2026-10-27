"""Exclude dimension mismatches without modifying dataset files or guessing rotations."""
import argparse
import json
from pathlib import Path
import sys

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clseg.config import load_config, validate
from clseg.data import discover, save_manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='runs/bdd100k_filtered_data')
    args = parser.parse_args()
    cfg = validate(load_config(args.config))
    splits = discover(cfg)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; choose another --output')
    kept, excluded = {}, []
    for split, records in splits.items():
        kept[split] = []
        for record in records:
            if record.mask:
                with Image.open(record.image) as image, Image.open(record.mask) as mask:
                    if image.size != mask.size:
                        excluded.append(dict(split=split, id=record.sample_id,
                                             image=record.image, mask=record.mask,
                                             image_size=list(image.size), mask_size=list(mask.size)))
                        continue
            kept[split].append(record)
        print(f'{split}: kept {len(kept[split])}/{len(records)}; excluded {len(records) - len(kept[split])}')
        if records and not kept[split]:
            raise ValueError(f'Filtering would empty {split}; no output written')
    output.mkdir(parents=True)
    save_manifests(kept, output)
    (output / 'excluded.json').write_text(json.dumps(excluded, indent=2), encoding='utf-8')
    cfg['dataset']['root'] = str(Path(cfg['dataset']['root']).resolve())
    cfg['dataset']['splits'] = {split: {'manifest': str(output / f'{split}.csv')} for split in kept}
    cfg['dataset']['partition'] = None  # Membership is already resolved in the manifests.
    cfg['output_dir'] = str(output / 'training')
    cfg['eda']['before_training'] = True
    cfg['eda']['max_images'] = None
    config_path = output / 'config.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print(f'Config: {config_path}')
    print(f'Exclusion report: {output / "excluded.json"}')
    print('Original files are unchanged. Metrics use filtered splits, not the full official benchmark.')


if __name__ == '__main__':
    main()
