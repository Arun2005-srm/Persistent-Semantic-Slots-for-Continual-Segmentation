# Configurable continual semantic segmentation

This implements the modular experimental framework for **Persistent Semantic Slots** in `architecture_notes.md`. It is a research implementation, not an experimentally validated method or a reproduction of FALCON. No real dataset or pretrained weights are bundled. Cityscapes' directory is intentionally blank in `configs/cityscapes.yaml`.

## Start

Use Python 3.10+ and install the dependencies in `requirements.txt` in your chosen environment. Install a PyTorch build appropriate for your GPU. The default model downloads `nvidia/mit-b2` on its first training run; subsequent checkpoint loading reconstructs the saved backbone architecture offline.

```powershell
python -m pip install -r requirements.txt
python -m clseg validate --config configs/cityscapes.yaml
```

Set `dataset.root` to the directory containing `leftImg8bit/` and `gtFine/`, then:

```powershell
python -m clseg eda --config configs/cityscapes.yaml
python -m clseg train --config configs/cityscapes.yaml
python -m clseg test --config configs/cityscapes.yaml --checkpoint runs/cityscapes/last.pt --split val
```

Training runs EDA automatically by default. Set `eda.before_training: false` to skip a repeated scan after inspecting your separate EDA run. Statistics are descriptive; validation/test statistics are never used to fit normalization, sampling, or model parameters.

```powershell
python -m clseg train --config configs/cityscapes.yaml --set training.lr=0.00005 --set training.epochs=100 --set output_dir=runs/cityscapes_lr5e5
python -m clseg train --config configs/cityscapes.yaml --resume runs/cityscapes/last.pt
python -m pytest -q tests
```

Resume with the same configuration (including command-line overrides); only `device` and `output_dir` can change. Runs refuse to overwrite an existing training checkpoint without `--resume`. Only load trusted locally produced checkpoints: training checkpoints contain optimizer and Python RNG state and use PyTorch's full checkpoint format.

## Dataset configuration

Create a YAML file extending `base.yaml`; standard paired semantic-mask datasets need no dataset-specific Python class. Configure:

- `root`, ordered `classes`, `colors`, `ignore_index`, raw `label_map` or `rgb_label_map`.
- `splits`: any named sets of paired files, configured using glob patterns or CSV manifests.
- Independent train, validation and test transformations.
- Epochs, optimizer, learning rate, scheduler, weight decay, class weights, CE/Dice loss switches, precision, batching, replay budget, model components and reporting limits.

An empty `label_map` means integer/palette-index masks already contain contiguous indices `0..C-1` plus the ignore index. Nonempty maps send unspecified raw IDs to ignore. RGB masks require a map such as `rgb_label_map: {"128,64,128": 0, "0,0,142": 1}`. Image dimensions must match their masks. Class IDs remain globally stable across stages.

### Folder patterns

```yaml
dataset:
  root: D:/datasets/example
  classes: [road, person, car]
  colors: [[128,64,128], [220,20,60], [0,0,142]]
  label_map: {7: 0, 24: 1, 26: 2}
  splits:
    train:
      images: rgb/train/*/*_image.png
      image_base: rgb/train
      strip_suffix: _image
      masks: labels/train/{parent}/{stem}_mask.png
    val:
      images: rgb/val/*/*_image.png
      image_base: rgb/val
      strip_suffix: _image
      masks: labels/val/{parent}/{stem}_mask.png
    test: null
```

Available template fields: `{relative}` (relative image path), `{parent}`, `{stem}` (optionally stripped), and `{name}`. `masks: null` means unlabelled images; setting the whole split to `null` disables it. By default `group` is the image's relative parent directory, e.g. a Cityscapes city.

### Manifests and custom splits

Use `train: {manifest: splits/train.csv}` instead of a glob specification. CSV columns are `image,mask,group,id`; `image` is required, paths are relative to `dataset.root` or absolute. Leave `mask` empty for unlabelled inference. `group` can identify a city, subject, sequence, site or acquisition session. Saved run manifests are reusable, including their `sample_id` column.

To derive disjoint splits, discover one source set and partition its groups:

```yaml
dataset:
  splits:
    train: {manifest: all_labelled.csv}
    val: null
    test: null
  partition:
    source: train
    by_group: true
    ratios: {train: 0.7, val: 0.2, test: 0.1}
```

The seed fixes the partition. Ratios apply to **group counts**, not image counts. Set `by_group: false` only for genuinely independent samples. For Cityscapes benchmark experiments retain the official split. File-path overlap and reused masks across splits are rejected; content-identical copies under different paths are not currently detected. Use manifests to enforce sequence-level isolation if parent folders are too coarse.

### Paired transformations

Supported names: `resize` (`size: [height,width]`), `random_resize` (`scale: [min,max]`), `random_crop` / `center_crop` (`size`), `random_horizontal_flip` (`p`), `color_jitter` (`brightness`, `contrast`, `saturation`), and `normalize` (`mean`, `std`). All geometric choices are shared by image and mask; masks use nearest-neighbour interpolation; padding labels use ignore. Normalization is applied after image operations. Stochastic validation/test transforms are rejected. Batches pad variable dimensions, but a consistent configured resolution is recommended, particularly for scene-pooled memory retrieval.

## Continual protocols

The default `stages: []` intentionally selects ordinary single-stage training until the actual continual protocol is chosen. No FALCON stage order or domain sequence is assumed.

For class-incremental experiments, specify global training indices:

```yaml
continual:
  stages:
    - {name: base, classes: [0, 1, 2], epochs: 40}
    - {name: next, classes: [3, 4], epochs: 20}
  replay: {enabled: true, budget_bytes: 67108864, batch_size: 2}
```

This example requires a taxonomy of at least five classes. Stage training hides all classes not listed in `classes` by mapping them to ignore. This is **partial-label supervision**, not a background-remapping implementation of an existing benchmark. A stage with only one active class has no softmax discrimination; design the initial stage accordingly. All discovered train images are retained, and all-ignore batches are skipped. Explicit manifests can implement a protocol's image inclusion rules.

Each stage can specify `train_split`, `val_split`, `test_split`, `train_groups`, `val_groups`, `test_groups`, `classes`, `eval_classes`, and `epochs`. Domain-incremental stages omit `classes` and select domain-specific manifests or group lists. Mixed class/domain stages combine them. Evaluation sets and class scopes must stay fixed across stages for meaningful forgetting. Use separate train and validation records within every domain; never train on official validation cities to simulate a domain protocol.

Validation/test retain all available ground-truth labels when building confusion matrices and boundaries. `eval_classes` (defaulting to the stage's `classes`) selects the reported class scores and macro averages, while other labelled pixels still supply false-positive evidence. Validation loss and accuracy are restricted to target pixels in the evaluation scope. This prevents hiding known negative pixels or turning every class border into an ignored border. Future labels used for evaluation never enter training, replay or semantic memory.

After each stage the runner evaluates every seen task and saves `R[t,j]`, the mIoU on task `j` after stage `t`. Reported forgetting averages `max(R[j:t,j]) - R[t,j]` over earlier tasks. Backward transfer averages `R[t,j] - R[j,j]`. Negative forgetting indicates improvement. Single-stage runs report CL metrics as not applicable, with no fabricated forgetting curve.

Replay samples stored classes uniformly, then examples within the selected class. It stores transformed RGB tensors and stage-visible masks; later labels cannot leak into older replay entries. Entries include their source-stage identity so annotations from subsequent stages do not overwrite earlier supervision. Storage is bounded by the serialized entries including metadata, not by example count. This budget excludes Python allocator overhead, RNG state and semantic banks. Oversize samples are not stored; inspect `replay_status.json`. Replay uses stored transformed views, without fresh augmentation. Replay never updates semantic memory. Optimizer and scheduler restart at stage boundaries; learning continues from the last epoch, not from the validation-best checkpoint.

## Model modules and ablations

Required input/output contracts remain present: an encoder, a pixel projection and a mask head are necessary to produce predictions. Optional mechanisms are real construction-time booleans, not output multipliers. A disabled module has no parameters, forward execution, or gradient path. Remaining/fallback paths still receive gradients where appropriate.

| Switch | Enabled | Disabled replacement |
|---|---|---|
| `adapters` | Residual bottleneck adapters | Unadapted encoder features |
| `multiscale_fusion` | Project and fuse all scales | Project first scale only |
| `descriptor` | Frozen fixed projection | No descriptor branch; both memory banks must be off |
| `identity_memory` | Consolidated class identities | Appearance-only query input, or learned queries if both banks off |
| `appearance_memory` | Bounded appearance slots | Identity-only query input, or learned queries if both banks off |
| `query_decoder` | Transformer query decoding | Initial class queries feed mask head directly |

`encoder.frozen` controls whether encoder gradients flow; it is not an encoder-removal switch. Persistent memory requires it to be true. Config validation rejects unused descriptor branches and incompatible dependencies. CE, Dice and replay also have independent enable switches. The provisional unlabelled bank is intentionally absent because the proposal supplies no supervised benefit without later verified annotation.

Example replay baseline override:

```yaml
extends: cityscapes.yaml
output_dir: runs/replay_baseline
model:
  modules:
    descriptor: false
    identity_memory: false
    appearance_memory: false
continual:
  replay: {enabled: true}
```

The fixed descriptor projection and frozen encoder define stable memory coordinates. Verified regions update class identities during their first labelled stage, then identities freeze. Appearance slots merge/update within class. Updates happen once per current source image per epoch and never during validation/test. An unseen class remains inactive until verified positive evidence arrives. Semantic banks are buffers included in checkpoints. `model_modules.json` records constructed components and trainable parameter counts.

## Metrics and outputs

Each epoch prints train accuracy/loss, validation accuracy/loss, validation mIoU, aggregate IoU, Boundary IoU and Dice, followed by **both train and validation per-class IoU/F1/Boundary IoU** tables. All aggregate train metrics are also saved. Training metrics include replay when enabled and are measured online before each optimizer update. Accuracy means valid-pixel accuracy, not mean class accuracy.

- Confusion counts are accumulated across the whole split before computing metrics.
- `miou`: unweighted mean of defined class IoUs in the evaluation scope.
- `iou`: micro/aggregate intersection-over-union across that scope.
- `dice` / `macro_f1`: mean per-class hard segmentation Dice. Per-class F1 and Dice are mathematically identical here.
- `biou`: mean of per-class boundary intersection/union, accumulated over images. Boundary width is `max(1, round(boundary_ratio * image_diagonal))`, default 0.02. Erosion uses a square neighbourhood and treats image borders as boundaries. Ignore neighbourhoods are excluded so void borders do not count as segmentation errors. This is semantic Boundary IoU, not instance/panoptic AP or `min(mask IoU, boundary IoU)`.
- A class absent in both truth and prediction has undefined IoU/F1 and is excluded from macro averages; predictions of a class absent from truth count as false positives. Classes outside an explicit evaluation scope are marked unavailable. JSON uses `null`, tables use `N/A`.
- CE and soft Dice both exclude ignored pixels. Dice loss averages classes with positives in the batch. Logged loss is a valid-pixel-weighted average of batch objectives, so the Dice component depends on batch composition.
- Metrics use configured evaluation resolution/crop. Resize affects boundary measurements: keep validation/test transforms identical across comparisons. Visual outputs may upsample predictions to original image size, without changing the resolution of reported metrics.

Artifacts include:

| Output | Contents |
|---|---|
| `eda/<split>/` | Class index/pixel counts, image presence, fractions, raw label counts, group counts, RGB moments, dimensions, co-occurrence, corrupt sample log, example GT maps |
| `history.json`, `history.csv` | Epoch aggregate metrics, with detailed records in JSON |
| `per_class_history.csv` | Epoch/split/class IoU, F1, Boundary IoU and support |
| `training_curves.png` | Train/validation loss, accuracy, IoU, mIoU, Boundary IoU and Dice |
| `continual.json`, `forgetting_curve.png` | Seen-task mIoU matrix, forgetting and backward transfer for multi-stage runs |
| `visuals_stage_XX/` | Configurable N original/GT/prediction triplets, bounded t-SNE samples and map |
| `evaluation_<split>/` | Test metrics and qualitative reports |
| `last.pt`, `stage_XX.pt`, `best_stage_XX.pt` | Full resumable model, memories, optimizer/scheduler/scaler, replay, RNG, config and history |
| `resolved_config.yaml`, `manifests/` | Exact settings and resolved data membership |

t-SNE samples labelled pixel-decoder features; it is a qualitative diagnostic, not evidence of separation or retention. Sampling is bounded by images, points/image and total points. It is skipped with a recorded reason when fewer than four usable points exist. Triplets use the original image when evaluation geometry is resize-only; configured crops are labelled as aligned evaluation crops. The number of examples is `min(reporting.num_predictions, dataset size)`.

Official Cityscapes test labels are withheld. Use `--split val` for local labelled evaluation, or a separately defined labelled test split. The runner never fabricates metrics for an unlabelled split.

Test runs also export all predicted masks at original resolution under `evaluation_<split>/masks/`, with a manifest linking each output to its input. Training-index masks are always exported; uniquely invertible label maps additionally produce raw-ID masks (including Cityscapes label IDs). This is not an automatic submission to the evaluation server. Set `reporting.export_masks: false` to disable full export. Full-resolution export requires resize/normalization-only evaluation transforms; crop-based evaluation still supports aligned triplets and metrics.

## Verification and remaining research work

`tests/` covers known metric values, ignore handling, Boundary IoU, ablation gradient flow, inference memory immutability, identity consolidation, mapping, leakage detection, replay budgets, checkpoint round trips and two-stage end-to-end training/reporting. `tiny` is a random small backbone for offline tests only. Cityscapes-scale training, the actual pretrained backbone download, performance tuning and research comparisons still require data and a chosen continual protocol. This is currently a single-process training runner; distributed training is not implemented.

To produce a fully offline synthetic demo with EDA, training curves, forgetting curves, t-SNE and prediction triplets:

```powershell
python scripts/make_toy_dataset.py
python -m clseg train --config configs/smoke.yaml
python -m clseg test --config configs/smoke.yaml --checkpoint runs/smoke_demo/last.pt
```

These deliberately tiny synthetic experiments validate execution only; their metrics do not support any accuracy or novelty claim.

Sources: [Cityscapes](https://www.cityscapes-dataset.com/), [official label definitions](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py), [FALCON project](https://uark-cviu.github.io/projects/FALCON/), [Boundary IoU reference](https://github.com/bowenc0221/boundary-iou-api).
