# Verification record

Verified on 2026-10-24 in the available Windows environment using Python 3.10, PyTorch 2.5.1, and Transformers 5.6.2.

- `python -m pytest -q tests`: **19 passed**.
- Cityscapes configuration validates with its intentionally empty dataset root.
- Offline SegFormer integration uses a small randomly initialized configuration to exercise the real Transformers backbone API. Pretrained MIT-B2 weights were not downloaded.
- `configs/smoke.yaml` completed two stages with replay, EDA, checkpoint saves, per-class reports, t-SNE, and prediction triplets. The final verified run is in `runs/smoke_verified/`.
- Test evaluation completed and exported predicted masks and a source manifest.
- Training, EDA, t-SNE, prediction-triplet and forgetting plots were visually inspected.

Covered checks include known confusion-matrix scores, Boundary IoU and ignore-region invariance, scoped evaluation preserving false positives, ignored-pixel gradients, optional-module ablations, frozen-encoder gradients, inference memory immutability, identity consolidation, checkpoint round trips, label mapping, split leakage, byte-bounded replay, continual scores, and a two-stage end-to-end run with resume.

The synthetic outputs validate software execution only. No Cityscapes accuracy, pretrained-backbone performance, novelty, or publication-readiness result is established. The actual continual stage protocol remains configurable and unselected.
