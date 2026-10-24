from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from clseg.config import load_config, validate
from clseg.data import discover, read_pair, SegDataset, collate
from clseg.model import Segmenter
from clseg.metrics import SegMetrics, segmentation_loss, continual_scores
from clseg.replay import ReplayBuffer


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def tiny_config():
    cfg = load_config(ROOT / "configs/base.yaml")
    cfg["model"].update(width=16, heads=4, decoder_layers=1, adapter_width=4)
    cfg["model"]["encoder"] = {"name": "tiny", "frozen": True}
    cfg["device"] = "cpu"
    cfg["training"].update(epochs=1, amp=False)
    cfg["loader"].update(batch_size=2, num_workers=0, pin_memory=False)
    cfg["transforms"] = {s: [{"name": "resize", "size": [32, 48]}] for s in ("train", "val", "test")}
    cfg["reporting"].update(num_predictions=2, tsne=True, tsne_max_images=2, tsne_points_per_image=8, tsne_max_points=16)
    return cfg


def make_data(root, count=4):
    for split in ("train", "val", "test"):
        for folder in ("images", "masks"):
            (root / folder / split).mkdir(parents=True)
        for i in range(count):
            mask = np.zeros((32, 48), np.uint8)
            mask[8:24, 12 + i:36 + i] = 1
            mask[:2] = 255
            image = np.zeros((32, 48, 3), np.uint8)
            image[..., 0] = np.where(mask == 1, 240, 20)
            image[..., 1] = 50 + i * 20
            Image.fromarray(image).save(root / "images" / split / f"{i}.png")
            Image.fromarray(mask).save(root / "masks" / split / f"{i}.png")


def test_metrics_known_confusion_and_void():
    m = SegMetrics(3)
    m.update(np.array([[0, 1, 1, 2]]), np.array([[0, 0, 1, 255]]))
    result = m.compute()
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["class_iou"][:2] == pytest.approx([.5, .5])
    assert np.isnan(result["class_iou"][2])
    assert result["class_f1"][:2] == pytest.approx([2 / 3, 2 / 3])
    assert result["miou"] == .5


def test_boundary_perfect_and_ignore_invariance():
    target = np.zeros((32, 32), np.int64)
    target[8:24, 8:24] = 1
    target[:, :2] = 255
    pred = target.copy()
    pred[:, :2] = 0
    m = SegMetrics(2)
    m.update(pred, target)
    assert m.compute()["biou"] == 1
    pred[:, :2] = 1
    n = SegMetrics(2)
    n.update(pred, target)
    assert np.array_equal(m.bi, n.bi) and np.array_equal(m.bu, n.bu)
    shifted = np.roll(pred, 3, axis=1)
    n = SegMetrics(2)
    n.update(shifted, target)
    assert n.compute()["biou"] < 1


def test_evaluation_scope_keeps_negative_evidence():
    metric = SegMetrics(2, eval_classes=[1])
    metric.update(np.array([[1, 1, 1, 1]]), np.array([[0, 0, 1, 1]]))
    result = metric.compute()
    assert result["miou"] == .5 # False positives on class 0 still count.
    assert result["accuracy"] == 1 # Accuracy uses target pixels in scope.
    assert np.isnan(result["class_iou"][0])


def test_loss_ignores_void_and_all_void():
    cfg = tiny_config()
    logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    y = torch.zeros(1, 4, 4, dtype=torch.long)
    y[:, :2] = 255
    loss = segmentation_loss(logits, y, cfg, 255)
    loss.backward()
    assert torch.all(logits.grad[:, :, :2] == 0)
    assert segmentation_loss(logits, torch.full_like(y, 255), cfg, 255) is None


@pytest.mark.parametrize("disable", [[], ["adapters"], ["multiscale_fusion"], ["identity_memory"], ["appearance_memory"], ["query_decoder"], ["identity_memory", "appearance_memory", "descriptor"], ["adapters", "multiscale_fusion", "identity_memory", "appearance_memory", "descriptor", "query_decoder"]])
def test_ablation_gradients(disable):
    cfg = tiny_config()
    for name in disable:
        cfg["model"]["modules"][name] = False
    validate(cfg)
    model = Segmenter(cfg).train()
    x = torch.randn(2, 3, 32, 48)
    y = torch.randint(0, 2, (2, 32, 48))
    logits = model(x, y, update_memory=True)
    segmentation_loss(logits, y, cfg, 255).backward()
    assert logits.shape == (2, 2, 32, 48)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name
    if "adapters" in disable:
        assert not hasattr(model, "adapters")
    if "query_decoder" in disable:
        assert not hasattr(model, "query_decoder")
    if "descriptor" in disable:
        assert not hasattr(model, "descriptor") and not hasattr(model, "memory")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        model(x)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    with pytest.raises(ValueError):
        model(x, y, update_memory=True)


def test_identity_consolidates_and_checkpoint_roundtrip():
    cfg = tiny_config()
    model = Segmenter(cfg).train()
    x = torch.randn(1, 3, 32, 48)
    y = torch.zeros(1, 32, 48, dtype=torch.long)
    model(x, y, update_memory=True)
    model.consolidate()
    identity = model.memory.identities.clone()
    model(x * 2, y, update_memory=True)
    assert torch.equal(identity, model.memory.identities)
    assert not model.memory.active[1]
    model.eval()
    other = Segmenter(cfg).eval()
    other.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.equal(model(x), other(x))


def test_invalid_module_dependency():
    cfg = tiny_config()
    cfg["model"]["modules"]["descriptor"] = False
    with pytest.raises(ValueError, match="descriptor"):
        validate(cfg)


def test_segformer_adapter_offline():
    cfg = tiny_config()
    cfg["model"]["encoder"] = {"name": "segformer", "pretrained": "unused-offline", "frozen": True}
    architecture = {"hidden_sizes": [8, 16, 32, 64], "depths": [1, 1, 1, 1], "num_attention_heads": [1, 2, 4, 8], "sr_ratios": [8, 4, 2, 1]}
    model = Segmenter(cfg, load_pretrained=False, encoder_config=architecture).train()
    x = torch.randn(1, 3, 64, 64)
    y = torch.randint(0, 2, (1, 64, 64))
    logits = model(x, y, update_memory=True)
    segmentation_loss(logits, y, cfg, 255).backward()
    assert logits.shape == (1, 2, 64, 64)
    assert all(p.grad is None for p in model.encoder.parameters())


def test_data_mapping_stage_mask_and_leakage(tmp_path):
    make_data(tmp_path)
    cfg = tiny_config()
    cfg["dataset"]["root"] = str(tmp_path)
    splits = discover(cfg)
    dataset = SegDataset(splits["train"], cfg, "train", classes=[1])
    assert set(dataset[0]["mask"].unique().tolist()) == {1, 255}
    validation = SegDataset(splits["val"], cfg, "val", classes=[1])
    assert set(validation[0]["mask"].unique().tolist()) == {0, 1, 255}
    cfg["dataset"]["label_map"] = {0: 1, 1: 0}
    _, mask = read_pair(splits["train"][0], cfg)
    assert mask[3, 3] == 1 and mask[12, 20] == 0 and mask[0, 0] == 255
    cfg["dataset"]["splits"]["val"] = deepcopy(cfg["dataset"]["splits"]["train"])
    with pytest.raises(ValueError, match="leakage"):
        discover(cfg)


def test_replay_budget_mask_and_restore():
    buffer = ReplayBuffer(20000, 255, 1)
    for i in range(10):
        buffer.add({"image": torch.randn(3, 16, 16), "mask": torch.full((16, 16), i % 2, dtype=torch.long), "id": str(i)})
    assert 0 < buffer.bytes_used <= 20000
    other = ReplayBuffer(20000, 255, 1)
    other.load_state_dict(buffer.state_dict())
    assert [s["id"] for s in buffer.sample(5)] == [s["id"] for s in other.sample(5)]


def test_continual_metrics():
    scores = continual_scores([[.8, np.nan], [.6, .9]])
    assert scores["forgetting"] == pytest.approx(.2)
    assert scores["backward_transfer"] == pytest.approx(-.2)
    assert scores["average_seen_miou"] == pytest.approx(.75)


def test_end_to_end_two_stage(tmp_path):
    from clseg.engine import train, test as run_test
    from clseg.eda import run_eda
    make_data(tmp_path / "data", count=2)
    cfg = tiny_config()
    cfg["dataset"]["root"] = str(tmp_path / "data")
    cfg["output_dir"] = str(tmp_path / "run")
    cfg["continual"]["stages"] = [{"name": "first", "classes": [0]}, {"name": "second", "classes": [1]}]
    cfg["continual"]["replay"]["enabled"] = True
    splits = discover(validate(cfg))
    assert run_eda(cfg, splits, cfg["output_dir"])["train"]["failures"] == 0
    train(cfg, splits)
    root = Path(cfg["output_dir"])
    assert (root / "forgetting_curve.png").is_file()
    assert (root / "training_curves.png").is_file()
    assert (root / "visuals_stage_01/tsne.png").is_file()
    assert (root / "visuals_stage_01/prediction_0000.png").is_file()
    checkpoint = torch.load(root / "last.pt", weights_only=False)
    assert checkpoint["model"]["memory.active"].all()
    assert len(checkpoint["matrix"]) == 2
    train(cfg, splits, root / "last.pt") # Completed-run resume must not retrain.
    run_test(cfg, splits, root / "last.pt", split="test")
    assert (root / "evaluation_test/metrics.json").is_file()
    assert (root / "evaluation_test/masks/manifest.json").is_file()
