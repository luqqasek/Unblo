import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from experiments.h100_sprint.core import (aligned, budget_steps, d4, gate_decision,
    loss_fn, majority, undo_d4, verify_split)
from experiments.h100_sprint.data import PairPool, shapes
from experiments.h100_sprint.models import HighPassInput
from experiments.h100_sprint.prepare import disjoint_sources
from experiments.h100_sprint.run import fuse


def fixture_pool(path, side=64, count=8):
    path.mkdir(parents=True)
    clean = np.full((count, 3, side, side), 100, np.uint8)
    marked = clean.copy()
    rows = []
    for i in range(count):
        rect = [0, 0, side, side] if i % 2 == 0 else [8, 8, 24, 24]
        x0, y0, x1, y1 = rect
        marked[i, :, y0:y1, x0:x1] = 102
        rows.append({"rectangle": rect, "message": [i % 2] * 8,
                     "source_id": str(i), "source_sha256": str(i)})
    np.save(path / "clean.npy", clean)
    np.save(path / "marked.npy", marked)
    (path / "rows.json").write_text(json.dumps(rows))
    return path


def test_exact_pixels_and_crop_labels(tmp_path):
    pool = PairPool(fixture_pool(tmp_path / "pool"))
    generator = torch.Generator().manual_seed(91)
    x, y, bits = pool.sample(16, generator, augment=False)
    pixels = (x * 255).round().to(torch.uint8)
    assert set(pixels.unique().tolist()) == {100, 102}
    support = pixels[:, 0] == 102
    assert torch.equal(majority(support), y)
    assert y[:8].mean() > .5  # holes always use a full-image encoded base
    assert (y[8:12].sum((1, 2)) == 16).all()  # exact cached 16x16 crop
    assert bits.shape == (16, 8)


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("flip", [False, True])
def test_d4_inverse(rotation, flip):
    x = torch.arange(64).reshape(1, 8, 8)
    assert torch.equal(x, undo_d4(d4(x, rotation, flip), rotation, flip))


def test_majority_ties():
    support = torch.zeros(1, 4, 4)
    support[:, :2] = 1
    assert majority(support).item() == 1
    support[:, 1, 0] = 0
    assert majority(support).item() == 0


@pytest.mark.parametrize("kind", ["zero", "one", "tiny", "random"])
def test_loss_finite_symmetric_and_gradients(kind):
    torch.manual_seed(4)
    z = torch.randn(2, 16, 16, requires_grad=True)
    y = torch.zeros_like(z)
    if kind == "one":
        y.fill_(1)
    elif kind == "tiny":
        y[:, 3, 7] = 1
    elif kind == "random":
        y = (torch.rand_like(z) > .5).float()
    value = loss_fn(z, y)
    torch.testing.assert_close(value, loss_fn(-z, 1 - y))
    value.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert loss_fn((2 * y - 1) * 10, y) < value


@pytest.mark.parametrize("gain", [1., 255.])
def test_highpass_zero_init_preserves_rgb_and_learns(gain):
    torch.manual_seed(5)
    conv = nn.Conv2d(3, 8, 3, padding=1)
    wrapper = HighPassInput(conv)
    wrapper.gain = gain
    x = torch.rand(2, 3, 32, 32)
    torch.testing.assert_close(wrapper(x), conv(x), atol=1e-6, rtol=1e-5)
    wrapper(x).square().mean().backward()
    assert wrapper.conv.weight.grad[:, 3:].abs().sum() > 0


def test_real_split_rejects_leaks_and_duplicates():
    with pytest.raises(ValueError):
        verify_split(["a"], ["a"], expected=None)
    with pytest.raises(ValueError):
        verify_split(["a", "a"], ["b"], expected=None)
    assert verify_split(["a"], ["b"], expected=None) == (["a"], ["b"])


def test_synthetic_source_hash_leak_rejected():
    tr = [({"source_id": "a", "source_sha256": "same"}, Path("."))]
    va = [({"source_id": "b", "source_sha256": "same"}, Path("."))]
    with pytest.raises(ValueError, match="source_sha256"):
        disjoint_sources(tr, va)


def test_alignment_is_by_name():
    archive = {"names": np.array(["b", "a"]), "p": np.array([2, 1])}
    assert aligned(archive, ["a", "b"], "p").tolist() == [1, 2]
    with pytest.raises(ValueError):
        aligned(archive, ["c"], "p")


def test_gate_and_budget():
    assert gate_decision(.76, .8) == "pass"
    assert gate_decision(.71, .91) == "stop_transfer_gap"
    assert gate_decision(.74, .8) == "stop_inconclusive"
    assert budget_steps(20, 1., 10., 1000, 7000) == 0
    assert budget_steps(720, .1, 10., 1000, 7000) < 7000


def test_pixelseal_preserves_partial_route_exactly():
    maps = {"base_mask": np.array([np.zeros((8, 8)), np.ones((8, 8))]),
            "current_mask": np.random.default_rng(4).integers(2, size=(2, 8, 8))}
    result = fuse("pixelseal", np.full((2, 8, 8), .4), maps)
    np.testing.assert_array_equal(result[0], maps["current_mask"][0])
    assert result.dtype == np.uint8


def test_trustmark_preserves_accepted_crop_additions():
    maps = {"base_prob": np.full((1, 8, 8), .1), "corrections": np.zeros((1, 8, 8), bool)}
    maps["corrections"][0, 7, 7] = True
    result = fuse("trustmark", np.zeros((1, 8, 8)), maps)
    assert result[0, 7, 7] == 1


def test_validation_blends_raw_fold_before_saved_nrg_corrections(tmp_path):
    from experiments.h100_sprint.prepare import pack_maps
    bundle, packed = tmp_path / "bundle", tmp_path / "packed"
    (bundle / "maps").mkdir(parents=True)
    raw_folder = bundle / "cluster/region-runs/trustmark/objects-adapt/fold1"
    raw_folder.mkdir(parents=True)
    packed.mkdir()
    names = [f"image-{i}" for i in range(57)]
    raw = np.full((57, 128, 128), .1, np.float32)
    corrected = raw.copy()
    corrected[:, :8, :8] = 1
    masks = (corrected >= .5).astype(np.uint8)
    np.savez(bundle / "maps/trustmark-oof.npz", names=names, probabilities=corrected, masks=masks, targets=masks)
    np.savez(raw_folder / "maps.npz", names=names, probabilities=raw, targets=masks)
    np.savez(packed / "real_valid.npz", targets=masks)
    dino = tmp_path / "dino.npz"
    np.savez(dino, names=names, maps=np.full((57, 2, 128, 128), .1, np.float32))
    pack_maps(bundle, "trustmark", names, [dino], packed)
    with np.load(packed / "validation_maps.npz") as result:
        np.testing.assert_array_equal(result["base_prob"], raw)
        np.testing.assert_array_equal(result["corrections"], masks.astype(bool))


def test_parent_does_not_launch_pixelseal_after_failed_gate(tmp_path, monkeypatch):
    from experiments.h100_sprint import run
    calls = []
    out = tmp_path / "run"
    def worker(command, timeout, gate_path):
        family = command[command.index("--family") + 1]
        calls.append(family)
        folder = out / family
        folder.mkdir()
        (folder / "completed.json").write_text(json.dumps({"gate": "stop_inconclusive"}))
    monkeypatch.setattr(run, "launch_worker", worker)
    run.orchestrate(tmp_path / "config.json", {"output": str(out)})
    assert calls == ["trustmark"]
    assert json.loads((out / "completed.json").read_text())["status"] == "stopped_after_trustmark"


def test_parent_timeout_preserves_work_and_skips_next_family(tmp_path, monkeypatch):
    from experiments.h100_sprint import run
    def worker(command, timeout, gate_path):
        raise run.subprocess.TimeoutExpired(command, timeout)
    monkeypatch.setattr(run, "launch_worker", worker)
    out = tmp_path / "run"
    run.orchestrate(tmp_path / "config.json", {"output": str(out)})
    done = json.loads((out / "completed.json").read_text())
    assert done["reason"] == "TimeoutExpired" and not (out / "pixelseal").exists()


def test_submission_rejects_incomplete_or_duplicate_names():
    from experiments.h100_sprint.infer import validate_submission
    with pytest.raises(ValueError, match="1020"):
        validate_submission(["x"], torch.zeros(1, 128, 128, dtype=torch.uint8), ["x"])


def test_highpass_state_dict_roundtrip():
    first = HighPassInput(nn.Conv2d(3, 4, 3, padding=1))
    second = HighPassInput(nn.Conv2d(3, 4, 3, padding=1))
    second.load_state_dict(first.state_dict(), strict=True)
    x = torch.rand(1, 3, 16, 16)
    torch.testing.assert_close(first(x), second(x), atol=0, rtol=0)


def test_watchdog_stops_hung_worker_at_minute7(tmp_path, monkeypatch):
    from experiments.h100_sprint import run
    ticks = iter([0., 0., 421.])
    monkeypatch.setattr(run.time, "monotonic", lambda: next(ticks))
    class Process:
        stopped = False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self, timeout=None):
            if self.stopped: return -15
            raise run.subprocess.TimeoutExpired(["fixture"], timeout)
        def terminate(self): self.stopped = True
    process = Process()
    monkeypatch.setattr(run.subprocess, "Popen", lambda command: process)
    with pytest.raises(run.GateTimeout):
        run.launch_worker(["fixture"], 780, tmp_path / "minute7-gate.json")
    assert process.stopped
