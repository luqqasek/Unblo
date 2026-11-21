import numpy as np
import pytest
import torch

from experiments.native_localizer.generator import provenance_composite, pair_views, rectangle, rect_mask
from experiments.native_localizer.losses import segmentation, local_balanced, pair_loss
from experiments.native_localizer.model import ChannelNorm, LocalStem, NativeLocalizer
from wmloc.utils import downsample_mask, miou_image


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_support_is_provenance_even_with_zero_residual_and_majority_ties():
    clean = np.zeros((512, 512, 3), np.uint8)
    support = np.zeros((512, 512), bool); support[:2, :4] = True
    out = provenance_composite(clean, clean.copy(), support)
    assert not out.any()
    target = downsample_mask(support)
    assert target[0, 0] == 1 and target.sum() == 1
    assert miou_image(target, target) == 1


def test_rectangle_aspect_constraints_after_integer_rounding():
    rng = np.random.default_rng(42)
    for maximum in (2, 16):
        for _ in range(300):
            x0, y0, x1, y1 = rectangle(rng, float(rng.uniform(.0025, .5)), maximum)
            assert 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512
            assert max((x1-x0)/(y1-y0), (y1-y0)/(x1-x0)) <= maximum


def test_move_and_complement_pairs_change_observed_signal_and_keep_support_valid():
    clean = np.zeros((512, 512, 3), np.uint8); marked = clean+1
    row = dict(rectangle=[0, 0, 512, 512], primary_kind="partial", regime="large_source_remnant",
               minority_fraction=.02, clean_minority=False)
    for kind in ("move", "clean_twin", "complement"):
        xs, ys, safe = pair_views(clean, marked, row, np.random.default_rng(1), kind, simple=True)
        for image, mask in zip(xs, ys):
            assert np.array_equal(downsample_mask(image[..., 0]), mask)
        assert safe.shape == (128, 128)
        assert np.any(xs[0] != xs[1])
        if kind == "move":
            assert np.sum(xs[0]) == np.sum(xs[1])
    row["rectangle"] = [100, 100, 200, 200]; row["regime"] = "local_crop"
    xs, ys, _ = pair_views(clean, marked, row, np.random.default_rng(1), "complement")
    assert not ys[1].any()  # complement ONLY inside actual marked crop
    with pytest.raises(ValueError):
        pair_views(clean, marked, row, np.random.default_rng(1), "move")


def test_stable_empty_full_losses_and_pair_direction():
    for value in (0., 1.):
        z = torch.zeros(2, 128, 128, requires_grad=True)
        y = torch.full_like(z, value)
        loss = segmentation(z, y)+.1*local_balanced(z, y)+pair_loss(z, y, [(0, 1)])
        assert torch.isfinite(loss)
        loss.backward(); assert torch.isfinite(z.grad).all()
    y = torch.tensor([[[1.]], [[0.]]])
    good = torch.tensor([[[2.]], [[-2.]]])
    assert pair_loss(good, y, [(0, 1)]) < pair_loss(-good, y, [(0, 1)])


def test_channel_norm_and_local_stem_have_no_global_spatial_dependency():
    norm = ChannelNorm(32)
    x = torch.randn(1, 32, 9, 9); y = x.clone(); y[:, :, -1, -1] += 100
    assert torch.equal(norm(x)[:, :, 0, 0], norm(y)[:, :, 0, 0])
    local = LocalStem().eval()
    x = torch.rand(1, 3, 64, 64); y = x.clone(); y[:, :, -1, -1] = 100
    with torch.no_grad():
        a, _ = local(x); b, _ = local(y)
    assert torch.equal(a[:, :, 0, 0], b[:, :, 0, 0])
    assert local.receptive_field == 27


def test_native_base_forward_and_gradients_through_every_stage():
    torch.manual_seed(42)
    model = NativeLocalizer(checkpointing=True)
    output = model(torch.rand(1, 3, 512, 512), return_features=True)
    assert output["logits"].shape == (1, 128, 128)
    assert [s[1:] for s in output["shapes"]] == [(128,128,128), (256,64,64), (512,32,32), (1024,16,16)]
    assert output["first"].dtype == output["local"].dtype == torch.float32
    segmentation(output["logits"], torch.randint(0,2,(1,128,128))).backward()
    for stage in model.encoder.stages:
        grad = next(stage.parameters()).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    for branch in (model.encoder.stem, model.local.first, model.fuse):
        grad = next(branch.parameters()).grad
        assert grad is not None and grad.abs().sum() > 0
    assert not any(name.startswith("encoder.head.") for name, _ in model.named_parameters())
