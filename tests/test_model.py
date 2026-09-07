"""Architecture invariants and regressions for the prompt-free pipeline."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from training.models import build_model
from training.models.cmspa_net import get_r50_b16_config, get_testing, VisionTransformer
from training.models.modules import CMSPA_Fusion, SSPA_SpatialAttention, SSPA_ZPool, channel_std, strip_rms
from training.models.backbones.resnet_v2 import ResNetV2


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("ablation", ["M0", "M1", "M2", "M3"])
def test_all_ablations_logits_and_gradients(ablation):
    torch.manual_seed(23)
    model = VisionTransformer(get_testing(), ablation=ablation)
    inputs = [torch.randn(1, 1, 32, 48) for _ in range(3)]
    logits = model(*inputs)
    assert logits.shape == (1, 4, 32, 48)
    loss = nn.functional.cross_entropy(logits, torch.randint(4, (1, 32, 48)))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Disconnected parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), name
    roots = [branch.embeddings.hybrid_model.root.conv.weight for branch in
             (model.transformer1, model.transformer2, model.transformer3)]
    assert len({weight.data_ptr() for weight in roots}) == 3
    assert all(weight.grad.abs().sum() > 0 for weight in roots)
    assert isinstance(model.sspanet_cine, nn.Identity) == (ablation == "M0")
    if ablation == "M3":
        assert model.cross_fusion.conv_patho[0].weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        evaluated = model(*inputs)
    assert isinstance(evaluated, torch.Tensor)
    assert evaluated.shape == logits.shape
    assert not torch.allclose(evaluated.sum(1), torch.ones_like(evaluated[:, 0]))


def test_default_pipeline_shapes_without_allocating_training_memory():
    config = get_r50_b16_config()
    assert tuple(config.resnet.num_layers) == (3, 4, 9)
    shapes = {}
    with torch.device("meta"):
        model = VisionTransformer(config)
        model.transformer1.register_forward_hook(
            lambda module, inputs, output: shapes.update(
                bottleneck=tuple(output[0].shape), skips=[tuple(x.shape) for x in output[1]]))
        model.cross_fusion.register_forward_hook(
            lambda module, inputs, output: shapes.update(fused=tuple(output.shape)))
        x = torch.empty(16, 1, 128, 128)
        assert model(x, x, x).shape == (16, 4, 128, 128)
    assert shapes == {
        "bottleneck": (16, 1024, 8, 8),
        "skips": [(16, 512, 16, 16), (16, 256, 32, 32), (16, 64, 64, 64)],
        "fused": (16, 512, 8, 8),
    }
    assert not any("text" in name or "prompt" in name for name, _ in model.named_parameters())
    assert sum(isinstance(module, type(model.decoder)) for module in model.modules()) == 1
    assert sum(p.numel() for p in model.parameters()) == 64_403_442


@pytest.mark.parametrize("name,ablation", [("CMSPA-Net", "M3"), ("concat_baseline", "M0"),
                                          ("sspanet_baseline", "M1"), ("cross_attn_baseline", "M2")])
def test_registry_selects_correct_ablation(name, ablation):
    model = build_model(name, config=get_testing())
    assert model.ablation == ablation
    if name.endswith("baseline"):
        with pytest.raises(ValueError, match="requires ablation"):
            build_model(name, config=get_testing(), ablation="M3")


def test_legacy_baselines_are_importable_without_module_namespace_conflict():
    from training.models.baselines import ResUNetPlusPlus2D, UNet2D, UNet3D
    assert all(issubclass(model, nn.Module) for model in (ResUNetPlusPlus2D, UNet2D, UNet3D))


def test_configuration_is_not_mutated_and_no_implicit_load(tmp_path):
    config = get_testing()
    config.pretrained_path = str(tmp_path / "does-not-exist.npz")
    before = deepcopy(config.to_dict())
    model = VisionTransformer(config, ablation="M0", num_classes=5)
    assert config.to_dict() == before
    assert model.config.n_classes == 5
    model.config.skip_channels[0] = 999
    assert config.to_dict() == before


@pytest.mark.parametrize("n_skip", [0, 1, 2, 3])
def test_skip_selection_does_not_leave_disconnected_parameters(n_skip):
    config = get_testing()
    config.n_skip = n_skip
    model = VisionTransformer(config, ablation="M0")
    x = torch.randn(1, 1, 32, 32)
    model(x, x, x).square().mean().backward()
    assert len(model.feature_fusion) == n_skip
    assert all(p.grad is not None for p in model.parameters())
    assert config.skip_channels == [256, 128, 32, 0]


def test_rms_and_variance_are_distinct_and_amp_safe():
    x = torch.full((2, 4, 3, 5), 300.0, dtype=torch.float16)
    rms = strip_rms(x, dim=3)
    std = channel_std(x)
    assert rms.dtype == torch.float32
    torch.testing.assert_close(rms, torch.full((2, 4, 3, 1), 300.0))
    torch.testing.assert_close(std, torch.full((2, 1, 3, 5), 0.001))
    assert torch.isfinite(channel_std(x[:, :1])).all()
    pooled = SSPA_ZPool()(torch.arange(24.).reshape(1, 3, 2, 4))
    assert pooled.shape == (1, 2, 2, 4)
    torch.testing.assert_close(pooled[:, 0], torch.arange(16., 24.).reshape(1, 2, 4))


def test_factored_sspa_projection_matches_author_forward_and_gradients():
    torch.manual_seed(7)
    factored = SSPA_SpatialAttention(4).double()
    reference = deepcopy(factored)
    x = torch.randn(2, 4, 3, 5, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual = factored(x)
    strip_h = reference.bn1(reference.conv1((y.square().mean(3, keepdim=True) + 1e-6).sqrt()))
    strip_w = reference.bn2(reference.conv2((y.square().mean(2, keepdim=True) + 1e-6).sqrt()))
    expected = y * torch.sigmoid(reference.conv3(strip_h + strip_w))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-9, atol=1e-10)
    for (name, parameter), (_, other) in zip(factored.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-9, atol=1e-10, msg=name)


def test_cmspa_learns_from_both_modalities_and_uses_sum_of_stds():
    torch.manual_seed(42)
    fusion = CMSPA_Fusion(8, 4)
    inputs = [torch.randn(2, 8, 3, 5, requires_grad=True) for _ in range(3)]
    captured = []
    fusion.conv_patho.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    output = fusion(*inputs)
    expected = (inputs[1].var(1, keepdim=True, unbiased=False) + 1e-6).sqrt()
    expected += (inputs[2].var(1, keepdim=True, unbiased=False) + 1e-6).sqrt()
    torch.testing.assert_close(captured[0], expected)
    assert fusion.conv_patho[0].in_channels == 1
    output.square().mean().backward()
    for image in inputs:
        assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    assert fusion.conv_patho[0].weight.grad.abs().sum() > 0


def test_encoder_checkpointing_has_same_outputs_and_gradients():
    torch.manual_seed(11)
    ordinary = ResNetV2((1, 1, 1), 0.5)
    checkpointed = deepcopy(ordinary)
    checkpointed.gradient_checkpointing = True
    x = torch.randn(1, 3, 32, 48, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    outputs_a = ordinary(x)
    outputs_b = checkpointed(y)
    torch.testing.assert_close(outputs_a[0], outputs_b[0])
    sum(t.square().mean() for t in [outputs_a[0], *outputs_a[1]]).backward()
    sum(t.square().mean() for t in [outputs_b[0], *outputs_b[1]]).backward()
    torch.testing.assert_close(x.grad, y.grad)
    for a, b in zip(ordinary.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


def test_explicit_npz_load_initializes_independent_encoders(tmp_path):
    model = VisionTransformer(get_testing(), ablation="M0")
    encoder = model.transformer1.embeddings.hybrid_model
    weights = {}
    for key, parameter, is_conv in encoder._pretrained_parameters():
        array = np.full(tuple(parameter.shape), 0.123, dtype=np.float32)
        weights[key] = array.transpose(2, 3, 1, 0) if is_conv else array
    path = tmp_path / "encoder.npz"
    np.savez(path, **weights)
    model.load_pretrained_encoders(path)
    for branch in (model.transformer1, model.transformer2, model.transformer3):
        for parameter in branch.embeddings.hybrid_model.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.123))
    del weights["gn_root/scale"]
    before = encoder.root.conv.weight.detach().clone()
    with pytest.raises(ValueError, match="missing gn_root/scale"):
        model.load_from(weights)
    torch.testing.assert_close(encoder.root.conv.weight, before)


def test_invalid_inputs_fail_before_encoder_execution():
    model = VisionTransformer(get_testing(), ablation="M0")
    x = torch.randn(1, 1, 32, 32)
    with pytest.raises(ValueError, match="aligned"):
        model(x, x[..., :16], x)
    with pytest.raises(ValueError, match="divisible by 16"):
        model(x[..., :31], x[..., :31], x[..., :31])
    with pytest.raises(TypeError, match="floating"):
        model(x.long(), x.long(), x.long())
    with pytest.raises(ValueError, match="shape"):
        model(x.expand(1, 2, 32, 32), x, x)
    model.eval()
    with torch.no_grad():
        assert model(x.repeat(1, 3, 1, 1), x, x).shape == (1, 4, 32, 32)
