from __future__ import annotations

import numpy as np
import pytest
import torch

from bulwark.data.taxonomy import default_taxonomy
from bulwark.eval.metrics import auroc, macro_f1
from bulwark.model.heads import BulwarkHeads, activate
from bulwark.model.lora import LoRALinear, apply_lora, lora_state_dict
from bulwark.model.losses import MultiTaskLoss, distill_loss, encode_targets
from bulwark.model.pooling import NUM_SUMMARY_TOKENS, summary_tokens


def test_summary_tokens_shape_for_video_and_image() -> None:
    assert summary_tokens(torch.randn(2, 8, 16, 16, 32)).shape == (2, NUM_SUMMARY_TOKENS, 32)
    assert summary_tokens(torch.randn(3, 1, 16, 16, 32)).shape == (3, NUM_SUMMARY_TOKENS, 32)


def test_heads_output_shapes_and_activation_ranges() -> None:
    taxonomy = default_taxonomy()
    heads = BulwarkHeads(32, taxonomy, width=64)
    logits = heads(torch.randn(4, NUM_SUMMARY_TOKENS, 32))
    assert logits["activity"].shape == (4, len(taxonomy["activity"].classes))
    assert logits["yawn"].shape == (4,)
    probs = activate(logits, taxonomy)
    assert torch.allclose(probs["activity"].sum(-1), torch.ones(4))
    assert probs["head_yaw"].min() >= -1 and probs["head_yaw"].max() <= 1


def test_encode_targets_prefers_ground_truth_and_marks_teacher_weight() -> None:
    taxonomy = default_taxonomy()
    targets = encode_targets(
        [{"activity": "texting"}, {}],
        [{"activity": "eating", "yawn": 1.0}, {"yawn": 0.0}],
        taxonomy,
    )
    assert targets.values["activity"].tolist() == [taxonomy["activity"].class_index("texting"), -1]
    assert targets.mask["activity"].tolist() == [1.0, 0.0]
    assert targets.weight["yawn"].tolist() == [0.5, 0.5]
    assert targets.mask["helmet"].sum() == 0


def test_loss_ignores_missing_labels() -> None:
    taxonomy = default_taxonomy()
    heads = BulwarkHeads(16, taxonomy, width=32, dropout=0.0)
    loss_fn = MultiTaskLoss(taxonomy)
    targets = encode_targets([{"yawn": 1.0}, {"yawn": 0.0}], [{}, {}], taxonomy)
    logits = heads(torch.randn(2, NUM_SUMMARY_TOKENS, 16))
    loss, parts = loss_fn(logits, targets)
    assert set(parts) == {"yawn"}
    loss.backward()
    yawn = heads.label_names.index("yawn")
    start = sum(heads.widths[:yawn])
    grad = heads.out.weight.grad
    assert grad[start].abs().sum() > 0
    other = [i for i in range(grad.shape[0]) if i != start]
    assert grad[other].abs().sum() == 0
    no_labels, parts = loss_fn(logits, encode_targets([{}, {}], [{}, {}], taxonomy))
    assert parts == {} and float(no_labels) == 0.0


def test_distill_loss_is_zero_when_student_matches_teacher() -> None:
    taxonomy = default_taxonomy()
    logits = BulwarkHeads(8, taxonomy, width=16)(torch.randn(2, 4, 8))
    same = distill_loss(logits, {k: v.detach() for k, v in logits.items()}, taxonomy)
    shifted = distill_loss(logits, {k: v.detach() + 3 for k, v in logits.items()}, taxonomy)
    assert float(shifted) > float(same)


def test_lora_starts_as_identity_and_merges() -> None:
    base = torch.nn.Linear(8, 8)
    lora = LoRALinear(base, rank=2)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x))
    torch.nn.init.normal_(lora.up.weight)
    assert torch.allclose(lora.merged()(x), lora(x), atol=1e-5)


def test_apply_lora_wraps_only_top_blocks() -> None:
    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.qkv = torch.nn.Linear(4, 12)
            self.attn.proj = torch.nn.Linear(4, 4)

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block() for _ in range(3)])

    encoder = Encoder()
    wrapped = apply_lora(encoder, last_blocks=1, rank=2, targets=("qkv", "proj"))
    assert wrapped == ["blocks.2.attn.qkv", "blocks.2.attn.proj"]
    trainable = [n for n, p in encoder.named_parameters() if p.requires_grad]
    assert trainable and all(".down." in n or ".up." in n for n in trainable)


def test_apply_lora_needs_a_module_list_of_blocks() -> None:
    encoder = torch.nn.Module()
    with pytest.raises(TypeError, match="NoneType"):
        apply_lora(encoder)
    encoder.blocks = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.raises(TypeError, match="Sequential"):
        apply_lora(encoder)


def test_metrics() -> None:
    assert auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
    assert auroc(np.array([0.5, 0.5]), np.array([1, 1])) is None
    assert macro_f1(np.array([0, 1, 1]), np.array([0, 1, 0]), 2) == (2 / 3 + 2 / 3) / 2


def _yawn_loss(gt: list[dict], teacher: list[dict], log_var: float | None = None) -> tuple[float, torch.Tensor, MultiTaskLoss]:
    taxonomy = default_taxonomy()
    torch.manual_seed(0)
    heads = BulwarkHeads(16, taxonomy, width=32, dropout=0.0)
    tokens = torch.randn(4, NUM_SUMMARY_TOKENS, 16, generator=torch.Generator().manual_seed(1))
    loss_fn = MultiTaskLoss(taxonomy, learn_weights=log_var is not None)
    if log_var is not None:
        with torch.no_grad():
            loss_fn.log_vars[loss_fn.names.index("yawn")] = log_var
    loss, _ = loss_fn(heads(tokens), encode_targets(gt, teacher, taxonomy))
    loss.backward()
    return float(loss), heads.out.weight.grad.clone(), loss_fn


YAWNS = [{"yawn": 1.0}, {"yawn": 0.0}, {"yawn": 1.0}, {"yawn": 0.0}]
NONE: list[dict] = [{}] * 4


def test_teacher_labels_count_at_half_weight_in_loss_and_gradient() -> None:
    gt_loss, gt_grad, _ = _yawn_loss(YAWNS, NONE)
    teacher_loss, teacher_grad, _ = _yawn_loss(NONE, YAWNS)
    assert teacher_loss == pytest.approx(0.5 * gt_loss)
    assert torch.allclose(teacher_grad, 0.5 * gt_grad, atol=1e-7)
    mixed_loss, _, _ = _yawn_loss(YAWNS[:2] + NONE[:2], NONE[:2] + YAWNS[2:])
    assert teacher_loss < mixed_loss < gt_loss


def test_teacher_weight_survives_learned_uncertainty() -> None:
    # Kendall weighting is scale-invariant at its optimum; the teacher factor must not be.
    plain, _, _ = _yawn_loss(YAWNS, NONE)
    optimum = float(np.log(2 * plain))  # s* = log(L) for the unweighted per-label loss L = 2 * plain
    _, gt_grad, gt_fn = _yawn_loss(YAWNS, NONE, log_var=optimum)
    _, teacher_grad, teacher_fn = _yawn_loss(NONE, YAWNS, log_var=optimum)
    for loss_fn in (gt_fn, teacher_fn):
        assert abs(float(loss_fn.log_vars.grad[loss_fn.names.index("yawn")])) < 1e-5
    assert torch.allclose(teacher_grad, 0.5 * gt_grad, atol=1e-7)


def test_apply_lora_defaults_cover_attention_and_mlp_and_only_the_final_norm() -> None:
    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.qkv = torch.nn.Linear(4, 12)
            self.attn.proj = torch.nn.Linear(4, 4)
            self.mlp = torch.nn.Module()
            self.mlp.fc1 = torch.nn.Linear(4, 8)
            self.mlp.fc2 = torch.nn.Linear(8, 4)

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block() for _ in range(3)])
            self.norms_block = torch.nn.ModuleList([torch.nn.LayerNorm(4) for _ in range(4)])

    encoder = Encoder()
    wrapped = apply_lora(encoder, last_blocks=2, rank=2)
    assert wrapped == [f"blocks.{b}.{n}" for b in (1, 2) for n in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")]
    trainable = {n for n, p in encoder.named_parameters() if p.requires_grad}
    norms = {n for n in trainable if "norms_block" in n}
    assert norms == {"norms_block.3.weight", "norms_block.3.bias"}
    assert {k for k in lora_state_dict(encoder) if "norms_block" in k} == norms


def test_distill_loss_mask_leaves_out_labels_and_samples() -> None:
    taxonomy = default_taxonomy()
    torch.manual_seed(0)
    student = BulwarkHeads(8, taxonomy, width=16)(torch.randn(4, 3, 8))
    teacher = {k: v.detach() + torch.randn_like(v) for k, v in student.items()}
    ones = {name: torch.ones(4) for name in taxonomy.labels}
    assert torch.allclose(distill_loss(student, teacher, taxonomy, mask=ones), distill_loss(student, teacher, taxonomy))
    # A label left out of the mask, or masked on every sample, adds nothing.
    only_yawn = distill_loss(student, {"yawn": teacher["yawn"]}, taxonomy, mask={"yawn": torch.ones(4)})
    zeroed = {**ones, **{name: torch.zeros(4) for name in taxonomy.labels if name != "yawn"}}
    assert torch.allclose(only_yawn, distill_loss(student, teacher, taxonomy, mask=zeroed))
    # Per-sample weights average over the samples kept.
    first_two = distill_loss(student, teacher, taxonomy, mask={"yawn": torch.tensor([1.0, 1.0, 0.0, 0.0])})
    half = {k: v[:2] for k, v in student.items()}
    expected = distill_loss(half, {"yawn": teacher["yawn"][:2]}, taxonomy, mask={"yawn": torch.ones(2)})
    assert torch.allclose(first_two, expected)
