"""The V-JEPA 2.1 encoder code vendored in bulwark/model/vjepa21 is Meta's code at the pinned
commit: the same definitions, the same parameters and, with the official weights, the same
outputs as the encoder built from a clone of facebookresearch/vjepa2 (third_party/vjepa2,
scripts/fetch_pretrained.sh). The comparisons are skipped without the clone, and the output
comparison also without the weights."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from bulwark.model import backbone
from bulwark.model import vjepa21 as vendored_package
from bulwark.model.backbone import VJEPA_COMMIT, VJepaEncoder

ROOT = Path(__file__).resolve().parents[1]
CLONE = ROOT / "third_party" / "vjepa2"
VENDORED = Path(vendored_package.__file__).parent
# Vendored module -> its file in the pinned clone, and the definitions kept from it.
FILES = {
    "vision_transformer": ("app/vjepa_2_1/models/vision_transformer.py", ("VisionTransformer", "vit_base")),
    "modules": (
        "app/vjepa_2_1/models/utils/modules.py",
        ("rotate_queries_or_keys", "DropPath", "MLP", "SwiGLUFFN", "RoPEAttention", "Attention", "Block"),
    ),
    "patch_embed": ("app/vjepa_2_1/models/utils/patch_embed.py", ("PatchEmbed", "PatchEmbed3D")),
    "tensors": ("src/utils/tensors.py", ("_no_grad_trunc_normal_", "trunc_normal_")),
    "masks": ("src/masks/utils.py", ("apply_masks",)),
}
needs_clone = pytest.mark.skipif(not CLONE.is_dir(), reason="needs third_party/vjepa2 (scripts/fetch_pretrained.sh)")


def upstream(module: str) -> ModuleType:
    """A module of the pinned clone, imported as Meta's own code imports it."""
    if str(CLONE) not in sys.path:
        sys.path.insert(0, str(CLONE))
    return importlib.import_module(module)


def upstream_vit_base() -> Any:
    return upstream("app.vjepa_2_1.models.vision_transformer").vit_base


def build(vit_base: Any) -> torch.nn.Module:
    """The encoder as VJepaEncoder builds it, from either copy of the code."""
    config = backbone.BackboneConfig()
    torch.manual_seed(0)
    return vit_base(
        patch_size=config.patch_size, img_size=(config.img_size, config.img_size), num_frames=config.num_frames,
        tubelet_size=config.tubelet_size, use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False,
        use_rope=True, img_temporal_dim_size=1, interpolate_rope=True, use_activation_checkpointing=False,
    )  # fmt: skip


def test_every_vendored_file_names_its_source_commit_and_meta_licence() -> None:
    licence = (VENDORED / "LICENSE").read_text(encoding="utf-8")
    assert licence.startswith("MIT License\n\nCopyright (c) Meta Platforms, Inc. and affiliates.")
    assert backbone.VJEPA_LICENSE == VENDORED / "LICENSE"
    assert sorted(p.stem for p in VENDORED.glob("*.py")) == sorted(["__init__", *FILES])
    for name, (path, _) in FILES.items():
        text = (VENDORED / f"{name}.py").read_text(encoding="utf-8")
        assert text.startswith("# Copyright (c) "), name
        assert f"facebookresearch/vjepa2 at commit\n# {VJEPA_COMMIT},\n# file {path}." in text, name
        assert "MIT License in LICENSE next to this file" in text and "# Changes: " in text, name
    assert VJEPA_COMMIT in (VENDORED / "__init__.py").read_text(encoding="utf-8")


def test_the_encoder_builds_from_the_package_alone(tmp_path: Path) -> None:
    """In a fresh interpreter outside the checkout, with no clone on the path."""
    code = (
        "import sys\n"
        "from bulwark.model.backbone import VJepaEncoder\n"
        "encoder = VJepaEncoder(pretrained=False)\n"
        "assert type(encoder.encoder).__module__ == 'bulwark.model.vjepa21.vision_transformer'\n"
        "assert not {'app', 'src'} & set(sys.modules), 'imported the clone'\n"
        "print(sum(p.numel() for p in encoder.encoder.parameters()))\n"
    )
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["86833152"]


@needs_clone
def test_the_clone_holds_the_pinned_files_unmodified() -> None:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=CLONE, capture_output=True, text=True)

    assert git("rev-parse", "HEAD").stdout.strip() == VJEPA_COMMIT
    paths = ["LICENSE", *(path for path, _ in FILES.values())]
    assert git("diff", "--quiet", "HEAD", "--", *paths).returncode == 0, "the clone's copy differs from the commit"
    assert (CLONE / "LICENSE").read_bytes() == (VENDORED / "LICENSE").read_bytes()


@needs_clone
@pytest.mark.parametrize("name", sorted(FILES))
def test_vendored_definitions_are_the_pinned_code_unchanged(name: str) -> None:
    path, kept = FILES[name]
    ours = importlib.import_module(f"bulwark.model.vjepa21.{name}")
    theirs = upstream(path.removesuffix(".py").replace("/", "."))
    for definition in kept:
        assert inspect.getsource(getattr(ours, definition)) == inspect.getsource(getattr(theirs, definition)), definition
    defined = {n for n, v in vars(ours).items() if getattr(v, "__module__", None) == ours.__name__}
    assert defined == set(kept)


@needs_clone
def test_vendored_and_cloned_encoders_have_the_same_parameters() -> None:
    from bulwark.model.vjepa21.vision_transformer import vit_base

    ours, theirs = build(vit_base).state_dict(), build(upstream_vit_base()).state_dict()
    assert list(ours) == list(theirs)
    assert {k: tuple(v.shape) for k, v in ours.items()} == {k: tuple(v.shape) for k, v in theirs.items()}
    for key in ours:  # same seed, same initialisation code
        assert torch.equal(ours[key], theirs[key]), key


@pytest.mark.vjepa
@needs_clone
@pytest.mark.skipif(not Path(backbone.DEFAULT_CHECKPOINT).is_file(), reason="needs the pretrained V-JEPA weights")
def test_vendored_and_cloned_encoders_give_identical_outputs_with_the_official_weights() -> None:
    """fp32 on CPU, a 16-frame clip (video path) and a 1-frame clip (image path)."""
    ours = VJepaEncoder(pretrained=True).eval()
    theirs = build(upstream_vit_base()).eval()
    theirs.load_state_dict(ours.encoder.state_dict(), strict=True)
    torch.manual_seed(0)
    for frames in (16, 1):
        clip = torch.randint(0, 256, (1, frames, 256, 256, 3), dtype=torch.uint8)
        x = (clip.float().div(255).permute(0, 4, 1, 2, 3) - ours.mean) / ours.std
        with torch.no_grad():
            expected = theirs(x)
            assert torch.equal(ours.encoder(x), expected), frames
            assert torch.equal(ours(clip).flatten(1, 3), expected), frames
