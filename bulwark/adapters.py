"""Task adapters: small classifiers trained on Bulwark embeddings for a new label set.

Bulwark's own heads cover the shared taxonomy. A product that needs a label the taxonomy
lacks (a site-specific unsafe action, a new object, a customer's own alert classes)
trains an adapter on embeddings of a few labelled clips instead of fine-tuning the
encoder:

    from bulwark.adapters import Adapter, fit_adapter

    adapter = fit_adapter(embeddings, labels)          # (N, 25, 768) B1 tokens, N labels
    adapter.save("adapters/unsafe_reach.pt")
    probs = Adapter.load("adapters/unsafe_reach.pt").predict_proba(new_embeddings)

Embeddings come from `bulwark.inference` (`model.embed`) or from a features directory
written by `bulwark train cache` (`load_cached_embeddings`). An adapter only fits the
embeddings it was trained on: the same model, checkpoint and embedding kind. Pass
`model=` (`model.embedding_source(kind)` or `features_source(features_dir)`) when fitting
so that `Adapter.check`, `predict_video` and `predict_image` can refuse other embeddings.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from bulwark.data.manifest import ClipRecord, read_manifest
from bulwark.data.taxonomy import default_taxonomy

ADAPTER_FORMAT = "bulwark-adapter"
ADAPTER_VERSION = 1
POOLINGS = ("mean", "first", "flatten")


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    classes: tuple[str, ...]
    in_dim: int                 # feature width after pooling
    hidden: int = 0             # 0 = linear (multinomial logistic regression); >0 = one hidden layer
    dropout: float = 0.0
    pooling: str = "mean"       # how (N, P, D) token embeddings become (N, D) vectors
    embedding: str = "tokens"   # which Bulwark embedding it was trained on (tokens, pooled, state)
    source: str = ""            # free text: the model or features dir that produced the embeddings
    # What produced the embeddings, {"family": "b1" | "e1", "weights": "frozen" | checkpoint
    # sha256}, as `embedding_source` / `features_source` give it; empty when unknown.
    model: Mapping[str, Any] = field(default_factory=dict)


class Adapter(nn.Module):
    """Standardise -> Linear (or a one-hidden-layer MLP) -> class logits."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        if len(config.classes) < 2:
            raise ValueError("An adapter needs at least two classes")
        if config.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {config.pooling!r}")
        self.config = config
        self.register_buffer("mean", torch.zeros(config.in_dim))
        self.register_buffer("std", torch.ones(config.in_dim))
        out = nn.Linear(config.hidden or config.in_dim, len(config.classes))
        nn.init.zeros_(out.weight)  # start from uniform class probabilities
        nn.init.zeros_(out.bias)
        self.net: nn.Module = out
        if config.hidden:
            self.net = nn.Sequential(nn.Linear(config.in_dim, config.hidden), nn.GELU(), nn.Dropout(config.dropout), out)

    def vectors(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Embeddings pooled as configured -> float32 (N, in_dim)."""
        x = pool(embeddings, self.config.pooling)
        if x.shape[1] != self.config.in_dim:
            raise ValueError(f"Expected {self.config.in_dim} features after pooling, got {x.shape[1]}")
        return x

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.net((vectors - self.mean) / self.std)

    @torch.no_grad()
    def predict_proba(self, embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
        """(N, C) class probabilities, columns in `config.classes` order."""
        self.eval()
        x = self.vectors(embeddings).to(self.mean.device)
        return self(x).softmax(-1).cpu().numpy()

    def predict(self, embeddings: np.ndarray | torch.Tensor) -> list[str]:
        return [self.config.classes[i] for i in self.predict_proba(embeddings).argmax(-1)]

    def check(self, source: Mapping[str, Any]) -> None:
        """Raise ValueError unless `source` matches the embeddings this adapter was trained on.

        `source` is `model.embedding_source(adapter.config.embedding)`. A field either side
        does not know (absent or None) is not compared, so adapters fitted without `model=`
        pass; the feature width is still checked at prediction time.
        """
        for key in ("family", "weights"):
            mine, theirs = self.config.model.get(key), source.get(key)
            if mine is not None and theirs is not None and mine != theirs:
                raise ValueError(
                    f"it was trained on embeddings with {key}={mine!r}, but this model gives {key}={theirs!r}. "
                    "Retrain the adapter on this model's embeddings."
                )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = asdict(self.config)
        config["classes"] = list(self.config.classes)
        config["model"] = dict(self.config.model)
        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        torch.save({"format": ADAPTER_FORMAT, "format_version": ADAPTER_VERSION, "config": config, "state": state}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Adapter:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != ADAPTER_FORMAT or payload.get("format_version") != ADAPTER_VERSION:
            raise ValueError(f"{path} is not a {ADAPTER_FORMAT} v{ADAPTER_VERSION} file")
        config = dict(payload["config"])
        config["classes"] = tuple(config["classes"])
        adapter = cls(AdapterConfig(**config))
        adapter.load_state_dict(payload["state"])
        return adapter.eval()


def pool(embeddings: np.ndarray | torch.Tensor, pooling: str = "mean") -> torch.Tensor:
    """(N, D) vectors pass through; (N, P, D) tokens are pooled to (N, D) or flattened.

    `mean` averages the tokens (for B1 summary tokens this equals the global token),
    `first` keeps token 0, `flatten` keeps every token (P * D features).
    """
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    x = (embeddings if isinstance(embeddings, torch.Tensor) else torch.from_numpy(np.asarray(embeddings))).float()
    if x.ndim == 3:
        x = x.mean(1) if pooling == "mean" else x[:, 0] if pooling == "first" else x.flatten(1)
    if x.ndim != 2:
        raise ValueError(f"Expected (N, D) vectors or (N, P, D) tokens, got shape {tuple(x.shape)}")
    return x


def fit_adapter(
    embeddings: np.ndarray | torch.Tensor,
    labels: Sequence[Any],
    *,
    classes: Sequence[str] | None = None,
    hidden: int = 0,
    dropout: float = 0.1,
    pooling: str = "mean",
    embedding: str = "tokens",
    source: str = "",
    model: Mapping[str, Any] | None = None,
    l2: float = 1e-3,
    steps: int = 300,
    lr: float = 1e-3,
    batch_size: int = 256,
    balanced: bool = True,
    seed: int = 0,
    device: str = "cpu",
) -> Adapter:
    """Train an adapter on (N, D) vectors or (N, P, D) tokens with one label per row.

    Labels are compared as strings; `classes` fixes their order (default: sorted). Inputs
    are standardised with training statistics, and `balanced` weights the loss so each
    class counts equally, which matters when positives are rare. `l2` penalises the
    weights (not the biases). `embedding`, `source` and `model` describe where the
    embeddings came from and are stored in the config; `model` is what `Adapter.check`
    compares against a model's `embedding_source`.

    A linear adapter is multinomial logistic regression solved with full-batch L-BFGS for
    up to `steps` iterations: the problem is convex, so the result depends on the data
    alone. (Adam moves every weight at a similar rate, which on a few examples gives
    noise features as much weight as informative ones.) An MLP adapter trains with Adam
    for `steps` steps on mini-batches of `batch_size`, seeded by `seed`.
    """
    names = [str(label) for label in labels]
    classes = tuple(classes) if classes is not None else tuple(sorted(set(names)))
    unknown = sorted(set(names) - set(classes))
    if unknown:
        raise ValueError(f"Labels {unknown[:5]} are not in classes {classes}")
    if len(set(names)) < 2:
        raise ValueError("Training labels must contain at least two classes")

    vectors = pool(embeddings, pooling)
    if len(vectors) != len(names):
        raise ValueError(f"{len(vectors)} embeddings but {len(names)} labels")
    config = AdapterConfig(
        classes=classes, in_dim=vectors.shape[1], hidden=hidden, dropout=dropout if hidden else 0.0,
        pooling=pooling, embedding=embedding, source=source, model=dict(model or {}),
    )
    targets = torch.tensor([classes.index(n) for n in names])
    counts = torch.bincount(targets, minlength=len(classes)).float()
    weight = (counts.sum() / (counts.clamp_min(1) * (counts > 0).sum())) * (counts > 0) if balanced else None

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        adapter = Adapter(config)
        adapter.mean.copy_(vectors.mean(0))
        adapter.std.copy_(vectors.std(0, unbiased=False).clamp_min(1e-3))
        adapter.to(device).train()
        x, y = vectors.to(device), targets.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
        weights = [m.weight for m in adapter.net.modules() if isinstance(m, nn.Linear)]

        def objective(rows: torch.Tensor | slice) -> torch.Tensor:
            return loss_fn(adapter(x[rows]), y[rows]) + l2 * sum(w.pow(2).sum() for w in weights)

        if not hidden:
            optimizer = torch.optim.LBFGS(
                adapter.net.parameters(), max_iter=steps, history_size=20, line_search_fn="strong_wolfe"
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss = objective(slice(None))
                loss.backward()
                return loss

            optimizer.step(closure)
        else:
            adam = torch.optim.Adam(adapter.net.parameters(), lr=lr)
            generator = torch.Generator().manual_seed(seed)
            for _ in range(steps):
                rows = torch.randperm(len(x), generator=generator)[:batch_size].to(device) if len(x) > batch_size else slice(None)
                loss = objective(rows)
                adam.zero_grad(set_to_none=True)
                loss.backward()
                adam.step()
    return adapter.cpu().eval()


def load_cached_embeddings(
    features: str | Path,
    manifest: str | Path,
    labels: str | Mapping[str, Any],
    *,
    split: str | None = "train",
    datasets: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str], list[ClipRecord]]:
    """Cached summary tokens with one label per clip: (tokens (N, 25, D), labels, records).

    `labels` is either a taxonomy label name, read from the manifest's ground truth only
    (teacher labels are never used as targets here), or a mapping clip_id -> label for a
    label set of your own. Binary labels become "0" / "1"; regression labels are refused.
    Pass `features_source(features)` as `fit_adapter(model=...)` to record which encoder
    produced the tokens.
    """
    from bulwark.train.common import load_feature_dir

    records = {r.clip_id: r for r in read_manifest(Path(manifest))}
    clip_ids, tokens, _ = load_feature_dir(Path(features))
    lookup = _manifest_labels(records.values(), labels) if isinstance(labels, str) else {k: str(v) for k, v in labels.items()}
    rows, names, kept = [], [], []
    for row, clip_id in enumerate(clip_ids):
        record = records.get(clip_id)
        if record is None or clip_id not in lookup:
            continue
        if split is not None and record.split != split:
            continue
        if datasets and record.dataset not in datasets:
            continue
        rows.append(row)
        names.append(lookup[clip_id])
        kept.append(record)
    if not rows:
        raise ValueError(f"No cached clips with labels for {labels if isinstance(labels, str) else 'the mapping'} in split {split!r}")
    return tokens[rows].astype(np.float32), names, kept


def features_source(features: str | Path) -> dict[str, Any]:
    """`embedding_source` of the tokens in a `bulwark train cache` directory, from its meta.json.

    Tokens cached without `init` come from the frozen encoder ("frozen"); with `init` they
    come from that B1 checkpoint (its sha256). Empty when the directory has no meta.json.
    """
    meta_path = Path(features) / "meta.json"
    if not meta_path.exists():
        return {}
    run = json.loads(meta_path.read_text(encoding="utf-8")).get("run") or {}
    init = run.get("init")
    return {"family": "b1", "weights": init.get("sha256") if init else "frozen"}


def _manifest_labels(records: Any, name: str) -> dict[str, str]:
    spec = default_taxonomy()[name]
    if spec.kind == "regression":
        raise ValueError(f"{name} is a regression label; adapters classify")
    out = {}
    for record in records:
        if name in record.labels_gt:
            value = record.labels_gt[name]
            out[record.clip_id] = str(int(float(value) >= 0.5)) if spec.kind == "binary" else str(value)
    return out
