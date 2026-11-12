# Examples

Scripts that use the public API (`bulwark.inference`, `bulwark.adapters`). Run them from the
repository root with `PYTHONPATH=.` or after `pip install -e .`. The API is described in
[docs/inference-and-adapters.md](../docs/inference-and-adapters.md).

| Script | Needs | What it does |
|---|---|---|
| `adapter_quickstart.py` | nothing | Fits an adapter on synthetic embeddings, scores it, saves it and loads it back. |
| `predict_video.py` | a model, a video or an image | Runs B1, E1, an ONNX graph or a release over a video and writes a per-second JSON timeline, optionally with adapters and a calibration.json (the temperatures and thresholds of the input's media type); an image gives one entry. Adds PERCLOS over the last 60 s when the model outputs `perclos`. `bulwark predict` does the same from the command line (`--perclos-seconds`, `--calibration`, `--out`). |
| `train_adapter.py` | a model, a CSV of labelled clips | Embeds each clip, scores an adapter on held-out subjects if the CSV names them, and saves the adapter. |
| `fewshot_curve.py` | a `bulwark train cache` features dir and its manifest | Measures adapter accuracy against the number of labelled clips per class on a subject-disjoint split. |
| `make_release.py` | a B1, heads or E1 checkpoint | Writes a release directory (safetensors weights, config.json, the V-JEPA notice for B1) that loads without the pretrained V-JEPA file, under the research licence unless `--weights-licence` names another. Refuses a licence that does not restrict use to non-commercial research unless the training data, including that of upstream models, was commercially clean. config.json records the `bulwark release-check` verdict. `bulwark release` does the same from the command line. |

```bash
PYTHONPATH=. python examples/adapter_quickstart.py
PYTHONPATH=. python examples/predict_video.py checkpoints/b1-lora-v0/best.pt cabin.mp4 --out runs/timeline.json
PYTHONPATH=. python examples/train_adapter.py checkpoints/b1-lora-v0/best.pt clips.csv --out adapters/my_label.pt
PYTHONPATH=. python examples/fewshot_curve.py --features data/features/frozen \
    --manifest data/manifest.parquet --label activity --dataset drive_and_act
PYTHONPATH=. python examples/make_release.py checkpoints/e1-v0/best.pt releases/bulwark-e1-v0 \
    --temperature runs/eval-e1/calibration.json
```

A model here is any path `bulwark.inference.load` accepts: a B1 checkpoint from
`bulwark train lora`, a heads checkpoint from `bulwark train heads` (frozen encoder), an E1
checkpoint from `bulwark train distill` (`best.pt`, `last.pt` or `final.pt` in the run's
`out` directory, for example `checkpoints/b1-lora-v0/` and `checkpoints/e1-v0/` with the
v0 configs), an exported `.onnx` file, or a release directory. B1 and heads checkpoints
also need the V-JEPA code and weights from `scripts/fetch_pretrained.sh`.

The v0 weights (Bulwark-B1 v0 and Bulwark-E1 v0) are for non-commercial research use
only, and that includes using them to compute embeddings and train adapters.
