# Structural Benchmark Results

**Method:** last-stage-capacity `WidthScaler` vs torch-pruning `MetaPruner`
**Hardware:** CPU, FP32, batch size 1, torch 2.12.1
**Date:** 2026-07-01

## Scope

This benchmark measures **structural compression on a controlled CNN stack** — a 4-layer Conv-BN-ReLU-MaxPool architecture with no skip connections or grouped convolutions. This isolates the channel-scaling primitive in both libraries.

**What this benchmark does NOT measure:** accuracy on real datasets. Both libraries produce untrained compressed models. Accuracy claims require fine-tuning runs on real datasets, which is out of scope for this script.

**Why not ResNet/EfficientNet?** torchvision models have skip connections and grouped convolutions that interact non-trivially with both compression libraries. A fair comparison on those requires fine-tuning evaluation (see the methodology notes at the bottom). The controlled stack isolates the channel-scaling primitive.

## Methodology

- **Input:** 1x3x64x64
- **Width ratios:** [1.0, 0.75, 0.5, 0.25]
- **last-stage-capacity:** `WidthScaler(features)` — uniform channel scaling through the Sequential's conv/BN pairs with channel propagation
- **torch-pruning:** `MetaPruner` with `MagnitudeImportance(p=2)`, layer-wise (not global), ignoring the final classifier
- **Latency:** median of 20 CPU forward passes after 5 warmup runs

## Results

| Width | Method | Params (K) | FLOPs (M) | Latency (ms) |
|---|---|---|---|---|
| 1.00 | baseline | 1557.1 | 233.58 | 1.1 |
| 0.75 | last-stage-capacity | 877.5 | 132.71 | 0.8 |
| 0.75 | torch-pruning | 877.5 | 132.71 | 0.8 |
| 0.50 | last-stage-capacity | 391.5 | 60.16 | 0.7 |
| 0.50 | torch-pruning | 391.5 | 60.16 | 0.6 |
| 0.25 | last-stage-capacity | 99.0 | 15.93 | 0.4 |
| 0.25 | torch-pruning | 99.0 | 15.93 | 0.4 |

## Compression Ratios (vs baseline)

| Width | Method | Param Ratio | FLOPs Ratio | Latency Ratio |
|---|---|---|---|---|
| 0.75 | last-stage-capacity | 0.56x | 0.57x | 0.74x |
| 0.75 | torch-pruning | 0.56x | 0.57x | 0.72x |
| 0.50 | last-stage-capacity | 0.25x | 0.26x | 0.61x |
| 0.50 | torch-pruning | 0.25x | 0.26x | 0.59x |
| 0.25 | last-stage-capacity | 0.06x | 0.07x | 0.39x |
| 0.25 | torch-pruning | 0.06x | 0.07x | 0.32x |

## What This Tells Us

- Both libraries successfully reduce parameters and FLOPs at sub-1.0 width ratios.
- **last-stage-capacity's WidthScaler** scales uniformly — all layers reduced by the same factor, preserving structural balance.
- **torch-pruning's MetaPruner** prunes non-uniformly — magnitude importance removes channels with lowest L2 norm across layers, which can produce uneven compression.
- Latency reduction is smaller than param/FLOPs reduction because of constant overhead (Python interpreter, BatchNorm, etc.).

## Reproducing

```bash
python -m venv .venv
source .venv/bin/activate
pip install last-stage-capacity torch-pruning
python benchmarks/structural.py
```

Output writes to `benchmarks/RESULTS.md` (this file) and `benchmarks/results.json`.
