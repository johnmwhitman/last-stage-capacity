# Last-Stage Capacity Reduction Library

Patterns for progressively narrowing neural network representations in the later stages of a model.

## Core Principle

Early layers capture low-level features; later layers must compress these into task-specific representations. Strategic narrowing at this stage forces beneficial compression, improves regularization, and reduces inference cost without significant accuracy loss.

Grounded in: Huang et al., "Exploring Architectural Ingredients of Adversarially Robust DNNs" (NeurIPS 2021).

## Installation

```bash
pip install last-stage-capacity
```

Or install from source:
```bash
git clone https://github.com/johnmwhitman/last-stage-capacity.git
cd last-stage-capacity
pip install -e .
```

## Library Structure

- `capacity_reduction/`: Core library (last-stage capacity detection and reduction)
- `examples/`: Usage examples
- `tests/`: Test suite

## Usage

```python
from last_stage_capacity import LastStageCapacity

# Analyze a model's last-stage capacity
analyzer = LastStageCapacity(model)
report = analyzer.benchmark()
```

## License

Apache-2.0
