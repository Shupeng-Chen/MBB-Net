# KITTI Zero-Shot Reproducibility

## Protocol

- Source training: PCN
- KITTI fine-tuning: None
- ShapeNetCars fine-tuning: None
- Target-domain adaptation: None
- Fixed KITTI inputs: 2,401 samples
- Input points: 2,048
- Output points: 16,384
- Checkpoint SHA256:
  b32d7a03dd759741ded8ca22b4fb48e60d7f5039cee120050229875058f36940
- Seed: 42

## Inference implementation validation

The portable release implementation uses the static canonical MBB definition.

It was compared against the frozen historical canonical KITTI
runtime-patched implementation.

Validation results:

- model state tensors: 328 vs 328
- unequal state tensors: 0
- same-process canonical old-vs-release prediction:
  max_abs = 0.0
- independent release inference smoke test:
  8 / 8 predictions bitwise identical
- global maximum absolute prediction difference:
  0.0

A full 2,401-sample release inference rerun was not repeated during
release auditing because the canonical 2,401 prediction set had already
been frozen. The released script supports the full inference run.

## Metric protocol

Reference library:

- PCN/ShapeNetCompletion Cars
- train: 5,677
- test: 150
- val: 100
- total references: 5,927
- points per reference: 16,384

Metrics:

- Fidelity: one-way CD-L2, input -> prediction
- MMD: minimum symmetric CD-L2 against all 5,927 complete car references
- reporting scale: x1000
- exact-batch backend
- GT batch size: 64

## Full metric migration validation

The portable release evaluator was rerun on all 2,401 frozen canonical
KITTI predictions.

Strict old-vs-release comparison:

- sample IDs: 2,401 / 2,401
- missing samples: 0
- extra samples: 0
- FD per-sample mismatches: 0
- MMD per-sample mismatches: 0
- metadata mismatches: 0
- maximum FD difference: 0.0
- maximum MMD difference: 0.0

Final canonical result:

- Fidelity-L2 x1000:
  1.1513109267036241
- MMD-CD-L2 x1000:
  0.7890661099725099

Paper precision:

- Fidelity: 1.1513
- MMD: 0.7891
