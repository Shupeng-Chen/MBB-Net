# KITTI Zero-Shot Reproducibility

## Protocol

- Source training: PCN
- KITTI fine-tuning: None
- ShapeNetCars fine-tuning: None
- Target-domain adaptation: None
- Fixed KITTI inputs: 2,401 samples
- Input points: 2,048
- Output points: 16,384
- Released paper modes:
  - CompletionOnly: `checkpoints/PCN/pcn_mbb_ablation_comp_only_best.pth`
    SHA256: `f29e541ff5b581bc99129f358bd64a9aad74498d781eb89dbe160df88a7e2b33`
  - MBB-Net: `checkpoints/PCN/pcn_mbb_ablation_ours_best.pth`
    SHA256: `b32d7a03dd759741ded8ca22b4fb48e60d7f5039cee120050229875058f36940`
- Seed: 42

## Inference implementation validation

The portable release inference was compared against the frozen historical fixed-input predictions for both released paper modes.

Validation results:

- CompletionOnly: strict-load 287 tensors; 8 / 8 predictions bitwise identical; global max_abs = 0.0
- MBB-Net: strict-load 328 tensors; 8 / 8 predictions bitwise identical; global max_abs = 0.0
- For both modes, the generated `.npy` prediction files were also 8 / 8 byte-identical to the historical fixed-input outputs.

The complete historical fixed-input prediction sets contain 2,401 KITTI samples per method. A new full 2,401-sample inference rerun was not repeated during final release auditing because the public implementation reproduced the historical predictions exactly on the audited smoke-test subset.

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

## Paper results

The stored official-equivalent 2,401-sample results are:

| Method | Fidelity-L2 x1000 | MMD-CD-L2 x1000 |
|---|---:|---:|
| CompletionOnly | 1.0180 | 0.7528 |
| MBB-Net | 1.1513 | 0.7891 |

Exact stored values:

- CompletionOnly Fidelity: `1.0180427590680878`
- CompletionOnly MMD: `0.7528334427342751`
- MBB-Net Fidelity: `1.1513109267036241`
- MBB-Net MMD: `0.7890661099725099`

The CompletionOnly values come from the stored historical official-equivalent full evaluation. The full old-vs-release evaluator migration audit described below was performed on the frozen MBB-Net 2,401-sample prediction set.

## MBB-Net full metric migration validation

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
