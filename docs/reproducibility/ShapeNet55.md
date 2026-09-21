# ShapeNet-55 Reproducibility

This repository distinguishes **checkpoint evaluation reproducibility** from **bitwise training-trajectory reproducibility**.

## What is verified

For the canonical ShapeNet-55 training implementation, the released code was audited against the historical implementation.

The following were verified to match exactly in the audited environment:

- model initialization;
- the first sampled training batch;
- model forward outputs;
- classification and completion losses.

Evaluation of the released canonical checkpoints reproduces the paper-reported metrics to the displayed precision.

## CUDA backward reproducibility

The historical training implementation uses custom CUDA operators. Its backward pass is not bitwise deterministic.

Repeated executions of the historical implementation with the same environment, random seed, initialized weights, sampled batch, forward outputs, and loss values produced numerically almost identical but not bitwise-identical gradient tensors.

Therefore this repository does **not** claim that repeated training runs produce bitwise-identical gradient tensors or bitwise-identical final weights.

Instead, backward reproducibility is defined numerically. The released implementation was compared with the historical implementation across the canonical modes `baseline`, `full`, `g2s`, `s2g`, and `ours`.

For all five modes, the old-vs-release gradient differences remained within the empirical numerical variation observed between repeated runs of the historical implementation itself.

The corresponding public contract is stored at:

`expected_results/reproducibility/shapenet55_training_contract.json`

## Full retraining

A new full 300-epoch OURS retraining is not required for checkpoint-level paper reproduction and is not used as a release blocker.

Such a run may be performed later to establish an empirical end-to-end final-metric tolerance for from-scratch retraining.

Until that experiment is completed, this repository makes the following claims:

| Reproducibility level | Status |
|---|---|
| Released checkpoint evaluation | Verified |
| Training protocol | Verified |
| Model initialization | Bitwise verified |
| First training batch | Bitwise verified |
| Forward outputs | Bitwise verified |
| Training losses | Bitwise verified |
| CUDA backward | Numerically verified |
| Bitwise complete training trajectory | Not claimed |
| New full 300-epoch OURS rerun | Deferred |

## Recommended interpretation

A fixed random seed controls the random-number streams used by the training protocol, but it cannot by itself guarantee bitwise determinism for every custom parallel CUDA backward operation.

Users should expect the released checkpoints to reproduce the reported evaluation results, while independent from-scratch training runs may exhibit small numerical differences in their optimization trajectories.
