# KITTI

MBB-Net follows the processed KITTI-Cars data convention used by PoinTr and SymmCompletion.

Raw processed layout:

```text
data/KITTI/
├── bboxes/
├── cars/
├── tracklets/
└── KITTI.json
```

The MBB-Net paper uses KITTI only for zero-shot evaluation.

For exact paper reproduction, extract `MBB-Net-KITTI-fixed-inputs.tar.gz` into:

```text
data/KITTI/fixed_inputs/
```

The directory must contain exactly 2,401 `<sample_id>_input.npy` files, each with shape `(2048, 3)`.

KITTI uses the PCN-trained checkpoint:

```text
checkpoints/PCN/pcn_mbb_ablation_ours_best.pth
```

No KITTI-specific MBB-Net checkpoint exists because no KITTI fine-tuning is performed.

Integrity manifests:

```text
expected_results/KITTI/fixed_inputs_manifest.json
expected_results/KITTI/fixed_inputs.sha256
```

See `docs/reproducibility/KITTI.md` for the exact FD/MMD protocol.
