# Datasets

MBB-Net uses the processed PCN, ShapeNet-55/34, ShapeNet-Unseen21 and KITTI benchmark conventions used by PoinTr and SymmCompletion.

Upstream preparation references:

- PoinTr: https://github.com/yuxumin/PoinTr/blob/master/DATASET.md
- SymmCompletion: https://github.com/HKUST-SAIL/SymmCompletion

Large benchmark files are not committed to this source repository.

## Recommended layout

```text
MBB-Net/
└── data/
    ├── PCN/
    ├── ShapeNet55-34/
    ├── KITTI/
    │   ├── bboxes/
    │   ├── cars/
    │   ├── tracklets/
    │   ├── KITTI.json
    │   └── fixed_inputs/
    └── splits/
        ├── ShapeNet-55/
        ├── ShapeNet-34/
        └── ShapeNet-Unseen21/
```

## PCN

Use the processed PCN / ShapeNetCompletion benchmark distributed through the PoinTr/PCN ecosystem.
The released commands accept its root through `--data_root`.

## ShapeNet-55 / ShapeNet-34 / ShapeNet-Unseen21

Use the processed ShapeNet55-34 point-cloud data distributed by PoinTr.

The exact MBB-Net split metadata is included in:

```text
data/splits/ShapeNet-55/train.txt
data/splits/ShapeNet-55/test.txt
data/splits/ShapeNet-34/train.txt
data/splits/ShapeNet-34/test.txt
data/splits/ShapeNet-Unseen21/test.txt
data/splits/shapenet_synset_dict.json
```

ShapeNet-Unseen21 is evaluation-only and uses checkpoints trained on the 34 seen categories.

## KITTI

The processed KITTI layout follows PoinTr / SymmCompletion:

```text
data/KITTI/
├── bboxes/
├── cars/
├── tracklets/
└── KITTI.json
```

For exact MBB-Net paper reproduction, additionally use the separately distributed `MBB-Net-KITTI-fixed-inputs.tar.gz`.

Extract its 2,401 float32 point clouds into `data/KITTI/fixed_inputs/`.
Each file is `<sample_id>_input.npy` with shape `(2048, 3)`.

KITTI is zero-shot: no KITTI, ShapeNetCars, or target-domain fine-tuning is used.
Inference therefore uses `checkpoints/PCN/pcn_mbb_ablation_ours_best.pth`.

The exact fixed-input and metric protocols are locked by:

```text
expected_results/KITTI/fixed_inputs_manifest.json
expected_results/KITTI/fixed_inputs.sha256
docs/reproducibility/KITTI.md
```
