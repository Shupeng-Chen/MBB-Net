# Datasets

MBB-Net uses the PCN, ShapeNet-55/34/21, and KITTI benchmarks.

Large dataset files are not included in this repository.
Please download the datasets separately and provide their local paths through the corresponding training or evaluation commands.

## ShapeNet-55 / ShapeNet-34 / ShapeNet-Unseen21

The released category splits follow the standard PoinTr / SymmCompletion benchmark protocol.

```text
data/splits/
├── shapenet_synset_dict.json
├── ShapeNet-55/
│   ├── train.txt
│   └── test.txt
├── ShapeNet-34/
│   ├── train.txt
│   └── test.txt
└── ShapeNet-Unseen21/
    └── test.txt
```

ShapeNet-34 is used for training and Seen-34 evaluation. ShapeNet-Unseen21 is evaluation-only and uses the ShapeNet-34-trained checkpoint.

## PCN

PCN experiments use the standard 8-category completion benchmark with 2048-point partial inputs and 16384-point complete point clouds.

## KITTI

KITTI is used only for zero-shot real-scan evaluation with PCN-trained completion models.

Dataset licenses and redistribution terms remain subject to the original dataset providers.
