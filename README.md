# MBB-Net

### MBB-Net: Asymmetric Decoupling Bidirectional Bridge for Point Cloud Completion and Classification


This repository provides the official PyTorch implementation of **MBB-Net**, a multi-task framework for joint point cloud completion and classification.


# ✨ News

- Code and reproducibility materials are prepared for release.
- Released checkpoints are accompanied by SHA256 hashes and paper-facing verification files.

# Introduction

MBB-Net studies joint point cloud completion and classification with separate geometric and semantic representations. The core **Asymmetric Decoupling Bidirectional Bridge (MBB)** keeps bidirectional forward feature interaction while restricting selected backward gradient paths between the two task streams.

The released code covers experiments on **PCN**, **ShapeNet-55**, **ShapeNet-34 / ShapeNet-Unseen21**, **KITTI**, and cross-backbone transfer to **SymmCompletion**.

# Installation

```bash

conda create --name MBBNet python=3.8 -y
conda activate MBBNet

pip install torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
sh extensions/install.sh
```

`extensions/install.sh` builds the CUDA extensions required by the released SnowflakeNet and SymmCompletion code paths.

# Training and Testing

## 1. Download datasets

Prepare the original datasets according to their official releases.

- **PCN**
- **ShapeNet-55 / ShapeNet-34**
- **KITTI**

Dataset files are not redistributed in this repository.

## 2. Set dataset paths

### PCN

Pass the original PCN / ShapeNetCompletion root with `--data_root`.

```text
<PCN_ROOT>/
```

### ShapeNet-55

```text
<SHAPENET55_ROOT>/
├── train.txt
├── test.txt
└── shapenet_pc/
```

### ShapeNet-34 / ShapeNet-Unseen21

The released category split files are included in:

```text
data/splits/ShapeNet-34/
data/splits/ShapeNet-Unseen21/
```

The point-cloud files reuse the ShapeNet `shapenet_pc/` directory.

### KITTI

The zero-shot experiment uses the fixed KITTI input protocol documented in:

```text
docs/reproducibility/KITTI.md
```

## 3. Training

### PCN

Train the canonical PCN MBB-Net configuration with:

```bash
python main.py pcn-train \
    --data_root <PCN_ROOT>
```

The reference configuration uses seed 42 and selects the canonical checkpoint by the lowest validation CD-L1 x1000. The PCN loader preloads the training set by default, so the first startup may take several minutes.

For all available options:

```bash
python tools/train/train_pcn.py --help
```

### ShapeNet-55

The public ShapeNet-55 training entry point supports the main controlled configurations:

```text
baseline
full
g2s
s2g
ours
```

Train the canonical MBB configuration with:

```bash
python main.py shapenet55-train \
    --mode ours \
    --data-root <SHAPENET55_ROOT>
```

The default training configuration is:

```text
cfgs/ShapeNet55_config.yaml
```

For all available options:

```bash
python tools/train/train_shapenet55.py --help
```

## 4. Testing

### PCN

```bash
python main.py pcn-test \
    --data_root <PCN_ROOT> \
    --checkpoint checkpoints/PCN/pcn_mbb_ablation_ours_best.pth
```

### ShapeNet-55

```bash
python main.py shapenet55-test \
    --dataset 55 \
    --mode ours \
    --data_root <SHAPENET55_ROOT>
```

The released evaluator also supports:

```text
baseline
cls_only
no_bridge
full
g2s
s2g
ours
pcgrad_ours
```

### ShapeNet-34

```bash
python main.py shapenet34-test \
    --mode baseline no_bridge ours \
    --data_root <SHAPENET_ROOT> \
    --split_root data/splits
```

### ShapeNet-Unseen21

```bash
python main.py shapenet21-test \
    --mode baseline no_bridge ours \
    --data_root <SHAPENET_ROOT> \
    --split_root data/splits
```

### KITTI zero-shot inference

```bash
python main.py kitti-infer \
    --fixed_input_dir <KITTI_FIXED_INPUT_DIR> \
    --checkpoint checkpoints/PCN/pcn_mbb_ablation_ours_best.pth \
    --output_dir outputs/KITTI/MBB_Ours_PCN_FixedInput
```

Evaluate generated KITTI predictions with:

```bash
python main.py kitti-eval \
    --results_root outputs/KITTI \
    --methods MBB_Ours_PCN_FixedInput \
    --pcn_cars_root <PCN_ROOT> \
    --backend exact_batch \
    --input_check strict
```

# Pretrained Models

Checkpoint binaries are distributed separately from the Git source repository.

After downloading the checkpoint package, place the files under:

```text
checkpoints/
```

The exact checkpoint layout, file sizes, and SHA256 hashes are provided in:

```text
checkpoints/README.md
checkpoints/MANIFEST.sha256
```

Verify downloaded checkpoints with:

```bash
cd checkpoints
sha256sum -c MANIFEST.sha256
cd ..
```

# Reproducibility

The public repository includes paper-facing expected results, protocol notes, and lightweight verification scripts.

Verify the reported complexity table:

```bash
python main.py table4
```

Verify the released Figure-5 gradient-interaction statistics and checkpoint provenance:

```bash
python main.py fig5
```

Detailed protocol notes are provided in:

```text
docs/reproducibility/
expected_results/
```

Released-checkpoint evaluation is intended to reproduce the reported values at the displayed precision. Training uses fixed seeds and the documented data/configuration protocol; exact bitwise-identical parameter trajectories across different GPUs, CUDA versions, drivers, or PyTorch builds are not assumed.

# Visualized Results

The assets directory is reserved for project figures and qualitative visualizations; it is not required for quantitative reproduction.

```text
assets/
```

# Acknowledgements

Our implementation builds on open-source point-cloud research code, including:

- **SnowflakeNet**
- **SymmCompletion**
- **PointNet++ / PointNet2 CUDA operators**

We thank the authors for making their implementations publicly available.

Third-party components remain subject to their original licenses.
