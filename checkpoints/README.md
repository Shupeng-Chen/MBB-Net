# Pretrained Checkpoints

The pretrained checkpoint binaries are distributed separately from the Git repository because the complete set is large.

After downloading the checkpoint package or GitHub Release assets, place the files under `checkpoints/` using the exact relative paths listed below.

Verify all files with:

```bash
cd checkpoints
sha256sum -c MANIFEST.sha256
```

| Checkpoint | Size (MiB) | SHA256 |
|---|---:|---|
| `PCN/pcn_mbb_ablation_ours_best.pth` | 82.65 | `b32d7a03dd759741ded8ca22b4fb48e60d7f5039cee120050229875058f36940` |
| `ShapeNet/ShapeNet34/baseline/best_model.pth` | 73.77 | `ad13d0a08ef7f454a53e8527d93bc8ccd85f607f22601c866cc8219b35ed2b8c` |
| `ShapeNet/ShapeNet34/no_bridge/best_model.pth` | 82.61 | `ac1da7463bf80183a702292a66435c842adb621026f334fbb6e86c9de2702baf` |
| `ShapeNet/ShapeNet34/ours/best_model.pth` | 82.61 | `1f5bd6003dd2516d0d572ba4d113a3823d34278679c3f3b77c8e908c9e8c16a7` |
| `ShapeNet/ShapeNet55/baseline/best_model.pth` | 73.77 | `37ca7509faa949cf3c4557fd7f02eaf29d4c13dcb0118dbaf084452028b79d15` |
| `ShapeNet/ShapeNet55/cls_only/best_model.pth` | 82.63 | `5b6d14dd35046938591521c51bcca2d3b2b665cb08be6059dfb66cd3ab034f16` |
| `ShapeNet/ShapeNet55/full/best_model.pth` | 82.63 | `1bfc68daeb01d6b5c3a4645d60b17751fd86503db73b107f6fd59f876182208f` |
| `ShapeNet/ShapeNet55/g2s/best_model.pth` | 82.63 | `e669f3097eb9b9e25e07dd05d045cf103ecb6fd46c5a19a07b261cf297174cc6` |
| `ShapeNet/ShapeNet55/no_bridge/best_model.pth` | 82.63 | `5a1c2b8c31cf0ff2046c91f484f40c59b3f96d957c6db32747376e9a1d666b7c` |
| `ShapeNet/ShapeNet55/ours/best_model.pth` | 82.63 | `4be6989cf1754381b4c6826a9c53f36bd9adceff9c7d3013ea36bce24a4de55d` |
| `ShapeNet/ShapeNet55/pcgrad_ours_aligned/best_model.pth` | 82.63 | `50a295b937decb03ea15c69eb4cc1a1249c45cb2ea44ef29c6048fbefead5246` |
| `ShapeNet/ShapeNet55/s2g/best_model.pth` | 82.63 | `71633b4e66ce719d99910f8335773eaa125151fc3a23231ea3643c8da1c6597b` |
| `SymmCompletion/ShapeNet55/baseline/ckpt-best.pth` | 152.35 | `137ad2fac429146832f3c3b2fd5719a24770143f65f5699ab665c143a612f923` |
| `SymmCompletion/ShapeNet55/no_bridge/ckpt-best.pth` | 51.64 | `5f95558396884735ac4885eb82a73fa255e34854a2ec3210137f1078e52d227b` |
| `SymmCompletion/ShapeNet55/ours/ckpt-best.pth` | 60.17 | `cbac28ca63f2cfb0d24c07e1ed606d5e6b4d07d758909287413ca4837fccb8bc` |
