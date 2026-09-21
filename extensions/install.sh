#!/bin/sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================================"
echo "Installing MBB-Net CUDA extensions"
echo "============================================================"

echo
echo "[1/3] PointNet2"
MAX_JOBS="${MAX_JOBS:-4}" \
python -m pip install \
  --no-deps \
  --no-build-isolation \
  "$ROOT/extensions/pointnet2_ops" || exit 1

echo
echo "[2/3] Chamfer Distance"
MAX_JOBS="${MAX_JOBS:-4}" \
python -m pip install \
  --no-deps \
  --no-build-isolation \
  "$ROOT/extensions/chamfer_dist" || exit 1

echo
echo "[3/3] SymmCompletion legacy PointNet2"
MAX_JOBS="${MAX_JOBS:-4}" \
python -m pip install \
  --no-deps \
  --no-build-isolation \
  "$ROOT/third_party/symmcompletion/pointnet2_legacy" || exit 1

echo
echo "============================================================"
echo "[PASS] MBB-Net CUDA extensions installed."
echo "============================================================"
