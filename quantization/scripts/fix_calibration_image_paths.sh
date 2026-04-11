#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────
# 修复校准数据中的 Flickr8K 图片路径
#
# 问题：校准 JSON 中的 image_path 指向旧服务器路径：
#   /data/14thdd/users/yongsencheng/Bagel/data/flickr8k/Images/...
# 当前服务器上该路径不存在，导致所有图片加载失败。
#
# 解决方案：
#   1. 下载 Flickr8K 数据集到本地
#   2. 运行本脚本更新路径
#
# 用法：
#   # 方法 A：直接更新已有校准 JSON 的路径
#   bash scripts/fix_calibration_image_paths.sh --new_root /path/to/flickr8k
#
#   # 方法 B：重新生成校准数据集
#   bash scripts/fix_calibration_image_paths.sh --rebuild --new_root /path/to/flickr8k
# ──────────────────────────────────────────────────────────────────

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CALIB_JSON="${QUANT_ROOT}/quantization_outputs/calibration_data/calibration_dataset_1000samples_20260324.json"
OLD_PREFIX="/data/14thdd/users/yongsencheng/Bagel/data/flickr8k"

NEW_ROOT=""
REBUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --new_root) NEW_ROOT="$2"; shift 2 ;;
        --rebuild) REBUILD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$NEW_ROOT" ]]; then
    echo "Usage: bash $0 --new_root /path/to/flickr8k [--rebuild]"
    echo ""
    echo "Flickr8K 数据集结构应为："
    echo "  /path/to/flickr8k/"
    echo "    ├── Images/"
    echo "    │   ├── 1000268201_693b08cb0e.jpg"
    echo "    │   ├── ..."
    echo "    └── captions.txt"
    echo ""
    echo "下载方式（任选一种）："
    echo "  1. Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k"
    echo "  2. HuggingFace: huggingface-cli download nlphuji/flickr_1k_test_image --local-dir /path/to/flickr8k"
    echo "  3. 从其他服务器 scp: scp -r server:/data/14thdd/users/yongsencheng/Bagel/data/flickr8k /path/to/"
    exit 1
fi

echo "=================================================="
echo "Fix Calibration Image Paths"
echo "=================================================="
echo "  Old prefix: ${OLD_PREFIX}"
echo "  New root:   ${NEW_ROOT}"
echo "  Calib JSON: ${CALIB_JSON}"

if [[ ! -d "${NEW_ROOT}/Images" ]]; then
    echo ""
    echo "ERROR: ${NEW_ROOT}/Images directory not found!"
    echo "Please ensure Flickr8K Images/ directory exists at the specified path."
    exit 1
fi

IMG_COUNT=$(ls "${NEW_ROOT}/Images/"*.jpg 2>/dev/null | wc -l)
echo "  Images found: ${IMG_COUNT}"

if [[ "$REBUILD" == "true" ]]; then
    echo ""
    echo "Rebuilding calibration dataset..."
    cd "${QUANT_ROOT}"
    python utils/build_calibration_dataset.py \
        --flickr8k_root "${NEW_ROOT}" \
        --num_samples 1000 \
        --seed 42
    echo "Done! New calibration dataset generated."
else
    if [[ ! -f "${CALIB_JSON}" ]]; then
        echo "ERROR: Calibration JSON not found: ${CALIB_JSON}"
        exit 1
    fi

    BACKUP="${CALIB_JSON}.bak"
    cp "${CALIB_JSON}" "${BACKUP}"
    echo "  Backup: ${BACKUP}"

    python3 -c "
import json, sys
calib = '${CALIB_JSON}'
old = '${OLD_PREFIX}'
new = '${NEW_ROOT}'
with open(calib, 'r') as f:
    data = json.load(f)
count = 0
for s in data.get('samples', []):
    ip = s.get('image_path', '')
    if ip.startswith(old):
        s['image_path'] = ip.replace(old, new, 1)
        count += 1
with open(calib, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f'  Updated {count} image paths')
"

    echo ""
    echo "Verifying..."
    python3 -c "
import json
from pathlib import Path
with open('${CALIB_JSON}') as f:
    data = json.load(f)
samples = data.get('samples', [])
found = sum(1 for s in samples if Path(s.get('image_path','')).exists())
print(f'  Images accessible: {found}/{len(samples)}')
if found == 0:
    print('  WARNING: No images found at new paths! Check the path.')
elif found < len(samples):
    print(f'  WARNING: {len(samples) - found} images still missing.')
else:
    print('  All images accessible!')
"
fi

echo ""
echo "Done! Re-run the experiments to get correct vision token analysis."
