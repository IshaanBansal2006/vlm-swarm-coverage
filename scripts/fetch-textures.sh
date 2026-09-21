#!/usr/bin/env bash
# Fetch the CC0 ground and road textures the Isaac scene projects onto its surfaces.
#   bash scripts/fetch-textures.sh            -> sim/assets/textures/*.jpg (2k diffuse maps)
# Source: Poly Haven (https://polyhaven.com), CC0 1.0. Names are Poly Haven asset ids.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$HERE/sim/assets/textures"
mkdir -p "$OUT"
for n in sparse_grass clean_asphalt road_damaged aerial_ground_rock aerial_mud_1 leafy_grass; do
  f="$OUT/$n.jpg"
  if [ -s "$f" ]; then echo "have   $n"; continue; fi
  curl -sSL -o "$f" "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/$n/${n}_diff_2k.jpg"
  echo "fetched $n ($(stat -c %s "$f") bytes)"
done
