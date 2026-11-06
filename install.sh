#!/bin/bash
# usage: ./install.sh /path/to/dt-core
set -e
DT=${1:?usage: ./install.sh /path/to/dt-core}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -d "$DT/packages" ] || { echo "$DT doesn't look like dt-core"; exit 1; }
rm -rf "$DT/packages/ente_nav"          # replace, never leave a stale copy behind
cp -r "$HERE/packages/ente_nav" "$DT/packages/"
cp "$HERE/launchers/ente-nav.sh" "$DT/launchers/"
chmod +x "$DT/launchers/ente-nav.sh" "$DT/packages/ente_nav/src/"*.py "$DT/packages/ente_nav/tools/"*.py
for dep in flask dt-apriltags onnxruntime pyyaml; do
  grep -qi "^$dep" "$DT/dependencies-py3.txt" 2>/dev/null || echo "$dep" >> "$DT/dependencies-py3.txt"
done
echo "Installed. Next:"
echo "  1. (duckie.onnx is already included in packages/ente_nav/models/)"
echo "  2. cd $DT && dts devel build -H entebot209"
echo "  3. dts devel run -H entebot209 -L ente-nav"
echo "  4. open http://entebot209.local:8090"
echo "  5. verify it drives: python3 $DT/packages/ente_nav/tools/verify_motion.py entebot209.local"
