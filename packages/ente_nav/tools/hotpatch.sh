#!/bin/bash
# Copy changed ente_nav sources into the running container and restart it (fast iteration; rebuild before finishing).
C=dts-run-dt-core; D="docker -H entebot209.local"; P=/code/src/dt-core/packages/ente_nav; L=~/dt-core/packages/ente_nav
for f in src/ente_brain_node.py include/ente_nav/*.py config/ente_nav.yaml web/index.html; do
  for g in $L/$f; do $D cp "$g" "$C:$P/${g#$L/}"; done
done
$D restart -t 3 $C >/dev/null
for i in $(seq 1 40); do sleep 2; curl -s -m2 http://entebot209.local:8090/api/state >/dev/null && { echo "UP after $((i*2))s"; exit 0; }; done
echo "NOT UP"; $D logs --tail 30 $C; exit 1
