#!/bin/bash
# Stop the running ente_nav container on the robot and start the freshly built image (code baked in, no rsync).
docker -H entebot209.local rm -f dts-run-dt-core >/dev/null 2>&1
cd ~/dt-core && (nohup dts devel run -H entebot209 -L ente-nav --no-mount > ~/ente_nav_run.log 2>&1 &)
for i in $(seq 1 50); do sleep 3; curl -s -m2 http://entebot209.local:8090/api/state >/dev/null && { echo "UP after $((i*3))s"; exit 0; }; done
echo "NOT UP"; tail -20 ~/ente_nav_run.log; exit 1
