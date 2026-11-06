#!/bin/bash
source /environment.sh
dt-launchfile-init
dt-exec roslaunch ente_nav ente_nav.launch
dt-launchfile-join
