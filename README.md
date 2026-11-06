# ente_nav for entebot209
Install:  ./install.sh ~/dt-core
Build:    cd ~/dt-core && dts devel build -H entebot209
Run:      dts devel run -H entebot209 -L ente-nav      -> http://entebot209.local:8090

Models:   packages/ente_nav/models/duckie.onnx = best.onnx from lx-object-detection-ente (YOLO11n, class: yellow rubber duck)
          Converted opset 18 -> 17 so the robot's onnxruntime 1.12.1 loads it (dt-core pins numpy<=1.21.5, so ort can't be upgraded).
          Outputs are bit-identical; the original is kept as duckie_opset18_original.onnx (not installed).
Needle:   see tools/test_needle.py and tools/nlp_server.py (runs on the laptop, no API key needed)
