"""Tiny Flask server: serves web/index.html, MJPEG camera streams, JSON state and commands."""
import os
import time
from flask import Flask, Response, jsonify, request, send_from_directory


def make_app(brain, web_dir):
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(web_dir, "index.html")

    @app.route("/api/state")
    def state():
        return jsonify(brain.state())

    @app.route("/api/cmd", methods=["POST"])
    def cmd():
        return jsonify(brain.api(request.get_json(force=True, silent=True) or {}))

    @app.route("/api/camera")
    def camera():
        return jsonify({"K": brain.scaled_K(), "D": brain.D, "wh": brain.img_wh, "src": brain.proj.src,
                        "H": None if brain.proj.H is None else brain.proj.H.flatten().tolist(),
                        "pinhole": [brain.cam_height, brain.cam_tilt, brain.cam_forward], "geom": brain.geom,
                        "scale": brain.proj.scale})

    @app.route("/api/snapshot")
    def snapshot():
        jpg = brain.snapshot_jpeg()
        if jpg is None:
            return Response("no camera frame yet", status=503)
        return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.route("/api/frame")
    def frame():
        jpg = brain.render_jpeg(set(request.args.get("layers", "").split(",")))
        if jpg is None:
            return Response("no camera frame yet", status=503)
        return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.route("/stream")
    def stream():
        layers = set(request.args.get("layers", "").split(","))

        def gen():
            while True:
                jpg = brain.render_jpeg(layers)
                if jpg is not None:
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.1)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    return app


def serve(brain, port=8090):
    web_dir = os.path.join(os.path.dirname(__file__), "..", "..", "web")
    web_dir = os.path.abspath(web_dir)
    try:
        make_app(brain, web_dir).run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
    except Exception as e:      # e.g. port already used by another container (host networking)
        brain._log("DASHBOARD FAILED on port %d: %s" % (port, e))
