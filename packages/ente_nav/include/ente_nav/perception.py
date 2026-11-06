"""AprilTag landmark detector + YOLO (ONNX via OpenCV DNN, so no torch needed on the Jetson)."""
import math
import os
import cv2
import numpy as np


class TagDetector:
    def __init__(self, tag_size=0.065, families="tag36h11", decimate=2.0):
        from dt_apriltags import Detector
        self.det = Detector(families=families, nthreads=2, quad_decimate=decimate)
        self.tag_size = tag_size

    def detect(self, gray, K, cam_forward=0.06):
        """Returns list of dicts: id, corners, range, bearing (robot frame: x forward, y left)."""
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        out = []
        for d in self.det.detect(gray, estimate_tag_pose=True,
                                 camera_params=(fx, fy, cx, cy), tag_size=self.tag_size):
            if d.decision_margin < 20:
                continue
            tx, _, tz = d.pose_t.flatten()
            fwd, left = tz + cam_forward, -tx
            out.append({"id": int(d.tag_id), "corners": d.corners.astype(int).tolist(),
                        "range": math.hypot(fwd, left), "bearing": math.atan2(left, fwd)})
        return out


class YoloDetector:
    """ONNX Runtime YOLO. Works with Duckietown's lx-object-detection model (NMS-free [1,300,6]) and with
    standard raw YOLOv8/11 exports ([1,4+nc,N]). Class names are read from the ONNX metadata."""

    def __init__(self, onnx_path, conf=0.5, iou=0.45, names=None, threads=1):
        import ast
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=provs)
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
        self.net_h, self.net_w = int(inp.shape[2]), int(inp.shape[3])
        meta = self.sess.get_modelmeta().custom_metadata_map
        try:
            nm = ast.literal_eval(meta.get("names", "{}"))
            self.names = [nm[i] for i in sorted(nm)]
        except Exception:
            self.names = []
        if names:
            self.names = names
        self.conf, self.iou = conf, iou
        self.tag = os.path.basename(onnx_path)

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        img = bgr if (h, w) == (self.net_h, self.net_w) else cv2.resize(bgr, (self.net_w, self.net_h))
        x = np.transpose(img[:, :, ::-1].astype(self.dtype) / 255.0, (2, 0, 1))[None]
        out = self.sess.run(None, {self.in_name: x})[0][0]
        sx, sy = w / float(self.net_w), h / float(self.net_h)
        res = []
        if out.ndim == 2 and out.shape[1] == 6:            # end-to-end: x1,y1,x2,y2,score,cls
            for x1, y1, x2, y2, sc, c in out:
                if sc >= self.conf:
                    res.append(self._mk(x1 * sx, y1 * sy, x2 * sx, y2 * sy, sc, int(c)))
            return res
        out = out.T if out.shape[0] < out.shape[1] else out   # raw: (N, 4+nc) with cx,cy,w,h
        scores = out[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        out, cls, conf = out[keep], cls[keep], conf[keep]
        if len(out) == 0:
            return []
        boxes = [[float((o[0] - o[2] / 2) * sx), float((o[1] - o[3] / 2) * sy), float(o[2] * sx), float(o[3] * sy)] for o in out]
        for i in np.array(cv2.dnn.NMSBoxes(boxes, conf.tolist(), self.conf, self.iou)).flatten():
            bx, by, bw, bh = boxes[i]
            res.append(self._mk(bx, by, bx + bw, by + bh, conf[i], int(cls[i])))
        return res

    def _mk(self, x1, y1, x2, y2, sc, c):
        name = self.names[c] if c < len(self.names) else str(c)
        return {"cls": name, "conf": float(sc), "box": [int(x1), int(y1), int(x2), int(y2)], "src": self.tag}


def ground_point(u, v, K, cam_height, cam_tilt, cam_forward=0.06):
    """Project a floor pixel to robot-frame (forward, left) metres. None if at/above the horizon."""
    g = GroundProjector._pinhole(np.array([(u - K[2]) / K[0]]), np.array([(v - K[5]) / K[4]]), cam_height, cam_tilt, cam_forward)
    return None if not np.isfinite(g[0, 0]) else (float(g[0, 0]), float(g[0, 1]))


class GroundProjector:
    """Pixel <-> floor point (forward, left) in metres, robot frame.
    Pixels are first undistorted with the camera_info distortion D (set .D from on_info) and normalised with K,
    exactly like dt-core's ground_projection_node. Then the Duckietown extrinsic homography is applied
    (its "resolution-independent" v2 files map NORMALISED rectified coords -> ground). Without a calibration
    file a pinhole model from the measured camera height/tilt is used."""

    def __init__(self, veh, cam_height, cam_tilt, cam_forward):
        self.H = self.Hinv = None
        self.D = None
        self.y_offset = 0.0       # ground_y_offset (m): lateral shift of the extrinsic calibration
        self.scale = 1.0          # ground_scale: corrects a uniformly mis-scaled extrinsic calibration
        self.r_max = 1.0          # undistorted radius beyond which the fisheye polynomial folds back
        self.fb = (cam_height, cam_tilt, cam_forward)
        import glob
        import yaml
        for pat in ("/data/config/calibrations/camera_extrinsic/%s.yaml" % veh,
                    "/data/config/calibrations/camera_extrinsic/default.yaml"):
            for f in glob.glob(pat):
                try:
                    h = yaml.safe_load(open(f)).get("homography")
                    if h and len(h) == 9:
                        self.H = np.array(h, dtype=float).reshape(3, 3)
                        self.Hinv = np.linalg.inv(self.H)
                        self.src = f
                        return
                except Exception:
                    pass
        self.src = "pinhole fallback"

    @staticmethod
    def _Kmat(K):
        return np.array(K, dtype=float).reshape(3, 3)

    def normalize(self, uv, K):
        """(N,2) raw pixels -> (N,2) undistorted normalised coords ((x-cx)/fx, (y-cy)/fy)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if self.D is not None and np.any(np.abs(self.D) > 0):
            return cv2.undistortPoints(uv.reshape(-1, 1, 2), self._Kmat(K), np.asarray(self.D, float)).reshape(-1, 2)
        return np.stack([(uv[:, 0] - K[2]) / K[0], (uv[:, 1] - K[5]) / K[4]], 1)

    @staticmethod
    def _pinhole(xn, yn, h, t, cf):
        den = math.sin(t) + yn * math.cos(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(den > 0.03, h / den, np.nan)
        return np.stack([s * (math.cos(t) - yn * math.sin(t)) + cf, -s * xn], 1)

    def project_many(self, uv, K, img_wh=None):
        """(N,2) raw pixels -> (N,2) (forward, left) metres; rows are NaN at/above the horizon."""
        n = self.normalize(uv, K)
        if self.H is not None:
            g = self.H @ np.vstack([n.T, np.ones(len(n))])
            with np.errstate(divide="ignore", invalid="ignore"):
                out = (g[:2] / g[2]).T
            out[(g[2] <= 1e-6) | (out[:, 0] <= 0)] = np.nan
            out = out * self.scale
        else:
            out = self._pinhole(n[:, 0], n[:, 1], *self.fb) * self.scale
        out[:, 1] += self.y_offset
        return out

    def project(self, u, v, K, img_wh):
        g = self.project_many([[u, v]], K, img_wh)[0]
        return None if not np.isfinite(g[0]) else (float(g[0]), float(g[1]))

    def to_pixel(self, fwd, left, K):
        """Floor points (arrays, metres) -> (N,2) raw (distorted) pixel coords; NaN behind the camera."""
        fwd = np.atleast_1d(np.asarray(fwd, float)) / self.scale
        left = (np.atleast_1d(np.asarray(left, float)) - self.y_offset) / self.scale
        if self.H is not None:
            q = self.Hinv @ np.vstack([fwd, left, np.ones(len(fwd))])
            ok = q[2] > 1e-9
            xn, yn = q[0] / np.where(ok, q[2], 1), q[1] / np.where(ok, q[2], 1)
        else:
            h, t, cf = self.fb
            X = fwd - cf
            zc = X * math.cos(t) + h * math.sin(t)
            ok = zc > 1e-3
            zc = np.where(ok, zc, 1)
            xn, yn = -left / zc, (-X * math.sin(t) + h * math.cos(t)) / zc
        ok &= np.hypot(xn, yn) < self.r_max
        pts = np.stack([xn, yn, np.ones_like(xn)], 1)
        D = np.asarray(self.D, float) if self.D is not None else np.zeros(5)
        px = cv2.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self._Kmat(K), D)[0].reshape(-1, 2)
        px[~ok] = np.nan
        return px
