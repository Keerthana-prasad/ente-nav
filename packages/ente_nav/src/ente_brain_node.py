#!/usr/bin/env python3
"""ente_brain: sensor fusion + EKF-SLAM + AprilTag/YOLO perception + navigation + NLP + web dashboard."""
import difflib
import json
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu, Range
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, WheelsCmdStamped

from ente_nav.ekf_slam import EKFSlam, wrap
from ente_nav.odometry import Odometry
from ente_nav.navigator import Navigator
from ente_nav.nlp import NLP
from ente_nav import dashboard
from ente_nav.perception import TagDetector, YoloDetector, GroundProjector
from ente_nav.lanes import LanePerception
from ente_nav.safety import Safety
from ente_nav.motor import WheelPI
from ente_nav.topo_map import TopoMap, Mission


class EnteBrain:
    def __init__(self):
        rospy.init_node("ente_brain")
        P = lambda k, d=None: rospy.get_param("~" + k, d)
        veh = os.environ.get("VEHICLE_NAME", "entebot209")
        self.cfg = {k: P(k, v) for k, v in dict(
            v_max=0.18, w_max=2.5, k_heading=1.8, turn_in_place=0.7, slow_radius=0.4, goal_tol=0.02,
            stop_dist=0.30, obstacle_wait=4.0, unc_max=0.35, reloc_after=3.0, reloc_timeout=20.0,
            offcourse=0.30, spin_w=1.0).items()}
        self.cam_height, self.cam_tilt = P("cam_height", 0.10), math.radians(P("cam_tilt_deg", 12.0))
        self.cam_forward = P("cam_forward", 0.06)
        self.corridor = P("corridor_half_width", 0.14)
        self.ignore_cls = set(P("ignore_classes", []))
        self.yolo_every = P("yolo_every_n_frames", 3)
        self.max_v_cmd, self.max_w_cmd = P("manual_v", 0.25), P("manual_w", 3.0)
        self.slew_v, self.slew_w = P("slew_v", 0.6), P("slew_w", 6.0)     # per second
        res = P("encoder_resolution", 135)
        self.enc_res = res

        self.log = deque(maxlen=40)
        self.slam = EKFSlam()
        self.vehicle_tags = set(int(i) for i in (P("vehicle_tag_ids", []) or []))   # tags on other Duckiebots
        for tid, xy in (P("known_landmarks", {}) or {}).items():
            self.slam.add_known(int(tid), xy[0], xy[1])
        self.odom = Odometry(P("wheel_radius", 0.0318), P("baseline", 0.102), P("gyro_weight", 0.85))
        self.places = {"home": (0.0, 0.0), "start": (0.0, 0.0)}
        self.topo = TopoMap(P("node_match_r", 0.40), P("node_match_r_edge", 0.80), P("node_match_th", 0.6))
        self.mission = Mission(self.topo, lambda m: self._log(m))
        self.tag_names = {k.lower(): int(v) for k, v in (P("tag_names", {}) or {}).items()}
        self.data_dir = os.environ.get("ENTE_DATA_DIR", "/data" if os.path.isdir("/data") else "/tmp")
        self.map_file = os.path.join(self.data_dir, "ente_map.json")
        self._load_map(P("load_map", True))
        self._map_dirty = False

        self.lock = threading.RLock()
        self.mode = "MANUAL"          # MANUAL | AUTO  (starts manual for safety)
        self.estop = False
        self.man_cmd, self.man_t = (0.0, 0.0), 0.0
        self.cmd_out = (0.0, 0.0)
        self.frame, self.frame_id, self.K, self.img_wh = None, 0, None, None
        self.tags, self.dets = [], []
        self.last_tag_t = 0.0
        self.last_det_t = 0.0
        self.obstacle_dist = None
        self.obstacles = deque(maxlen=150)
        self.trail = deque(maxlen=1500)
        self.history = deque(maxlen=120)
        self.ticks = {"l": None, "r": None}
        self.fps = {"tag": 0.0, "yolo": 0.0}
        self.t_prev = time.time()
        self.odo_s = 0.0                 # odometer (m, signed), ages slow YOLO distances
        self.tag_period = 1.0 / max(0.1, P("tag_hz", 4.0))
        self.tag_times, self.yolo_times = deque(maxlen=20), deque(maxlen=20)

        self.tagdet = TagDetector(P("tag_size", 0.065), decimate=P("tag_decimate", 2.0))
        self.proj = GroundProjector(veh, self.cam_height, self.cam_tilt, self.cam_forward)
        self.proj.scale = P("ground_scale", 1.0)
        self.proj.y_offset = P("ground_y_offset", 0.0)
        self._log("ground projection: %s (scale %.3f)" % (self.proj.src, self.proj.scale))
        self.geom = {k: P(k, d) for k, d in (("lane_width", 0.23), ("white_tape_width", 0.048), ("yellow_tape_width", 0.025))}
        self.lanes = LanePerception(P("lanes", {}), self.proj, self.geom)
        self.nav = Navigator(self.cfg, P("lane_control", {}), self.geom, P("intersection", {}))
        self.nav.hooks = {"stopline": self._on_stopline, "decide": self._on_decide, "turn_failed": self._on_turn_failed,
                          "arrived": self._on_arrived, "lane_reset": self.lanes.reset}
        self.safety = Safety(P("safety", {}), self._log)
        self.lane = self.lanes.state()
        self.lane_t = 0.0
        self.lane_times = deque(maxlen=20)
        self.lane_period = 1.0 / max(1.0, P("lane_hz", 15.0))
        self.D = None
        mdir = os.path.join(os.path.dirname(__file__), "..", "models")
        self.yolos = []
        for spec in P("yolo_models", [{"file": "duckie.onnx", "conf": 0.4}]):
            f = os.path.join(mdir, spec["file"])
            if os.path.exists(f):
                try:
                    self.yolos.append(YoloDetector(f, spec.get("conf", 0.4), names=spec.get("names"), threads=P("yolo_threads", 1)))
                    self._log("YOLO loaded: %s classes=%s" % (spec["file"], self.yolos[-1].names))
                except Exception as e:
                    self._log("YOLO %s failed to load: %s" % (spec["file"], e))
            else:
                self._log("YOLO model missing: %s" % f)
        self.yolo = self.yolos[0] if self.yolos else None
        self.nlp = NLP(None, self._log, url=P("nlp_url", "") or None)

        self.pub = rospy.Publisher(P("car_cmd_topic", "/%s/car_cmd_switch_node/cmd" % veh), Twist2DStamped, queue_size=1)
        # Direct wheel output: used when nothing (kinematics/car_cmd_switch) is listening on car_cmd_topic,
        # e.g. when dt-core runs ONLY this launcher. drive_output: auto | car_cmd | wheels
        self.drive_output = P("drive_output", "auto")
        self.wheels_pub = rospy.Publisher(P("wheels_cmd_topic", "/%s/wheels_driver_node/wheels_cmd" % veh), WheelsCmdStamped, queue_size=1)
        self.kin = self._load_kinematics(veh)
        self._log("wheel kinematics: gain=%.2f trim=%.3f baseline=%.3f radius=%.4f k=%.1f (%s)" % (
            self.kin["gain"], self.kin["trim"], self.kin["baseline"], self.kin["radius"], self.kin["k"], self.kin["src"]))
        self._out_path = None
        # closed-loop wheel speeds (encoders): the motors have a large deadband at lane-following speeds
        self.wheel_pi = WheelPI(P("motor", {}), self.kin, self.enc_res, P("wheel_radius", 0.0318), self._log) if P("wheel_pi", True) else None
        self._pub_dt = 1 / 30.0
        rospy.Subscriber("/%s/camera_node/image/compressed" % veh, CompressedImage, self.on_img, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber("/%s/camera_node/camera_info" % veh, CameraInfo, self.on_info, queue_size=1)
        # encoder topics differ between duckiebot-interface versions (left_wheel_encoder_node vs left_wheel_encoder_driver_node)
        lt = P("left_tick_topic", "/%s/left_wheel_encoder_driver_node/tick" % veh)
        rt = P("right_tick_topic", "/%s/right_wheel_encoder_driver_node/tick" % veh)
        rospy.Subscriber(lt, WheelEncoderStamped, lambda m: self._on_tick("l", m.data), queue_size=5)
        rospy.Subscriber(rt, WheelEncoderStamped, lambda m: self._on_tick("r", m.data), queue_size=5)
        self._log("encoder ticks <- %s , %s" % (lt, rt))
        imu_topic = P("imu_topic", "/%s/imu_node/raw" % veh)
        self._log("gyro <- %s" % imu_topic)
        rospy.Subscriber(imu_topic, Imu, lambda m: self.odom.on_imu(m.angular_velocity.z, m.header.stamp.to_sec()), queue_size=5)

        tof = P("tof_topic", "/%s/front_center_tof_driver_node/range" % veh)
        try:
            have_tof = any(t == tof for t, _ in rospy.get_published_topics())
        except Exception:
            have_tof = False
        if have_tof:
            rospy.Subscriber(tof, Range, lambda m: self.safety.set_tof(m.range, time.time(), m.max_range, m.min_range), queue_size=1)
            self._log("front ToF <- %s" % tof)
        else:
            self._log("front ToF topic %s not found: obstacle safety uses YOLO only" % tof)
        threading.Thread(target=self._perception_loop, daemon=True).start()
        threading.Thread(target=self._lane_loop, daemon=True).start()
        if self.yolos:
            threading.Thread(target=self._yolo_loop, daemon=True).start()
        threading.Thread(target=dashboard.serve, args=(self, P("port", 8090)), daemon=True).start()
        rospy.Timer(rospy.Duration(1 / 30.0), self.control_loop)
        rospy.on_shutdown(self.shutdown)
        self._log("ente_brain ready. Dashboard on port %d. Starting in MANUAL mode." % P("port", 8090))

    # ---------- helpers ----------
    def _log(self, s):
        rospy.loginfo(s)
        self.log.appendleft("%s  %s" % (time.strftime("%H:%M:%S"), s))

    def _load_map(self, load_landmarks):
        """Restores saved places, tag names and (optionally) landmarks. Coordinates are in the frame whose
        origin is the robot's START pose, so park the robot on the same start mark every run."""
        try:
            m = json.load(open(self.map_file))
        except Exception:
            return
        self.places.update({k: tuple(v) for k, v in m.get("places", {}).items()})
        self.tag_names.update({k: int(v) for k, v in m.get("tag_names", {}).items()})
        self.topo.load(m.get("topo"))
        self.mission.places = {k: v for k, v in m.get("place_topo", {}).items() if not v.get("pending")}
        if self.topo.nodes:
            self._log("loaded track graph: %d intersections, %d connections" % (len(self.topo.nodes), len(self.topo.edges)))
        if load_landmarks:
            for tid, xy in m.get("landmarks", {}).items():
                if int(tid) not in self.slam.ids:
                    self.slam.add_known(int(tid), xy[0], xy[1], var=0.04)
        self._log("loaded map: %d landmarks, %d places" % (len(m.get("landmarks", {})), len(m.get("places", {}))))

    def save_map(self):
        try:
            with self.lock:
                lms = {str(i): list(p) for i, p in self.slam.landmarks().items()}
            json.dump({"landmarks": lms,
                       "places": {k: list(v) for k, v in self.places.items() if k not in ("home", "start")},
                       "tag_names": self.tag_names, "topo": self.topo.to_json(),
                       "place_topo": {k: v for k, v in self.mission.places.items() if not v.get("pending")}},
                      open(self.map_file, "w"), indent=1)
            self._map_dirty = False
            return True
        except Exception as e:
            self._log("could not save map: %s" % e)
            return False

    @staticmethod
    def _load_kinematics(veh):
        kin = {"gain": 1.0, "trim": 0.0, "baseline": 0.1, "radius": 0.0318, "k": 27.0, "limit": 1.0, "src": "defaults"}
        import yaml
        for f in ("/data/config/calibrations/kinematics/%s.yaml" % veh, "/data/config/calibrations/kinematics/default.yaml"):
            try:
                d = yaml.safe_load(open(f)) or {}
                for k in ("gain", "trim", "baseline", "radius", "k", "limit"):
                    if k in d:
                        kin[k] = float(d[k])
                kin["src"] = f
                break
            except Exception:
                pass
        return kin

    def _on_tick(self, side, data):
        self.ticks[side] = data
        if self.wheel_pi:
            self.wheel_pi.update_ticks(side, data, time.time())

    def _publish_cmd(self, v, w):
        """Send (v, omega) to the robot. With wheel_pi: closed-loop wheel speeds straight to the wheel driver.
        Otherwise car_cmd if a kinematics node listens, else open-loop wheel duty."""
        if self.wheel_pi is not None and self.ticks["l"] is not None:
            k = self.kin
            s_r, s_l = v + 0.5 * w * k["baseline"], v - 0.5 * w * k["baseline"]
            u_l, u_r = self.wheel_pi.step(s_l, s_r, time.time(), self._pub_dt)
            if self._out_path != "wheels_pi":
                self._out_path = "wheels_pi"
                self._log("motor output -> %s (closed-loop wheel speeds)" % self.wheels_pub.resolved_name)
            m = WheelsCmdStamped()
            m.header.stamp = rospy.Time.now()
            m.vel_left, m.vel_right = u_l, u_r
            self.wheels_pub.publish(m)
            return
        use_car = self.drive_output == "car_cmd" or (self.drive_output == "auto" and self.pub.get_num_connections() > 0)
        path = "car_cmd" if use_car else "wheels"
        if path != self._out_path:
            self._out_path = path
            self._log("motor output -> %s" % (self.pub.resolved_name if use_car else self.wheels_pub.resolved_name))
        if use_car:
            m = Twist2DStamped()
            m.header.stamp = rospy.Time.now()
            m.v, m.omega = v, w
            self.pub.publish(m)
            return
        k = self.kin
        om_r = (v + 0.5 * w * k["baseline"]) / k["radius"]
        om_l = (v - 0.5 * w * k["baseline"]) / k["radius"]
        u_r = om_r * (k["gain"] + k["trim"]) / k["k"]
        u_l = om_l * (k["gain"] - k["trim"]) / k["k"]
        lim = k["limit"]
        m = WheelsCmdStamped()
        m.header.stamp = rospy.Time.now()
        m.vel_left, m.vel_right = max(-lim, min(lim, u_l)), max(-lim, min(lim, u_r))
        self.wheels_pub.publish(m)

    def on_info(self, m):
        self.K = list(m.K)
        self.info_wh = (m.width, m.height)
        if self.D is None and len(m.D) >= 4:
            self.D = list(m.D)
            self.proj.D = self.D

    def on_img(self, m):
        img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            self.frame, self.frame_id = img, self.frame_id + 1
            self.img_wh = (img.shape[1], img.shape[0])

    def scaled_K(self):
        if self.K is None or self.img_wh is None:
            return None
        s = self.img_wh[0] / float(self.info_wh[0] or self.img_wh[0])
        K = list(self.K)
        for i in (0, 2, 4, 5):
            K[i] *= s
        return K

    # ---------- perception thread ----------
    def _perception_loop(self):
        last_id, n = -1, 0
        while not rospy.is_shutdown():
            K, frame = self.scaled_K(), self.frame
            if frame is None or K is None or self.frame_id == last_id:
                time.sleep(0.01)
                continue
            if time.time() - getattr(self, "_tag_t", 0.0) < self.tag_period:
                time.sleep(0.01)
                continue
            self._tag_t = time.time()
            last_id, n = self.frame_id, n + 1
            tags = self.tagdet.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), K, self.cam_forward)
            now = time.time()
            with self.lock:
                self.tags = tags
                for t in tags:
                    if t["id"] in self.vehicle_tags:
                        continue                       # moving robots are obstacles, never landmarks
                    if t["range"] < 1.5:
                        before = len(self.slam.ids)
                        self.slam.update(t["id"], t["range"], t["bearing"])
                        self._map_dirty |= len(self.slam.ids) != before
                if tags:
                    self.last_tag_t = now
            self.tag_times.append(now)
            if not self.yolos:
                self._process_dets([], K, tags)

    def _yolo_loop(self):
        """YOLO on the newest frame, as fast as the CPU allows (its own thread so tags/lanes never delay it)."""
        last_id = -1
        while not rospy.is_shutdown():
            K, frame = self.scaled_K(), self.frame
            if frame is None or K is None or self.frame_id == last_id:
                time.sleep(0.01)
                continue
            last_id = self.frame_id
            t_cap, s_cap = time.time(), self.odo_s
            try:
                dets = [d for y in self.yolos for d in y.detect(frame)]
            except Exception as e:
                self._log("YOLO error: %s" % e)
                time.sleep(0.5)
                continue
            self._process_dets(dets, K, list(self.tags), t_cap, s_cap)
            self.yolo_times.append(time.time())

    def _lane_loop(self):
        """Lane perception on every new frame, independent of the slower tag/YOLO thread."""
        last_id = -1
        while not rospy.is_shutdown():
            K, frame = self.scaled_K(), self.frame
            if frame is None or K is None or self.frame_id == last_id:
                time.sleep(0.005)
                continue
            if time.time() - getattr(self, "_lane_start", 0.0) < self.lane_period:
                time.sleep(0.005)
                continue
            self._lane_start = time.time()
            last_id = self.frame_id
            try:
                st = self.lanes.process(frame, K, self.img_wh, self.odom.vs, self.odom.ws)
            except Exception as e:
                self._log("lane perception error: %s" % e)
                time.sleep(0.5)
                continue
            self.lane, self.lane_t = st, time.time()
            self.lane_times.append(self.lane_t)

    def _process_dets(self, dets, K, tags, t_cap=None, s_cap=None):
        nearest = None
        with self.lock:
            x, y, th = self.slam.pose
        wh = self.img_wh or (640, 480)
        cand = []
        for d in dets:
            if d["cls"] in self.ignore_cls:
                continue
            u, v = (d["box"][0] + d["box"][2]) / 2, d["box"][3]
            gp = self.proj.project(u, v, K, wh)
            if gp is None:
                continue
            d["dist"] = gp[0]
            cand.append(gp)
        if self.yolos:
            self.safety.set_yolo(list(cand), t_cap or time.time(), self.lane if self.lane.get("conf", 0) >= 0.35 else None,
                                 self.geom["lane_width"], self.corridor, s_cap)
        for t in tags:                                  # other Duckiebots seen via their back tag
            if t["id"] in self.vehicle_tags:
                cand.append((t["range"] * math.cos(t["bearing"]), t["range"] * math.sin(t["bearing"])))
        for fwd, left in cand:
            if abs(left) < self.corridor and fwd > 0:
                nearest = fwd if nearest is None else min(nearest, fwd)
            self.obstacles.append((x + fwd * math.cos(th) - left * math.sin(th),
                                   y + fwd * math.sin(th) + left * math.cos(th), time.time()))
        self.dets, self.last_det_t, self.obstacle_dist = dets, time.time(), nearest

    # ---------- control loop ----------
    def control_loop(self, _):
        now = time.time()
        dt, self.t_prev = now - self.t_prev, now
        if self.ticks["l"] is not None and self.ticks["r"] is not None:
            step = self.odom.step(self.ticks["l"], self.ticks["r"], self.enc_res, dt)
            if step:
                with self.lock:
                    self.slam.predict(*step)
                self.mission.edge_dist += step[0]
                self.odo_s += step[0]
        with self.lock:
            pose = self.slam.pose
            unc = self.slam.uncertainty()
        if now - self.last_det_t > 1.0:
            self.obstacle_dist = None
        tag_age = now - self.last_tag_t if self.last_tag_t else 999.0

        v = w = 0.0
        if self.estop:
            pass
        elif self.mode == "MANUAL":
            if now - self.man_t < 0.4:          # dead-man: stops if the browser stops sending
                v, w = self.man_cmd
        else:
            lane = self.lane if now - self.lane_t < 0.5 else None
            v, w = self.nav.step(pose, now, self.obstacle_dist, unc, tag_age, lane, dt, self.mission.edge_dist)
            if self.nav.lost:
                self.mode = "MANUAL"
                self.mission.stop()
                self._log(self.nav.alert)
                self.nav.lost = False
        v_cap, hard = self.safety.update(now, self.odo_s)
        if self.mode == "AUTO" and not self.estop:
            if hard:
                spinning = self.nav.cur is not None and self.nav.cur.get("t") == "spin" or self.nav.lc.state == "REACQUIRE"
                v, w = 0.0, (w if spinning else 0.0)
                self.cmd_out = (0.0, self.cmd_out[1])            # brake immediately, no slew on a safety stop
            elif v_cap is not None and v > v_cap:
                v = v_cap

        # slew-rate limit -> smooth motion, less wheel slip, less drift
        dv, dw = self.slew_v * dt, self.slew_w * dt
        ov, ow = self.cmd_out
        ov += max(-dv, min(dv, v - ov))
        ow += max(-dw, min(dw, w - ow))
        self.cmd_out = (0.0, 0.0) if self.estop else (ov, ow)
        self._pub_dt = dt
        self._publish_cmd(*self.cmd_out)

        if self._map_dirty and int(now) % 30 == 0 and int(now * 10) % 10 == 0:
            self.save_map()
        if int(now * 10) % 3 == 0:
            self.trail.append((round(pose[0], 3), round(pose[1], 3)))
        if int(now * 2) != getattr(self, "_h", -1):
            self._h = int(now * 2)
            self.history.append({"v": round(self.odom.v, 3), "u": round(unc, 3)})

    # ---------- mission hooks (called from the navigator inside the control loop) ----------
    def _on_stopline(self, opened):
        with self.lock:
            x, y, th = self.slam.pose
        phi = self.lane.get("phi", 0.0) if self.lane.get("conf", 0) >= 0.35 else 0.0
        lane_th = th - phi                                   # heading of the lane at this stop line
        n, known = self.mission.identify(x, y, lane_th)
        if known is not None:                                # recognised: remove odometry drift
            with self.lock:
                self.slam.reset_pose(known[0], known[1], known[2] + phi)
            self._log("at intersection node %d: odometry corrected by %.2f m" % (n, math.hypot(known[0] - x, known[1] - y)))
        if opened:
            self.topo.observe_exits(n, opened)
        res = self.mission.decide(n)
        self._log("node %d: %s" % (n, ("turn " + res["turn"]) if "turn" in res else res.get("done", "")))
        self.save_map()
        return res

    def _on_decide(self):
        n = self.mission.at_node
        if n is None:                                        # not at a node: just drive to the next stop line
            return {"continue": True, "stop_s": self.mission.stop_s_now()}
        return self.mission.decide(n)

    def _on_turn_failed(self):
        self.mission.turn_failed()
        self.save_map()

    def _on_arrived(self, msg):
        if self.mission.mode == "goto" and not msg:
            msg = "Arrived at %s." % self.mission.goal_name
        self.mission.stop()
        if msg:
            self._log(msg)

    def _start_lane_mission(self, mode, name="", place=None):
        r = self.mission.start(mode, name, place)
        if r and r.get("stop_s") is not None:
            self.nav.lane_follow(stop_s=r["stop_s"])
        else:
            self.nav.lane_follow(decide_first=self.mission.at_node is not None)
        self.mode, self.estop = "AUTO", False

    # ---------- NLP / commands ----------
    def resolve_place(self, name):
        n = name.lower().strip()
        m = n.replace("tag", "").strip()
        with self.lock:
            lms = self.slam.landmarks()
        if n in self.places:
            return self.places[n]
        if n in self.tag_names and self.tag_names[n] in lms:
            return lms[self.tag_names[n]]
        if m.isdigit() and int(m) in lms:
            return lms[int(m)]
        pool = list(self.places) + [k for k in self.tag_names if self.tag_names[k] in lms]
        close = difflib.get_close_matches(n, pool, n=1, cutoff=0.6)
        if close:
            return self.resolve_place(close[0])
        return None

    def run_intent(self, it):
        k = it["intent"]
        with self.lock:
            pose = self.slam.pose
        if k == "lane_follow":
            self._start_lane_mission("follow")
            return "Following the lane. I'll stop 1 s at every red line."
        if k == "map_track":
            self._start_lane_mission("map")
            return "Mapping the track: I'll try every turn at every intersection (%d known so far)." % len(self.topo.nodes)
        if k == "goto" and it["place"].lower().strip() in self.mission.places:
            nm = it["place"].lower().strip()
            pl = self.mission.places[nm]
            if pl.get("pending"):
                return "I haven't reached a red line since saving '%s', so I don't know its road yet. Drive on (or say 'map the track') first." % nm
            self._start_lane_mission("goto", nm, pl)
            return "Driving to %s along the lanes." % nm
        if k in ("goto", "goto_xy", "forward", "turn", "explore"):
            self.mission.stop()                              # metric motion ends any lane mission
        if k in ("goto", "goto_xy"):
            if k == "goto":
                xy = self.resolve_place(it["place"])
                if xy is None:
                    return "I don't know '%s' yet. Say 'explore' so I can find it, or save it first." % it["place"]
                name = it["place"]
            else:
                xy, name = (it["x"], it["y"]), "(%.1f, %.1f)" % (it["x"], it["y"])
            self.nav.goto(xy[0], xy[1], name)
            self.mode, self.estop = "AUTO", False
            return "Heading to %s" % name
        if k == "forward":
            self.nav.forward(float(it["d"]), pose)
            self.mode, self.estop = "AUTO", False
            return "Driving forward %.2f m" % it["d"]
        if k == "turn":
            self.nav.spin(math.radians(it["deg"]))
            self.mode, self.estop = "AUTO", False
            return "Turning %.0f deg" % it["deg"]
        if k == "explore":
            self.nav.explore()
            self.mode, self.estop = "AUTO", False
            return "Exploring and mapping"
        if k == "stop":
            self.nav.cancel()
            self.mission.stop()
            return "Stopped"
        if k == "manual":
            self.nav.cancel()
            self.mission.stop()
            self.mode = "MANUAL"
            return "Manual mode. You have control."
        if k == "save":
            nm = it["name"].lower().strip()
            self.places[nm] = (pose[0], pose[1])
            pl = self.mission.place_here()
            self.mission.places[nm] = pl
            if pl.get("pending"):
                self.mission.pending_places[nm] = pl["s"]
                where = "it gets a road once I reach the next red line"
            elif "to" in pl:
                where = "at intersection %d" % pl["to"]
            else:
                where = "%.2f m after the %s turn at intersection %d" % (pl["s"], pl["turn"], pl["from"])
            self.save_map()
            return "Marked this spot as '%s' (%s). Say 'go to %s' any time." % (it["name"], where, it["name"])
        if k == "name_tag":
            return self.name_visible_tag(it["name"])
        if k == "speed":
            self.nav.speed_scale = {"slow": 0.5, "medium": 0.8, "fast": 1.0}.get(it["level"], 0.8)
            return "Speed: %s" % it["level"]
        return "Unknown command"

    def name_visible_tag(self, name, tag_id=None):
        vis = [t for t in self.tags if time.time() - self.last_tag_t < 1.0 and t["id"] not in self.vehicle_tags]
        if tag_id is None:
            if not vis:
                return "I can't see a tag right now. Point the camera at one first."
            tag_id = min(vis, key=lambda t: t["range"])["id"]
        with self.lock:
            known = tag_id in self.slam.ids
        self.tag_names[name.lower()] = int(tag_id)
        self.save_map()
        return "Tag %d is now '%s'%s" % (tag_id, name, "" if known else " (mapping it as I drive closer)")

    def api(self, c):
        t = c.get("type")
        if t == "mark":
            return {"ok": True, "reply": self.run_intent({"intent": "save", "name": c.get("name", "spot")})}
        if t == "name_tag":
            return {"ok": True, "reply": self.name_visible_tag(c.get("name", "tag"), c.get("id"))}
        if t == "save_map":
            return {"ok": self.save_map(), "reply": "Map saved"}
        if t == "manual":
            self.man_cmd = (max(-1, min(1, c.get("v", 0))) * self.max_v_cmd, max(-1, min(1, c.get("w", 0))) * self.max_w_cmd)
            self.man_t = time.time()
            return {"ok": True}
        if t == "mode":
            if c["mode"] == "MANUAL":
                self.nav.cancel()
                self.mission.stop()
            self.mode, self.estop = c["mode"], False
            return {"ok": True}
        if t == "estop":
            self.estop = bool(c.get("on", True))
            if not self.estop and self.wheel_pi:
                self.wheel_pi.reset_stall()
            if self.estop:
                self.nav.cancel()
                self.mission.stop()
                self.mode = "MANUAL"
            return {"ok": True}
        if t == "nlp":
            text = c.get("text", "")
            intents = self.nlp.parse(text)
            reply = " / ".join(self.run_intent(i) for i in intents) if intents else "Sorry, I didn't understand that."
            self._log("> %s  =>  %s" % (text, reply))
            return {"ok": True, "reply": reply, "intents": intents}
        if t == "goto_xy":
            return {"ok": True, "reply": self.run_intent({"intent": "goto_xy", "x": c["x"], "y": c["y"]})}
        if t == "goto_place":
            return {"ok": True, "reply": self.run_intent({"intent": "goto", "place": c.get("name", "")})}
        if t == "reset_map":
            with self.lock:
                self.slam = EKFSlam()
            self.trail.clear()
            self.obstacles.clear()
            self.nav.cancel()
            self.mission.stop()
            self.topo.__init__(self.topo.match_r, self.topo.match_r_edge, self.topo.match_th)
            self.mission.places.clear()
            self.mission.last_node = self.mission.last_turn = self.mission.at_node = None
            self.mission.edge_dist = 0.0
            self.save_map()
            self._log("map reset: track graph and road positions of places cleared")
            return {"ok": True}
        return {"ok": False}

    # ---------- dashboard data ----------
    @staticmethod
    def _fps(q):
        return round((len(q) - 1) / (q[-1] - q[0]), 1) if len(q) > 2 and q[-1] > q[0] and time.time() - q[-1] < 3 else 0.0

    def state(self):
        now = time.time()
        with self.lock:
            x, y, th = self.slam.pose
            a, b, ang = self.slam.cov_ellipse()
            lms = self.slam.landmarks()
            unc, acc, rej = self.slam.uncertainty(), self.slam.accepted, self.slam.rejected
        names = {v: k for k, v in self.tag_names.items()}
        try:
            temp = round(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0, 1)
        except Exception:
            temp = None
        return {
            "mode": self.mode, "estop": self.estop, "nav_state": self.nav.state, "goal": self.nav.goal_name,
            "alert": self.nav.alert,
            "pose": {"x": x, "y": y, "th": th}, "ellipse": [a, b, ang],
            "v": self.odom.v, "w": self.odom.w, "cmd_v": self.cmd_out[0], "cmd_w": self.cmd_out[1],
            "unc": unc, "xte": self.nav.xte, "dist_goal": self.nav.dist_to_goal,
            "obstacle": self.obstacle_dist,
            "landmarks": [{"id": i, "x": p[0], "y": p[1], "name": names.get(i, "")} for i, p in lms.items()],
            "trail": list(self.trail)[::2],
            "obstacles": [{"x": o[0], "y": o[1]} for o in self.obstacles if now - o[2] < 30],
            "places": {k: list(v) for k, v in self.places.items()},
            "tags_visible": [t["id"] for t in self.tags if now - self.last_tag_t < 0.5],
            "tag_view": [{"id": t["id"], "range": round(t["range"], 2), "vehicle": t["id"] in self.vehicle_tags} for t in self.tags if now - self.last_tag_t < 0.5],
            "detections": [{"cls": d["cls"], "conf": round(d["conf"], 2), "dist": round(d.get("dist", -1), 2)} for d in self.dets] if now - self.last_det_t < 1.5 else [],
            "slam": {"accepted": acc, "rejected": rej, "gyro": self.odom.have_gyro and self.odom._calibrated},
            "fps": {"tag": self._fps(self.tag_times), "yolo": self._fps(self.yolo_times), "lane": self._fps(self.lane_times)},
            "lane": dict(self.lane, age=round(now - self.lane_t, 2) if self.lane_t else None, width=self.geom["lane_width"],
                         state=self.nav.lane_state or self.nav.lc.state),
            "lane_state": self.nav.lane_state, "nav_message": self.nav.message,
            "mission": self.mission.view(), "next_turn": self.mission.next_turn,
            "route": self.mission.view()["route"], "graph": self.topo.view(),
            "place_topo": self.mission.places,
            "safety": self.safety.state(),
            "ticks": dict(self.ticks), "motor_out": self._out_path,
            "motor": None if not self.wheel_pi else {"l": round(self.wheel_pi.speed["l"], 3), "r": round(self.wheel_pi.speed["r"], 3), "stalled": self.wheel_pi.stalled},
            "odom_dbg": {"th_gyro": round(self.odom.th_gyro, 3), "th_enc": round(self.odom.th_enc, 3), "vs": round(self.odom.vs, 3), "ws": round(self.odom.ws, 3)},
            "sys": {"temp": temp, "yolo_on": bool(self.yolos), "needle": self.nlp.agent is not None},
            "history": list(self.history), "log": list(self.log),
        }

    def render_jpeg(self, layers):
        f = self.frame
        if f is None:
            return None
        img = f.copy()
        if "tags" in layers:
            for t in self.tags:
                pts = np.array(t["corners"], np.int32)
                col = (60, 60, 255) if t["id"] in self.vehicle_tags else (0, 220, 255)
                nm = {v: k for k, v in self.tag_names.items()}.get(t["id"], "")
                cv2.polylines(img, [pts], True, col, 2)
                cv2.putText(img, ("robot " if t["id"] in self.vehicle_tags else "tag ") + "%d %s %.2fm" % (t["id"], nm, t["range"]), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if "yolo" in layers:
            for d in self.dets:
                x0, y0, x1, y1 = d["box"]
                cv2.rectangle(img, (x0, y0), (x1, y1), (80, 200, 80), 2)
                cv2.putText(img, "%s %.2f" % (d["cls"], d["conf"]), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 2)
        if "lanes" in layers and self.K:
            self.lanes.draw(img, self.scaled_K())
        if "corridor" in layers and self.K:
            K = self.scaled_K()
            dist = np.linspace(0.12, 1.0, 10)
            for sgn in (-1, 1):
                px = self.proj.to_pixel(dist, np.full_like(dist, sgn * self.corridor), K)
                px = px[np.isfinite(px[:, 0])]
                if len(px) > 1:
                    cv2.polylines(img, [px.astype(np.int32)], False, (255, 160, 0), 1)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpg.tobytes() if ok else None

    def snapshot_jpeg(self):
        f = self.frame
        if f is None:
            return None
        ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return jpg.tobytes() if ok else None

    def shutdown(self):
        self.save_map()
        for _ in range(3):
            self._publish_cmd(0.0, 0.0)


if __name__ == "__main__":
    EnteBrain()
    rospy.spin()
