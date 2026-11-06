"""Topological track map learned while driving, plus the mission logic that uses it.

Nodes are red stop lines, identified by odometry pose (x, y, heading) and by the edge we arrived on.
Edges are (node, turn) -> next node with the driven lane distance. Places are stored on edges:
    from-form: {"from": A, "turn": t, "s": metres driven since leaving A}
    to-form:   {"to": B, "rem": metres still to drive before reaching B's stop line}
Nothing about the layout is assumed; rearranging the mat just means mapping again ("reset map").
"""
import heapq
import math

TURNS = ("left", "straight", "right")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def key(n, t):
    return "%s:%s" % (n, t)


class TopoMap:
    def __init__(self, match_r=0.40, match_r_edge=0.80, match_th=0.6):
        self.match_r, self.match_r_edge, self.match_th = match_r, match_r_edge, match_th
        self.nodes = {}        # id -> {"x","y","th","visits","exits": {turn: True|False|None}}
        self.edges = {}        # "A:turn" -> {"to": B, "dist": m}
        self.blocked = set()   # "A:turn" known not to exist
        self.next_id = 1

    # ---------- building ----------
    def match(self, x, y, th, from_key=None):
        """Existing node for a stop line seen at pose (x, y, th), or None. Arriving over a known edge allows
        a larger position error (odometry drift), but the heading must still agree."""
        if from_key in self.edges:
            n = self.edges[from_key]["to"]
            nd = self.nodes.get(n)
            if nd and math.hypot(nd["x"] - x, nd["y"] - y) < self.match_r_edge and abs(wrap(nd["th"] - th)) < self.match_th:
                return n
        best, bd = None, self.match_r
        for n, nd in self.nodes.items():
            dist = math.hypot(nd["x"] - x, nd["y"] - y)
            if dist < bd and abs(wrap(nd["th"] - th)) < self.match_th:
                best, bd = n, dist
        return best

    def add_node(self, x, y, th):
        n = self.next_id
        self.next_id += 1
        self.nodes[n] = {"x": x, "y": y, "th": th, "visits": 0, "exits": {t: None for t in TURNS}}
        return n

    def observe_exits(self, n, open_by_turn):
        ex = self.nodes[n]["exits"]
        for t, ok in open_by_turn.items():
            if ok is None:
                continue
            if key(n, t) in self.edges:          # driven it: certainly open
                ex[t] = True
            elif ex[t] is None:
                ex[t] = bool(ok)
            else:                                # later observations can overturn earlier ones
                ex[t] = bool(ok) if ok == ex[t] else ex[t] if self.nodes[n]["visits"] > 2 else bool(ok)

    def set_edge(self, a, t, b, dist):
        self.edges[key(a, t)] = {"to": b, "dist": round(float(dist), 3)}
        self.blocked.discard(key(a, t))
        self.nodes[a]["exits"][t] = True

    def mark_blocked(self, a, t):
        if a in self.nodes:
            self.blocked.add(key(a, t))
            self.nodes[a]["exits"][t] = False
            self.edges.pop(key(a, t), None)

    def open_turns(self, n):
        ex = self.nodes[n]["exits"]
        return [t for t in TURNS if ex.get(t) is not False and key(n, t) not in self.blocked]

    def unexplored(self, n):
        return [t for t in self.open_turns(n) if key(n, t) not in self.edges]

    # ---------- planning ----------
    def dijkstra(self, src, turn_cost=0.05):
        dist, prev, pq = {src: 0.0}, {}, [(0.0, src)]
        while pq:
            dcur, n = heapq.heappop(pq)
            if dcur > dist.get(n, 1e18):
                continue
            for t in TURNS:
                e = self.edges.get(key(n, t))
                if not e:
                    continue
                nd = dcur + e["dist"] + (turn_cost if t != "straight" else 0.0)
                if nd < dist.get(e["to"], 1e18):
                    dist[e["to"]], prev[e["to"]] = nd, (n, t)
                    heapq.heappush(pq, (nd, e["to"]))
        return dist, prev

    def path(self, src, dst):
        """List of (node, turn) from src to dst over known edges, or None."""
        if src == dst:
            return []
        dist, prev = self.dijkstra(src)
        if dst not in dist:
            return None
        out, n = [], dst
        while n != src:
            p, t = prev[n]
            out.append((p, t))
            n = p
        return out[::-1]

    def route_to_place(self, src, place):
        """Best plan from node src to a place: {"steps": [(node, turn), ...], "stop_s": s or None, "cost": m}.
        The last step's turn leads onto the place's edge; stop_s is how far to drive along it (None = stop at
        the next stop line, for places that sit at a stop line)."""
        dist, prev = self.dijkstra(src)
        best = None

        def steps_to(n):
            return self.path(src, n)

        if place.get("from") in self.nodes and place.get("turn"):
            a = place["from"]
            if a in dist and key(a, place["turn"]) not in self.blocked:
                c = dist[a] + place["s"]
                best = {"steps": steps_to(a) + [(a, place["turn"])], "stop_s": place["s"], "cost": c}
        b = place.get("to")
        if b in self.nodes and place.get("rem") is not None:
            rem = place["rem"]
            if rem < 0.10:                                       # the place is (at) stop line b
                if b in dist and (best is None or dist[b] < best["cost"]):
                    best = {"steps": steps_to(b), "stop_s": None, "cost": dist[b], "at_node": b}
            for k, e in self.edges.items():
                if e["to"] != b or e["dist"] < rem - 0.05:
                    continue
                a, t = k.rsplit(":", 1)
                a = int(a)
                if a not in dist:
                    continue
                c = dist[a] + e["dist"] - rem
                if best is None or c < best["cost"]:
                    best = {"steps": steps_to(a) + [(a, t)], "stop_s": max(0.0, e["dist"] - rem), "cost": c}
        return best

    def toward_unexplored(self, src, order=("right", "straight", "left")):
        """Turn to take at src to make progress on exploration, or None when everything reachable is known."""
        here = self.unexplored(src)
        for t in order:
            if t in here:
                return t
        dist, prev = self.dijkstra(src)
        cands = [(d, n) for n, d in dist.items() if n != src and self.unexplored(n)]
        if not cands:
            return None
        _, goal = min(cands)
        p = self.path(src, goal)
        return p[0][1] if p else None

    def complete(self):
        return all(not self.unexplored(n) for n in self.nodes) and bool(self.nodes)

    # ---------- persistence / display ----------
    def to_json(self):
        return {"nodes": {str(k): v for k, v in self.nodes.items()}, "edges": self.edges,
                "blocked": sorted(self.blocked), "next_id": self.next_id}

    def load(self, d):
        if not d:
            return
        self.nodes = {int(k): v for k, v in d.get("nodes", {}).items()}
        self.edges = dict(d.get("edges", {}))
        self.blocked = set(d.get("blocked", []))
        self.next_id = max([d.get("next_id", 1)] + [k + 1 for k in self.nodes])

    def view(self):
        return {"nodes": [{"id": n, "x": round(v["x"], 3), "y": round(v["y"], 3), "th": round(v["th"], 3),
                           "exits": v["exits"], "unexplored": self.unexplored(n)} for n, v in self.nodes.items()],
                "edges": [{"from": int(k.rsplit(":", 1)[0]), "turn": k.rsplit(":", 1)[1], "to": e["to"], "dist": e["dist"]}
                          for k, e in self.edges.items()]}


class Mission:
    """What to do at each stop line: follow the lane, map the track, or drive a route to a place."""

    def __init__(self, topo, log=print):
        self.topo = topo
        self.log = log
        self.mode = "idle"            # idle | follow | map | goto
        self.last_node = None         # node we left most recently
        self.last_turn = None         # turn taken there (the edge we are on)
        self.at_node = None           # stopped at this node and not yet left it
        self.edge_dist = 0.0          # metres driven since leaving last_node (since start if none)
        self.goal_name, self.goal = "", None
        self.route = []               # planned [(node, turn), ...] still ahead
        self.next_turn = None
        self.pending_places = {}      # name -> s since start, saved before the first stop line
        self.places = {}              # name -> place dict (see module doc)
        self.message = ""

    # ---------- commands ----------
    def start(self, mode, name="", place=None):
        self.mode, self.goal_name, self.goal = mode, name, place
        self.route, self.next_turn = [], None
        if mode == "goto":
            return self._replan_here()
        return None

    def stop(self):
        self.mode, self.route, self.next_turn = "idle", [], None

    def place_here(self):
        if self.at_node is not None:
            return {"to": self.at_node, "rem": 0.0}
        if self.last_node is not None:
            return {"from": self.last_node, "turn": self.last_turn, "s": round(self.edge_dist, 3)}
        return {"pending": True, "s": round(self.edge_dist, 3)}

    def stop_s_now(self):
        """If the goal lies ahead on the edge we are driving, how far along it to stop (else None)."""
        g = self.goal
        if self.mode != "goto" or not g or self.at_node is not None:
            return None
        if g.get("from") == self.last_node and g.get("turn") == self.last_turn and self.last_node is not None:
            if g["s"] >= self.edge_dist - 0.02:
                return g["s"]
        return None

    def _replan_here(self):
        """Called when a goto starts. Returns {"stop_s": s} if the goal is ahead on the current edge."""
        s = self.stop_s_now()
        if s is not None:
            self.route = []
            self.next_turn = None
            return {"stop_s": s}
        if self.at_node is not None:
            plan = self.topo.route_to_place(self.at_node, self.goal)
            self.route = plan["steps"] if plan else []
            self.next_turn = self.route[0][1] if self.route else None
        return None

    # ---------- events ----------
    def identify(self, x, y, th):
        """Stop line reached at pose (x, y, th). Returns (node_id, known_node_pose or None)."""
        tp = self.topo
        fk = key(self.last_node, self.last_turn) if self.last_node is not None else None
        n = tp.match(x, y, th, fk)
        known = n is not None
        if not known:
            n = tp.add_node(x, y, th)
            self.log("new intersection node %d at (%.2f, %.2f)" % (n, x, y))
        nd = tp.nodes[n]
        nd["visits"] += 1
        if fk is not None:
            if fk in tp.edges and tp.edges[fk]["to"] != n:
                self.log("map: edge %s now leads to %d (was %d)" % (fk, n, tp.edges[fk]["to"]))
            if fk not in tp.edges or tp.edges[fk]["to"] != n:
                tp.set_edge(self.last_node, self.last_turn, n, self.edge_dist)
                self.log("learned edge %s -> node %d (%.2f m)" % (fk, n, self.edge_dist))
        # resolve places saved on the edge we just finished
        for name, p in self.places.items():
            if p.get("pending"):
                p.clear()
                p.update({"to": n, "rem": round(max(0.0, self.edge_dist - self.pending_places.pop(name, 0.0)), 3)})
            elif p.get("from") == self.last_node and p.get("turn") == self.last_turn and p.get("to") is None and self.last_node is not None:
                p["to"], p["rem"] = n, round(max(0.0, self.edge_dist - p["s"]), 3)
        self.at_node = n
        return n, ((nd["x"], nd["y"], nd["th"]) if known else None)

    def decide(self, n):
        """Choose the turn at node n. Returns {"turn": t, "stop_s": s|None} or {"done": message}."""
        tp = self.topo
        if self.mode == "follow":
            ops = tp.open_turns(n)
            for t in ("straight", "right", "left"):
                if t in ops:
                    return self._leave(n, t)
            return {"done": "No open exit at this intersection; stopping."}
        if self.mode == "map":
            t = tp.toward_unexplored(n)
            if t is None:
                self.mode = "idle"
                msg = "Track mapped: %d intersections, %d connections." % (len(tp.nodes), len(tp.edges))
                return {"done": msg}
            return self._leave(n, t)
        if self.mode == "goto":
            g = self.goal
            plan = tp.route_to_place(n, g)
            if plan is not None:
                self.route = plan["steps"]
                if not plan["steps"]:                       # the place is this stop line
                    self.mode = "idle"
                    return {"done": "Arrived at %s." % self.goal_name, "arrived": True}
                t = plan["steps"][0][1]
                last = len(plan["steps"]) == 1
                return self._leave(n, t, plan["stop_s"] if last else None)
            t = tp.toward_unexplored(n)
            if t is None:
                self.mode = "idle"
                return {"done": "I can't find a way to %s on this track." % self.goal_name}
            self.message = "exploring to find a way to %s" % self.goal_name
            return self._leave(n, t)
        return {"done": "Stopped at the intersection."}

    def _leave(self, n, t, stop_s=None):
        self.last_node, self.last_turn, self.at_node = n, t, None
        self.next_turn = None
        if self.route and self.route[0][0] == n:
            self.route = self.route[1:]
        self.next_turn = self.route[0][1] if self.route else None
        self.edge_dist = 0.0
        return {"turn": t, "stop_s": stop_s}

    def turn_failed(self):
        if self.last_node is not None and self.last_turn:
            self.topo.mark_blocked(self.last_node, self.last_turn)
            self.log("map: %s turn at node %d marked as not existing" % (self.last_turn, self.last_node))

    def view(self):
        return {"mode": self.mode, "goal": self.goal_name, "last_node": self.last_node, "last_turn": self.last_turn,
                "at_node": self.at_node, "edge_dist": round(self.edge_dist, 3), "next_turn": self.next_turn,
                "route": [{"node": n, "turn": t} for n, t in self.route], "message": self.message}
