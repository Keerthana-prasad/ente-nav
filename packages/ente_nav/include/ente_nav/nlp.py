"""Natural-language -> robot intents.

Primary: Cactus Needle (tiny on-device tool-calling model, pip install cactus-needle).
Fallback: regex parser, so commands still work if Needle is missing or errors.
Optional: set NLP_URL to a Needle server running on your laptop (tools/nlp_server.py).

Needle never drives the robot. Its "tools" only record an intent; the node validates and executes it.
"""
import json
import os
import re

_INTENTS = []

try:
    import needle
    HAVE_NEEDLE = True
except Exception:  # not installed / unsupported python in the image
    needle = None
    HAVE_NEEDLE = False

if HAVE_NEEDLE:
    @needle.tool
    def go_to(place: str) -> str:
        """Drive the robot to a named place or landmark, for example home, start, stop sign, kitchen, tag 12."""
        _INTENTS.append({"intent": "goto", "place": place})
        return "ok"

    @needle.tool
    def go_to_xy(x: float, y: float) -> str:
        """Drive the robot to map coordinates x and y given in metres."""
        _INTENTS.append({"intent": "goto_xy", "x": x, "y": y})
        return "ok"

    @needle.tool
    def drive_forward(distance_m: float) -> str:
        """Drive straight ahead by a distance in metres."""
        _INTENTS.append({"intent": "forward", "d": distance_m})
        return "ok"

    @needle.tool
    def turn_left(degrees: float) -> str:
        """Turn the robot to the left in place by a number of degrees."""
        _INTENTS.append({"intent": "turn", "deg": abs(degrees)})
        return "ok"

    @needle.tool
    def turn_right(degrees: float) -> str:
        """Turn the robot to the right in place by a number of degrees."""
        _INTENTS.append({"intent": "turn", "deg": -abs(degrees)})
        return "ok"

    @needle.tool
    def stop() -> str:
        """Stop the robot and cancel the current trip."""
        _INTENTS.append({"intent": "stop"})
        return "ok"

    @needle.tool
    def explore() -> str:
        """Explore and map the surroundings by looking around and moving in small steps."""
        _INTENTS.append({"intent": "explore"})
        return "ok"

    @needle.tool
    def save_location(name: str) -> str:
        """Remember the robot's current position under a name, for example kitchen or parking."""
        _INTENTS.append({"intent": "save", "name": name})
        return "ok"

    @needle.tool
    def set_speed(level: str) -> str:
        """Set the driving speed to slow, medium or fast."""
        _INTENTS.append({"intent": "speed", "level": level})
        return "ok"

    @needle.tool
    def name_tag_in_view(name: str) -> str:
        """Give a name to the landmark tag the robot is currently looking at, for example stop sign or door."""
        _INTENTS.append({"intent": "name_tag", "name": name})
        return "ok"

    @needle.tool
    def manual_mode() -> str:
        """Hand control to the human operator (manual driving)."""
        _INTENTS.append({"intent": "manual"})
        return "ok"

    @needle.tool
    def follow_lane() -> str:
        """Drive along the lane on the track, stopping at every red stop line."""
        _INTENTS.append({"intent": "lane_follow"})
        return "ok"

    @needle.tool
    def map_track() -> str:
        """Learn the whole track: drive every road and try every turn at every intersection."""
        _INTENTS.append({"intent": "map_track"})
        return "ok"

    TOOLS = [follow_lane, map_track, go_to, go_to_xy, drive_forward, turn_left, turn_right, stop, explore, save_location, set_speed, name_tag_in_view, manual_mode]
else:
    TOOLS = []

_NUM = r"(-?\d+(?:\.\d+)?)"


def regex_parse(text):
    t = text.lower().strip()
    if re.search(r"\b(manual|take over|let me drive)\b", t):
        return [{"intent": "manual"}]
    if re.search(r"\b(map|learn|explore|survey)\b.*\b(track|road|roads|mat|course|intersections?)\b", t):
        return [{"intent": "map_track"}]
    if re.search(r"\b(follow|stay in|keep (to|in)|drive (in|along)|lane[- ]?follow(ing)?)\b.*\b(lanes?|road|track)\b|\blane[- ]?follow", t):
        return [{"intent": "lane_follow"}]
    if re.search(r"\b(explore|map (the )?(area|room|place)|look around)\b", t):
        return [{"intent": "explore"}]
    m = re.search(r"\b(?:name|call|label)\b\s+(?:the\s+)?(?:tag|sign|marker|landmark)(?:\s+(?:in view|in front|here|ahead))?\s*(?:as|is)?\s*(?:the\s+)?([a-z0-9 _-]+)$", t)
    if m:
        return [{"intent": "name_tag", "name": m.group(1).strip()}]
    m = re.search(r"\b(?:save|remember|mark|name|call)\b\s*(?:this|here|this spot|this place|the spot|my position)?\s*(?:as|is|:)?\s*(?:the\s+)?([a-z0-9 _-]+)$", t)
    if m and m.group(1).strip():
        return [{"intent": "save", "name": m.group(1).strip()}]
    m = re.search(r"^(?:this|here)\s+is\s+(?:the\s+)?([a-z0-9 _-]+)$", t)
    if m:
        return [{"intent": "save", "name": m.group(1).strip()}]
    if re.search(r"\b(stop|halt|freeze|cancel|abort)\b", t) and not re.search(r"\b(go|drive|navigate|head|take me|bring me|move|travel)\b", t):
        return [{"intent": "stop"}]
    m = re.search(r"\b(slow|medium|fast)\b", t)
    if m and (re.fullmatch(r"(slow|medium|fast)( speed)?", t) or re.search(r"\b(speed|go|drive|move)\b", t)) and not re.search(r"\b(to|towards)\b", t):
        return [{"intent": "speed", "level": m.group(1)}]
    m = re.search(r"turn\s+(left|right)(?:\s+(?:by\s+)?" + _NUM + r")?", t)
    if m:
        deg = float(m.group(2) or 90)
        return [{"intent": "turn", "deg": deg if m.group(1) == "left" else -deg}]
    m = re.search(r"(?:forward|ahead|straight)\D*?(\d+(?:\.\d+)?|half|a|one|two|three)\s*(?:a\s+)?(cm|centimeters?|centimetres?|m|meters?|metres?)?", t)
    if m:
        words = {"half": 0.5, "a": 1.0, "one": 1.0, "two": 2.0, "three": 3.0}
        d = words.get(m.group(1)) or float(m.group(1))
        if m.group(2) and m.group(2).startswith(("cm", "centi")):
            d /= 100.0
        return [{"intent": "forward", "d": d}]
    m = re.search(r"\b(?:x\s*=?\s*)" + _NUM + r"[ ,]+(?:and\s+)?y\s*=?\s*" + _NUM, t)
    if m:
        return [{"intent": "goto_xy", "x": float(m.group(1)), "y": float(m.group(2))}]
    m = re.search(r"\b(?:go|drive|navigate|move|head|take me|bring me|come|return|travel)\b.*?\b(?:to|towards|back to)\s+(?:the\s+)?(.+)$", t)
    if m:
        return [{"intent": "goto", "place": m.group(1).strip(" .!?")}]
    m = re.search(r"\b(?:go|return|come)\s+(home|back)\b", t)
    if m:
        return [{"intent": "goto", "place": "home"}]
    return []


class NLP:
    def __init__(self, weights=None, log=print, url=None):
        self.log = log
        self.agent = None
        self.url = url or os.environ.get("NLP_URL")
        if HAVE_NEEDLE and not self.url:
            try:
                kw = {"tools": TOOLS}
                if weights and os.path.exists(weights):
                    kw["weights"] = weights
                try:
                    self.agent = needle.Needle(stateless=True, **kw)   # every command is independent
                except TypeError:
                    self.agent = needle.Needle(**kw)
                log("Needle loaded")
            except Exception as e:
                log("Needle failed to load (%s); using regex fallback" % e)

    def parse(self, text):
        intents = []
        if self.url:
            try:
                import urllib.request
                req = urllib.request.Request(self.url, json.dumps({"text": text}).encode(),
                                             {"Content-Type": "application/json"})
                intents = json.loads(urllib.request.urlopen(req, timeout=3).read())["intents"]
            except Exception as e:
                self.log("NLP server unreachable: %s" % e)
        elif self.agent:
            del _INTENTS[:]
            try:
                if hasattr(self.agent, "reset"):
                    self.agent.reset()
                res = self.agent.run(text)
                conf = res.get("confidence") if isinstance(res, dict) else None
                if conf is not None and conf < 0.2:      # Needle itself isn't sure -> don't act on it
                    self.log("Needle low confidence (%.2f), using rules" % conf)
                else:
                    for it in _INTENTS:                      # drop repeats, cap at 3 steps
                        if it not in intents:
                            intents.append(it)
                    intents = intents[:3]
            except Exception as e:
                self.log("Needle error: %s" % e)
        return intents or regex_parse(text)
