#!/usr/bin/env python3
"""Step 1 of the Needle setup: check it works on your laptop, before the robot is involved.
   pip install cactus-needle  &&  PYTHONPATH=../include python3 test_needle.py"""
from ente_nav.nlp import NLP, HAVE_NEEDLE
print("cactus-needle importable:", HAVE_NEEDLE)
n = NLP()
print("Needle loaded:", n.agent is not None, "(False = rule-based fallback is being used)")
for q in ["go to the kitchen", "turn right 30 degrees", "save this as parking", "drive forward half a metre then stop",
          "what is the capital of France"]:
    print(f"{q!r:45} ->", n.parse(q))
