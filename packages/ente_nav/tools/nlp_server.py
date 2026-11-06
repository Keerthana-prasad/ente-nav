#!/usr/bin/env python3
"""Run Needle on your laptop and let the Duckiebot call it over Wi-Fi.
   pip install cactus-needle flask
   PYTHONPATH=../include python3 nlp_server.py
Then set  nlp_url: http://<laptop-ip>:5005/parse  in config/ente_nav.yaml.
Needs no API key: Needle runs fully offline. (An API key is only for Cactus' optional cloud fine-tuning.)"""
from flask import Flask, request, jsonify
from ente_nav.nlp import NLP, HAVE_NEEDLE

nlp, app = NLP(), Flask(__name__)
print("Needle available:", HAVE_NEEDLE, "| agent loaded:", nlp.agent is not None)


@app.route("/parse", methods=["POST"])
def parse():
    intents = nlp.parse(request.get_json()["text"])
    print(request.get_json()["text"], "->", intents)
    return jsonify(intents=intents)


app.run(host="0.0.0.0", port=5005)
