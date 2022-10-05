from flask import Flask, jsonify, request

app = Flask(__name__)


def unpack_code(req_json):
    return req_json["0"]["code"]


def evaluate_sodaracer(code_string):
    """dummy"""
    return code_string


@app.route("/eval", methods=["POST"])
def eval():
    req_json = request.get_json()
    code_string = unpack_code(req_json)
    result = evaluate_sodaracer(code_string)
    return result
