import json

from flask import Flask, abort, jsonify, request
from numpy import ndarray

from .utils import sandbox_unsafe_execute
from .walker.walk_creator import Walker

app = Flask(__name__)


def bad_request(message, **kwargs):
    return {"message": message, **kwargs}, 500


def generate_racer(code_str, timeout):
    try:
        execution_result = sandbox_unsafe_execute(code_str, "make_walker", timeout)
    except:
        # print("hi 3", execution_result)
        return bad_request(f"failed to execute code")
    if isinstance(execution_result, Walker):
        sodaracer_dict: dict = execution_result.serialize_walker_sodarace()
        if execution_result.validate():
            return {
                "program_str": code_str,
                "sodaracer": sodaracer_dict,
            }, 200
        else:
            return bad_request(
                f"invalid walker", walker=execution_result.serialize_walker_sodarace()
            )
    elif isinstance(execution_result, int):
        return bad_request(
            f"failed sandbox_unsafe_execute", unsafe_execute_error_code=execution_result
        )
    else:
        return bad_request(f"not walker")


@app.route("/gen_racer", methods=["POST"])
def gen_racer():
    req_json = request.get_json()
    if isinstance(req_json["code"], list):
        return [
            generate_racer(code, req_json["timeout"])[0] for code in req_json["code"]
        ], 200
    else:
        return generate_racer(req_json["code"], req_json["timeout"])


@app.route("/eval_imageoptim_func", methods=["POST"])
def evaluate_function():
    req_json = request.get_json()
    try:
        execution_result = sandbox_unsafe_execute(
            req_json["code"], req_json["func_name"], 5.0
        )
        if isinstance(execution_result, ndarray):
            return execution_result.tolist().__repr__()
        elif isinstance(execution_result, int):
            abort(
                500,
                description=f"failed sandbox_unsafe_execute with code {execution_result}",
            )
    except:
        abort(500, description="failed to execute code")
