import json

from flask import Flask, abort, jsonify, request
from numpy import ndarray

from .environments.walker.walk_creator import Walker
from .utils import sandbox_unsafe_execute

app = Flask(__name__)


def bad_request(message, **kwargs):
    return {"message": message, **kwargs}, 500


def generate_racer(code_str, timeout):
    try:
        execution_result = sandbox_unsafe_execute(code_str, "make_walker", timeout)
    except:
        return bad_request(f"failed to execute code", unsafe_execute_error_code=6)  # 6: Other errors.
    if isinstance(execution_result, Walker):
        sodaracer_dict: dict = execution_result.to_dict()
        if execution_result.validate():
            return {
                "program_str": code_str,
                "result_obj": sodaracer_dict,
            }, 200
        else:
            # 1: Code runs but fails a test (valid walker).
            return bad_request(f"invalid walker", walker=execution_result.to_dict(), unsafe_execute_error_code=1)
    elif isinstance(execution_result, int):
        return bad_request(
            f"failed sandbox_unsafe_execute", unsafe_execute_error_code=execution_result
        )
    else:
        return bad_request(f"not walker", unsafe_execute_error_code=1)  # 1: Code runs but fails a test (valid walker).


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
            req_json["code"], req_json["func_name"], req_json["timeout"]
        )
        if isinstance(execution_result, ndarray):
            return {"program_str": req_json["code"],
                    "result_obj": execution_result.tolist().__repr__()}, 200
        elif isinstance(execution_result, int):
            return bad_request(
                f"failed sandbox_unsafe_execute", unsafe_execute_error_code=execution_result
            )
        else:
            bad_request(f"not image", unsafe_execute_error_code=1)  # 1: Code runs but fails a test (valid image).
    except:
        return bad_request(f"failed to execute code", unsafe_execute_error_code=6)  # 6: Other errors.
