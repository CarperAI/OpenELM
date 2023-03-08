from flask import Flask, request
from numpy import ndarray

from .environments.walker.walk_creator import Walker
from .sandbox_codex_execute import ExecResult, unsafe_execute

app = Flask(__name__)


def bad_request(message, **kwargs):
    return {"message": message, **kwargs}, 500


def generate_racer(code_str: str, timeout: float):
    try:
        execution_result = unsafe_execute(
            code_str, func_name="make_walker", timeout=timeout
        )
    except Exception:
        return bad_request(
            "Failed to execute code", unsafe_execute_error_code=6
        )  # 6: Other errors.
    if isinstance(execution_result, Walker):
        sodaracer_dict: dict = execution_result.to_dict()
        if execution_result.validate():
            return {
                "program_str": code_str,
                "result_obj": sodaracer_dict,
            }, 200
        else:
            # 1: Code runs but fails a test (valid walker).
            return bad_request(
                "Invalid walker",
                walker=execution_result.to_dict(),
                unsafe_execute_error_code=1,
            )
    elif isinstance(execution_result, ExecResult):
        return bad_request(
            "Failed sandbox_unsafe_execute",
            unsafe_execute_error_code=execution_result.name,
        )
    else:
        return bad_request(
            "Not walker", unsafe_execute_error_code=1
        )  # 1: Code runs but fails a test (valid walker).


@app.route("/gen_racer", methods=["POST"])
def gen_racer():
    req_json = request.get_json()
    if req_json:
        if isinstance(req_json["code"], list):
            return [
                generate_racer(code, req_json["timeout"])[0]
                for code in req_json["code"]
            ], 200
        else:
            return generate_racer(req_json["code"], req_json["timeout"])


@app.route("/eval_imageoptim_func", methods=["POST"])
def evaluate_function():
    req_json: dict = request.get_json()
    try:
        execution_result = unsafe_execute(
            code_str=req_json["code"],
            func_name=req_json["func_name"],
            timeout=req_json["timeout"],
        )
        if isinstance(execution_result, ndarray):
            return {
                "program_str": req_json["code"],
                "result_obj": execution_result.tolist().__repr__(),
            }, 200
        elif isinstance(execution_result, ExecResult):
            return bad_request(
                "Failed sandbox_unsafe_execute",
                unsafe_execute_error_code=execution_result.name,
            )
        else:
            bad_request(
                "Not image", unsafe_execute_error_code=1
            )  # 1: Code runs but fails a test (valid image).
    except Exception:
        return bad_request(
            "Failed to execute code", unsafe_execute_error_code=6
        )  # 6: Other errors.


@app.route("/eval_p3_solution", methods=["POST"])
def evaluate_p3_solution():
    req_json = request.get_json()
    try:
        execution_result = unsafe_execute(
            req_json["code"], req_json["func_name"], req_json["timeout"]
        )
        if isinstance(execution_result, ExecResult):
            return bad_request(
                "Failed sandbox_unsafe_execute",
                unsafe_execute_error_code=execution_result.name,
            )
        return {
            "program_str": req_json["code"],
            "result_obj": execution_result.__repr__(),
        }, 200
    except Exception:
        return bad_request("Failed to execute code", unsafe_execute_error_code=6)
        # 6: Other errors.
