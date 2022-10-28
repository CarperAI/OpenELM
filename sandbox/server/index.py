from flask import Flask, jsonify, request
from .utils import unsafe_execute
from .walker.walk_creator import Walker
from numpy import ndarray


app = Flask(__name__)


def generate_racer(code_str, func_name, timeout):
    try:
        execution_result = unsafe_execute(code_str, func_name, timeout)
        if isinstance(execution_result, Walker):
            if execution_result.validate():
                sodaracer_dict: dict = execution_result.serialize_walker_sodarace()
                return {
                    "program_str": code_str,
                    "sodaracer": sodaracer_dict,
                }
            else:
                return "walker not valid"
        else: 
            return "not walker"
    except:
        return "failed to execute code"




@app.route("/gen_racer", methods=["POST"])
def gen_racer():
    req_json = request.get_json()
    return generate_racer(req_json["code"], "make_walker", req_json["timeout"]).__repr__()


@app.route("/eval_imageoptim_func", methods=["POST"])
def evaluate_function():
    req_json = request.get_json()
    try:
        execution_result = unsafe_execute(req_json["code"], req_json["func_name"], 5.0)
        if isinstance(execution_result, ndarray):
            return execution_result.tolist().__repr__()
        elif isinstance(execution_result, int):
            return f"failed unsafe_execute with code {execution_result}"
    except:
        return "failed to execute code"
