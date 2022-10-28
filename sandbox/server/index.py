from flask import Flask, jsonify, request, abort
from .utils import unsafe_execute
from .walker.walk_creator import Walker
from numpy import ndarray


app = Flask(__name__)


def generate_racer(code_str, timeout):
    try:
        execution_result = unsafe_execute(code_str, "make_walker", timeout)
    except:
        # print("hi 3", execution_result)
        abort(500, description=f"failed to execute code")
    if isinstance(execution_result, Walker):
        if execution_result.validate():
            sodaracer_dict: dict = execution_result.serialize_walker_sodarace()
            return {
                "program_str": code_str,
                "sodaracer": sodaracer_dict,
            }
        else:
            abort(500, description=f"walker not valid")    
    elif isinstance(execution_result, int):
        abort(500, description=f"failed unsafe_execute with code {execution_result}")
    else: 
        abort(500, description=f"not walker")
    




@app.route("/gen_racer", methods=["POST"])
def gen_racer():
    req_json = request.get_json()
    return generate_racer(req_json["code"], req_json["timeout"]).__repr__()


@app.route("/eval_imageoptim_func", methods=["POST"])
def evaluate_function():
    req_json = request.get_json()
    try:
        execution_result = unsafe_execute(req_json["code"], req_json["func_name"], 5.0)
        if isinstance(execution_result, ndarray):
            return execution_result.tolist().__repr__()
        elif isinstance(execution_result, int):
            abort(500, description=f"failed unsafe_execute with code {execution_result}")
    except:
        abort(500, description="failed to execute code")

