from flask import Flask, jsonify, request
from .utils import unsafe_execute
from .walker.walk_creator import Walker


app = Flask(__name__)


def generate_racer(code_str, timeout):
    try:
        execution_result = unsafe_execute(code_str, timeout)
        print(execution_result)
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




@app.route("/gen", methods=["POST"])
def gen():
    req_json = request.get_json()
    code_str = req_json["code"]
    timeout = req_json["timeout"]
    result = generate_racer(code_str, timeout)
    return result
