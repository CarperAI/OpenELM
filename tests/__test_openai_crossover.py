from time import sleep

from openelm.codegen.codegen_utilities import truncate
from openelm.environments.sodaracer.walker import Walker
from openelm.sandbox.server.utils import sandbox_unsafe_execute

# import openai


API_KEY = ""

PROMPT = """
from openelm.environments.sodaracer.walker.walk_creator import walker_creator
import math


def make_circle(wc, cx, cy, radius, num_points):
    \"\"\"Approximate a circle with center (cx,cy) square with num_points points.\"\"\"
    joints = []

    tot_ang = 3.14 * 2.0

    for idx in range(num_points):
        ang = (tot_ang / num_points) * idx
        x = math.cos(ang) * radius + cx
        y = math.sin(ang) * radius + cy
        joints.append(wc.add_joint(x, y))

    return joints

def make_radial_walker():
    \"\"\"Create a radial walker.\"\"\"
    wc = walker_creator()

    num_points = 8
    rad = 5.0
    cx, cy = (5, 5)
    # the main body is a square
    points = make_circle(wc, cx, cy, rad, num_points)
    center = wc.add_joint(cx, cy)

    for k in range(num_points):
        wc.add_muscle(points[k], points[(k + 1) % num_points])
        wc.add_muscle(points[k], center, float(k) / num_points, float(k) / num_points)

    return wc.get_walker()

def make_wheel_walker():
    \"\"\"Create a wheel walker.\"\"\"
    wc = walker_creator()
    num_points = 8
    rad = 3.0
    cx, cy = (11, 5)
    points = make_circle(wc, 0.6, -0.5, rad / 2, num_points)
    center = wc.add_joint(cx + 1, cy + 1)
    for j in range(num_points):
        for i in range(num_points - 5):
            wc.add_muscle(points[j], points[(i + j) % num_points],
                          0.0, 1.0, (j + 1) / num_points)
        wc.add_muscle(points[j], center, 3, (j + 1) / num_points)
    return wc.get_walker()

def make_square(wc, x0, y0, x1, y1):
    \"\"\"Make a square with top left x0,y0 and top right x1,y1.\"\"\"
    j0 = wc.add_joint(x0, y0)
    j1 = wc.add_joint(x0, y1)
    j2 = wc.add_joint(x1, y1)
    j3 = wc.add_joint(x1, y0)
    return j0, j1, j2, j3


def make_square_walker():
    \"\"\"Create a square walker.\"\"\"
    wc = walker_creator()

    # the main body is a square
    sides = make_square(wc, 0, 0, 10, 10)
    center = wc.add_joint(5, 5)

    # connect the square with distance muscles
    for k in range(len(sides) - 1):
        wc.add_muscle(sides[k], sides[k + 1])
    wc.add_muscle(sides[3], sides[0])

    # one prong of the square is a distance muscle
    wc.add_muscle(sides[3], center)

    # the other prongs from the center of the square are active
    wc.add_muscle(sides[0], center, 5.0, 0.0)
    wc.add_muscle(sides[1], center, 10.0, 0.0)
    wc.add_muscle(sides[2], center, 2.0, 0.0)

    return wc.get_walker()

##Combine the radial, wheel, and square seed programs above to make a new walker.
"""

TEST_PROMPT = """
from openelm.environments.sodaracer.walker.walk_creator import walker_creator

def make_square(wc, x0, y0, x1, y1):
    \"\"\"Make a square with top left x0,y0 and top right x1,y1.\"\"\"
    j0 = wc.add_joint(x0, y0)
    j1 = wc.add_joint(x0, y1)
    j2 = wc.add_joint(x1, y1)
    j3 = wc.add_joint(x1, y0)
    return j0, j1, j2, j3


def make_walker():
    \"\"\"Create a square walker.\"\"\"
    wc = walker_creator()

    # the main body is a square
    sides = make_square(wc, 0, 0, 10, 10)
    center = wc.add_joint(5, 5)

    # connect the square with distance muscles
    for k in range(len(sides) - 1):
        wc.add_muscle(sides[k], sides[k + 1])
    wc.add_muscle(sides[3], sides[0])

    # one prong of the square is a distance muscle
    wc.add_muscle(sides[3], center)

    # the other prongs from the center of the square are active
    wc.add_muscle(sides[0], center, 5.0, 0.0)
    wc.add_muscle(sides[1], center, 10.0, 0.0)
    wc.add_muscle(sides[2], center, 2.0, 0.0)

    return wc.get_walker()
"""


def query(n=1):
    func_start = "\ndef make_walker():\n"
    temperature = 0.88
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=PROMPT + func_start,
        stop="```",
        max_tokens=2048,
        temperature=temperature,
        n=n,
        stream=True,
    )
    completion = func_start
    for event in response:
        event_text = event["choices"][0]["text"]
        completion += event_text
    completion = truncate(completion)
    try:
        execution_result = sandbox_unsafe_execute(PROMPT + completion, "make_walker")
        if isinstance(execution_result, Walker):
            if execution_result.validate():
                print(completion)
                return 1
        else:
            print("Failed execution, type:", type(execution_result))
            return execution_result
    except Exception as e:
        print(e, "Exception:")
        return 6


if __name__ == "__main__":
    openai.api_key = API_KEY
    # print(truncate(TEST_PROMPT, def_num=2))
    # try:
    #     ast.parse(TEST_PROMPT)
    #     execution_result = sandbox_unsafe_execute(TEST_PROMPT, "make_walker")
    #     if isinstance(execution_result, Walker):
    #         if execution_result.validate():
    #             print(1)
    # except:
    #     print(0)
    results = []
    for i in range(10):
        i_result = query(1)
        print(i_result)
        results.append(i_result)
        sleep(31)
    print((results.count(1) / len(results)) * 100, "%")
