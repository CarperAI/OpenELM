from elm.environments.environments import (
    FunctionOptim,
    ImageOptim,
    MatchString,
    Sodarace,
)

IMAGE_SEED = """def draw_blue_rectangle() -> np.ndarray:
\tpic = np.zeros((32, 32, 3))
\tfor x in range(2, 30):
\t\tfor y in range(2, 30):
\t\t\tpic[x, y] = np.array([0, 0, 255])
\treturn pic
"""
