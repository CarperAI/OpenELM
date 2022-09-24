# %%
import requests
import json

with open("generated_code.json") as fl:
    data = json.load(fl)

x = requests.post("http://localhost:5000/eval", json=data)
print(x.text)

