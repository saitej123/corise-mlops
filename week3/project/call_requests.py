import requests


REQUESTS_JSON_PATH = "./data/requests.json"

with open(REQUESTS_JSON_PATH) as file:
    for line in file:
        response = requests.post('http://127.0.0.1:8000/predict', data=line)

    print('Finished making predictions. Check output predictions in data/logs.out')