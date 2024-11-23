import requests
import pandas as pd

def api_request(text):
    url = "https://seal-touched-bat.ngrok-free.app/generate"

    payload = {
        "text": text,
        "max_length": 10000
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print("Ошибка:", response.status_code, response.text)

def csv_to_string(sport_name):
    if sport_name == 'hockey':
        df = pd.read_csv(r'C:\AI camps\Hackatons\MPIT\Data\hockey.csv')
    else:
        df = pd.read_csv(r'football.csv')
    df = df.sample(frac=1)
    return df.to_string()

text = csv_to_string(sport_name='hockey')

answer = api_request(text)
print(answer)