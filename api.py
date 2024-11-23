import requests

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
        print("Результат:", response.json())
    else:
        print("Ошибка:", response.status_code, response.text)