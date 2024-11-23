from fastapi import FastAPI
from pydantic import BaseModel
from model import Model


app = FastAPI()
model = Model()

class InputText(BaseModel):
    text: str
    max_length: int = 10000

# Маршрут для генерации текста
@app.post("/generate")
async def generate_text(input_text: InputText):
    response = model.predict(input_text)
    return {"result": response}
