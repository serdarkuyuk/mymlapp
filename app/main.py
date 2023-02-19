import uvicorn
from fastapi import FastAPI
import numpy as np
import joblib
from pydantic.main import BaseModel

app = FastAPI(title='My ML App')

@app.get('/')
def home():
    return "Api is healthy"

class MyData(BaseModel):
    x : int


@app.post('/predict')
def predict(data:MyData):
    my_model = joblib.load('app/my_model')
    x = np.array(data.x).reshape(-1, 1)
    return str(my_model.predict(x))



if __name__== "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)