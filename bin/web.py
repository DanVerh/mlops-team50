import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import classifier.actions as actions


# Define a request model for FastAPI
class PredictionRequest(BaseModel):
    text: str


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model
    model = actions.load_trained_model()
    yield
    # Clean up the ML models and release the resources
    pass


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


@app.post("/censorship_status")
async def censorship_status(request: PredictionRequest) -> str:
    is_bad = actions.predict([request.text], model)
    if bool(is_bad):
        return "Bad"
    else:
        return "Good"


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
