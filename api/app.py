# Standard library imports
import os
import re
import pickle

# Third-party library imports
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Create FastAPI app
app = FastAPI()


# Pydantic data model: Pipeline input dictionary
class pipeline_input_dict(BaseModel):
    age: int | float
    married: str
    income: int | float
    car_ownership: str
    house_ownership: str 
    current_house_yrs: int | float
    city: str 
    state: str 
    profession: str 
    experience: int | float
    current_job_yrs: int | float


# Launch API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
