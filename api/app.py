# Standard library imports
import os
import re
import pickle

# Third-party library imports
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, StrictInt, StrictFloat
import uvicorn

# Create FastAPI app
app = FastAPI()


# Pydantic data model: Pipeline input dictionary
class pipeline_input_dict(BaseModel):
    age: StrictInt | StrictFloat
    married: str
    income: StrictInt | StrictFloat
    car_ownership: str
    house_ownership: str 
    current_house_yrs: StrictInt | StrictFloat
    city: str 
    state: str 
    profession: str 
    experience: StrictInt | StrictFloat
    current_job_yrs: StrictInt | StrictFloat


# Launch API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
