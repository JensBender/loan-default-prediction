# Standard library imports
import os
import re
import pickle

# Third-party library imports
import pandas as pd
from fastapi import FastAPI
import uvicorn

# Create FastAPI app
app = FastAPI()

# Launch API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
