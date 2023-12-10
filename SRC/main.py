from fastapi import FastAPI,HTTPException
from typing import Literal,List
import uvicorn
from pydantic import BaseModel
import pandas as pd
import os
import pickle

# setup
SRC = os.path.abspath('.')

# Load the pipeline using pickle
pipeline_path = os.path.join(SRC, 'RandomForestClassifier_pipeline.pkl')
with open(pipeline_path, 'rb') as file:
    rfc_pipeline = pickle.load(file)

# Load the encoder using pickle
encoder_path = os.path.join(SRC, 'encoder.pkl')
with open(encoder_path, 'rb') as file:
    encoder = pickle.load(file)


app = FastAPI(
    title= 'Sepsis Classification FastAPI',
    description='FastAPI to classify Sepsis condition (Positive / Negative)',
    version= '0.104.1'
)


class Sepsis(BaseModel):
    PRG: int
    PL: int 
    PR: int 
    SK : int
    TS: int 
    M11 : float
    BD2 : float
    Age : int
    Insurance: int
    

class Sepsis_Status(BaseModel):
    Sepssis : str
    ConfidenceScore : float


# get
@app.get('/')
def home():
    return{
        'Sepsis Classification FastAPI: FastAPI to classify Sepsis condition (Positive / Negative)',
        
        'Click here (/docs) to access API docs.'
    }
    

# post
@app.post('/classify', response_model=Sepsis_Status)
def sep_classification(sepsis: Sepsis):
    try:
        df = pd.DataFrame([sepsis.model_dump()])
           
        # Make predictions
        prediction = rfc_pipeline.predict(df)
        output = rfc_pipeline.predict_proba(df)

        confidence_score = output.max(axis=-1)[0]
        pred_index = output.argmax(axis=-1)[0]

        # Return result if successful
        return {
            'Sepssis': 'Positive' if pred_index == 1 else 'Negative',
            'ConfidenceScore': confidence_score
        }

    except Exception as e:
        # Return error message and details if an exception occurs
        error_detail = str(e)
        raise HTTPException(status_code=500, detail=f"Error during classification: {error_detail}")




