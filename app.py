from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from base_model.preprocessing import preprocess_data

app = FastAPI(title="Heart Disease Prediction")

templates = Jinja2Templates(directory="/home/stark/Documents/Ai_project/heart_diseases_prediction/application/templates/")

# Load model
model = joblib.load("saved_models/base_logistic_model.pkl")["model"]

FEATURES = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ---------- FORM PREDICTION ----------
@app.post("/predict-form")
async def predict_form(
    male: int = Form(...),
    age: int = Form(...),
    currentSmoker: int = Form(...),
    cigsPerDay: float = Form(0),
    BPMeds: float = Form(0),
    prevalentStroke: int = Form(...),
    prevalentHyp: int = Form(...),
    diabetes: int = Form(...),
    totChol: float = Form(...),
    sysBP: float = Form(...),
    diaBP: float = Form(...),
    BMI: float = Form(...),
    heartRate: float = Form(...),
    glucose: float = Form(...)
):
    X = pd.DataFrame([{
    "male": male,
    "age": age,
    "currentSmoker": currentSmoker,
    "cigsPerDay": cigsPerDay,
    "BPMeds": BPMeds,
    "prevalentStroke": prevalentStroke,
    "prevalentHyp": prevalentHyp,
    "diabetes": diabetes,
    "totChol": totChol,
    "sysBP": sysBP,
    "diaBP": diaBP,
    "BMI": BMI,
    "heartRate": heartRate,
    "glucose": glucose
    }])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return {
        "prediction": "High Risk" if pred == 1 else "Low Risk",
        "probability": round(prob * 100, 2)
    }



# ---------- FILE PREDICTION ----------
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"error": "Only CSV files allowed"}

    df = pd.read_csv(file.file)
    # df = preprocess_data(df)
    print(df.info())

    try:
        X = df[FEATURES]   # exact order, exact columns
    except KeyError as e:
        return {
            "error": "CSV schema mismatch",
            "missing": list(set(FEATURES) - set(df.columns))
        }

    df["Prediction"] = model.predict(X)
    df["Risk_Probability"] = (model.predict_proba(X)[:, 1] * 100).round(2)

    return df[["Prediction", "Risk_Probability"]].to_dict(orient="records")

