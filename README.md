# SepsisPredictor-ML

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Docker](https://img.shields.io/badge/container-Docker-2496ED)
![scikit-learn](https://img.shields.io/badge/model-Random%20Forest-orange)

A machine-learning model that classifies a patient as **sepsis positive or
negative** from clinical measurements, served through a **FastAPI** web service
and packaged with **Docker**. Early and accurate detection matters in
healthcare, so the aim is to support, not replace, clinical decisions.

> **Disclaimer:** this is a learning project trained on a small dataset. It is
> not a medical device and must not be used for real diagnosis.

Read the full write-up on Medium:
[Revolutionizing early sepsis detection: from data to diagnosis with machine learning](https://medium.com/@feisalhassan77/revolutionizing-early-sepsis-detection-a-journey-from-data-to-diagnosis-with-machine-learning-and-52b8c07bea0d)

## What the project covers

- Data cleaning and preprocessing
- Exploratory data analysis and hypothesis testing
- Training and comparing several models (SVC, Naive Bayes, Random Forest,
  XGBoost and others)
- Model evaluation and selection
- A FastAPI service and a Docker image for deployment

## Results

The **Random Forest** was selected. In the notebook it reached **0.88
accuracy** on the held-out set of 135 patients, and about **0.83 mean
cross-validated accuracy**, ahead of XGBoost (about 0.82), SVC (0.76) and
Gaussian Naive Bayes (0.77). With a dataset this small, treat these numbers as
indicative.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service information |
| POST | `/classify` | Classify one patient |
| GET | `/docs` | Interactive API documentation (Swagger UI) |

![API interface](Images/Api%20Interface.png)

Example request body for `POST /classify`:

```json
{
  "PRG": 6, "PL": 148, "PR": 72, "SK": 35, "TS": 0,
  "M11": 33.6, "BD2": 0.627, "Age": 50, "Insurance": 1
}
```

The response contains the predicted status (Positive or Negative) and the model's
confidence score.

## Repository contents

```
Dev/
  sepsis2.ipynb                    Analysis, hypothesis tests and modelling
  Patients_Files_Train.csv, Patients_Files_Test.csv
SRC/
  main.py                          FastAPI service
  RandomForestClassifier_pipeline.pkl, encoder.pkl   Saved model files
  requirements.txt, dockerfile
Images/Api Interface.png
```

## Setup and run

```bash
git clone https://github.com/Feiiiisal/SepsisPredictor-ML.git
cd SepsisPredictor-ML/SRC
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open <http://127.0.0.1:8000/docs>. The service loads the saved model files
from the folder it is started in, so run it from `SRC`.

## Docker

The image is on Docker Hub:
[feiiisal/sepsis_app2](https://hub.docker.com/r/feiiisal/sepsis_app2)

```bash
docker pull feiiisal/sepsis_app2
```

## Author

**Feisal Ali Hassan**

## License

[MIT](LICENSE)
