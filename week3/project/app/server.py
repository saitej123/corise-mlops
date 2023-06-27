from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from datetime import datetime
from time import time

from classifier import NewsCategoryClassifier


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


MODEL_PATH = "../data/news_classifier.joblib"
LOGS_OUTPUT_PATH = "../data/logs.out"

app = FastAPI()

@app.on_event("startup")
def startup_event():
    """
    [TO BE IMPLEMENTED]
    1. Initialize an instance of `NewsCategoryClassifier`.
    2. Load the serialized trained model parameters (pointed to by `MODEL_PATH`) into the NewsCategoryClassifier you initialized.
    3. Open an output file to write logs, at the destimation specififed by `LOGS_OUTPUT_PATH`
        
    Access to the model instance and log file will be needed in /predict endpoint, make sure you
    store them as global variables
    """
    global news_clf, logs

    news_clf = NewsCategoryClassifier()
    news_clf.load(MODEL_PATH)

    logs = open(LOGS_OUTPUT_PATH, 'a+')
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
    [TO BE IMPLEMENTED]
    2. Any other cleanups
    """
    logs.flush()
    logs.close()
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
    [TO BE IMPLEMENTED]
    1. run model inference and get model predictions for model inputs specified in `request`
    2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`)
    {
        'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
        'request': dictionary representation of the input request,
        'prediction': dictionary representation of the response,
        'latency': time it took to serve the request, in millisec
    }
    3. Construct an instance of `PredictResponse` and return
    """
    start_time = time()

    predicted_label = news_clf.predict_label(request)[0]
    prediction_scores = news_clf.predict_proba(request)

    request_output = {
        'timestamp': datetime.fromtimestamp(start_time).strftime('%Y/%m/%d %H:%M:%S'),
        'request': request,
        'prediction': prediction_scores,
        'latency': (time() - start_time) * 1000 # time in milliseconds
    }

    logs.write(str(request_output))
    logs.write('\n')
    logs.flush()

    response = PredictResponse(scores=prediction_scores, label=predicted_label)
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}