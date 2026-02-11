from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.ml.model import IrisModel
from app.schemas import IrisFeatures, PredictionResponse

app = FastAPI(title="Iris ML API", version="0.1.0")

iris_model = IrisModel()

@app.get("/")
def root():
    return {"message": "Iris FastAPI works. Go to /docs"}


@app.on_event("startup")
def startup_event() -> None:
    
    iris_model.load()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: IrisFeatures) -> PredictionResponse:
    features = [
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ]
    pred_class, pred_label = iris_model.predict(features)
    return PredictionResponse(predicted_class=pred_class, predicted_label=pred_label)


@app.exception_handler(FileNotFoundError)
def model_missing_handler(_, exc: FileNotFoundError):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )

