from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    sepal_length: float = Field(5.1, ge=0, description="Sepal length (cm)", examples=[5.1])
    sepal_width: float = Field(3.5, ge=0, description="Sepal width (cm)", examples=[3.5])
    petal_length: float = Field(1.4, ge=0, description="Petal length (cm)", examples=[1.4])
    petal_width: float = Field(0.2, ge=0, description="Petal width (cm)", examples=[0.2])


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str