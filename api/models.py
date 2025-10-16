# -*- coding: utf-8 -*-
"""
Modelos Pydantic para la API de Predicción de Deserción Estudiantil.

Estos modelos definen la estructura de los datos para las solicitudes (requests)
y respuestas (responses) de la API, asegurando una validación de tipos robusta.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class StudentData(BaseModel):
    """
    Define la estructura y los tipos de datos para un estudiante de entrada.
    Todos los campos son requeridos por el modelo para realizar una predicción.
    Los valores de ejemplo se toman del notebook de entrenamiento.
    """
    model_config = ConfigDict(populate_by_name=True)

    curricular_units_1st_sem_enrolled: int = Field(..., alias="Curricular units 1st sem (enrolled)", json_schema_extra={"example": 6})
    curricular_units_1st_sem_evaluations: int = Field(..., alias="Curricular units 1st sem (evaluations)", json_schema_extra={"example": 8})
    curricular_units_1st_sem_approved: int = Field(..., alias="Curricular units 1st sem (approved)", json_schema_extra={"example": 5})
    curricular_units_1st_sem_grade: float = Field(..., alias="Curricular units 1st sem (grade)", json_schema_extra={"example": 12.5})
    curricular_units_2nd_sem_enrolled: int = Field(..., alias="Curricular units 2nd sem (enrolled)", json_schema_extra={"example": 6})
    curricular_units_2nd_sem_evaluations: int = Field(..., alias="Curricular units 2nd sem (evaluations)", json_schema_extra={"example": 7})
    curricular_units_2nd_sem_approved: int = Field(..., alias="Curricular units 2nd sem (approved)", json_schema_extra={"example": 4})
    curricular_units_2nd_sem_grade: float = Field(..., alias="Curricular units 2nd sem (grade)", json_schema_extra={"example": 11.8})
    age_at_enrollment: int = Field(..., alias="Age at enrollment", json_schema_extra={"example": 19})
    scholarship_holder: int = Field(..., alias="Scholarship holder", json_schema_extra={"example": 1})
    gender: int = Field(..., alias="Gender", json_schema_extra={"example": 0})
    mothers_qualification: int = Field(..., alias="Mother's qualification", json_schema_extra={"example": 1})
    fathers_qualification: int = Field(..., alias="Father's qualification", json_schema_extra={"example": 1})
    mothers_occupation: int = Field(..., alias="Mother's occupation", json_schema_extra={"example": 5})
    fathers_occupation: int = Field(..., alias="Father's occupation", json_schema_extra={"example": 5})
    displaced: int = Field(..., alias="Displaced", json_schema_extra={"example": 0})
    educational_special_needs: int = Field(..., alias="Educational special needs", json_schema_extra={"example": 0})
    tuition_fees_up_to_date: int = Field(..., alias="Tuition fees up to date", json_schema_extra={"example": 1})
    debtor: int = Field(..., alias="Debtor", json_schema_extra={"example": 0})
    daytime_evening_attendance: int = Field(..., alias="Daytime/evening attendance", json_schema_extra={"example": 1})
    previous_qualification: int = Field(..., alias="Previous qualification", json_schema_extra={"example": 1})


class PredictionResponse(BaseModel):
    """
    Define la estructura de la respuesta de la predicción.
    """
    prediction: Literal["Dropout", "Graduate"] = Field(..., json_schema_extra={"example": "Graduate"})


class HealthResponse(BaseModel):
    """
    Define la estructura de la respuesta del health check.
    """
    status: str = Field(..., json_schema_extra={"example": "healthy"})
