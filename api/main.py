# -*- coding: utf-8 -*-
"""
API de FastAPI para el Modelo de Predicción de Deserción Estudiantil
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import StudentData, PredictionResponse, HealthResponse
from src.pipeline.inference import Predictor

# --- Añadir el directorio raíz al sys.path ---
# Esto es necesario para que la API (que corre desde el directorio raíz)
# pueda encontrar los módulos en 'src' y 'api'.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

# --- Inicialización de la App y el Modelo ---

# Crear la instancia de la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Deserción Estudiantil",
    description=(
        "Una API para predecir si un estudiante se graduará o abandonará "
        "sus estudios."
    ),
    version="1.0.0"
)

# Añadir middleware de CORS para permitir solicitudes desde cualquier origen.
# En un entorno de producción, esto debería restringirse a dominios específicos.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"])

# Cargar el modelo y el preprocesador.
# Esto se hace una sola vez cuando la aplicación se inicia para evitar recargar
# el modelo en cada solicitud, lo cual sería muy ineficiente.
try:
    predictor = Predictor()
except Exception as e:
    # Si el modelo no se puede cargar, la API no puede funcionar.
    # Lanzamos una excepción para detener el inicio.
    raise RuntimeError(
        "Error al iniciar la API: No se pudo cargar el modelo. "
        f"Detalles: {e}"
    )


# --- Endpoints de la API ---

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Endpoint de Health Check.

    Devuelve el estado de la API. Es útil para sistemas de monitoreo
    para verificar que la aplicación está viva y funcionando.
    """
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_student_status(student_data: StudentData):
    """
    Endpoint para realizar una predicción sobre un estudiante.

    Recibe los datos de un estudiante en el cuerpo de la solicitud
    (request body) y devuelve una predicción de si el estudiante es
    propenso a abandonar ("Dropout") o graduarse ("Graduate").
    """
    try:
        # El modelo Pydantic (StudentData) ya ha validado que los datos de
        # entrada tienen el tipo correcto. Ahora los convertimos a un
        # diccionario. Usamos `by_alias=True` para que las claves del
        # diccionario sean los nombres de las columnas originales que el
        # modelo espera.
        input_dict = student_data.model_dump(by_alias=True)

        # Realizar la predicción. El método `predict` de nuestra clase
        # `Predictor` espera un diccionario o DataFrame.
        prediction_list = predictor.predict(input_dict)

        # El predictor devuelve una lista, pero para este endpoint solo
        # procesamos un estudiante. Tomamos el primer y único resultado.
        prediction = prediction_list[0]

        # Devolver la predicción en el formato definido por
        # PredictionResponse.
        return {"prediction": prediction}

    except Exception as e:
        # Si ocurre cualquier error durante la predicción, devolvemos un
        # error HTTP 500. Esto podría pasar si hay un problema inesperado
        # con los datos de entrada a pesar de la validación de Pydantic.
        raise HTTPException(
            status_code=500,
            detail=f"Ocurrió un error durante la predicción: {str(e)}"
        )

# --- Bloque para ejecución directa (para pruebas) ---


if __name__ == "__main__":
    """
    Este bloque permite correr la API directamente con Uvicorn para
    pruebas locales. Comando para ejecutar: `python api/main.py`
    """
    import uvicorn

    print("Iniciando servidor de FastAPI en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
