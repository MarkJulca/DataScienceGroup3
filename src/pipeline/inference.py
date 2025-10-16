# -*- coding: utf-8 -*-
"""
Pipeline de Inferencia para el Modelo de Deserción Estudiantil

Este script carga el modelo entrenado y recrea el preprocesador necesario
para realizar predicciones sobre nuevos datos de estudiantes.
"""

import os
import joblib
import pandas as pd
from typing import Union, Dict, List
from sklearn.preprocessing import StandardScaler

# --- Configuración de Rutas ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))

_MODEL_PATH = os.path.join(_ROOT_DIR, "src", "modelo", "best_model.pkl")
_TRAINING_DATA_PATH = os.path.join(_ROOT_DIR, "src", "preprocessing", "data", "cleaned_raw", "dataset_binary.csv")


class Predictor:
    """
    Clase para cargar el modelo, ajustar el preprocesador y realizar predicciones.
    """

    def __init__(self, model_path: str = _MODEL_PATH, training_data_path: str = _TRAINING_DATA_PATH):
        """
        Inicializa el predictor.

        Args:
            model_path (str): Ruta al archivo .pkl del modelo.
            training_data_path (str): Ruta al CSV con los datos de entrenamiento para ajustar el scaler.
        """
        print(f"Cargando modelo desde: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El archivo del modelo no se encontró en: {model_path}")
        self.model = joblib.load(model_path)
        print("Modelo cargado exitosamente.")

        # El preprocesamiento (StandardScaler) no se guardó con el modelo,
        # por lo que debemos recrearlo y ajustarlo con los datos de entrenamiento.
        self.scaler = self._fit_scaler(training_data_path)
        self.class_labels = {0: "Dropout", 1: "Graduate"}

    def _fit_scaler(self, data_path: str) -> StandardScaler:
        """
        Ajusta un StandardScaler utilizando los datos de entrenamiento originales.
        Esto es crucial para que los nuevos datos se transformen de la misma manera.
        """
        print(f"Ajustando el scaler con los datos de: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"El archivo de datos de entrenamiento no se encontró en: {data_path}")

        # Cargar datos de entrenamiento
        train_df = pd.read_csv(data_path)

        # Definir las 21 columnas de características (features)
        self.feature_columns = [
            'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
            'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
            'Age at enrollment', 'Scholarship holder', 'Gender', "Mother's qualification",
            "Father's qualification", "Mother's occupation", "Father's occupation", 'Displaced',
            'Educational special needs', 'Tuition fees up to date', 'Debtor',
            'Daytime/evening attendance', 'Previous qualification'
        ]

        # Asegurarse de que todas las columnas esperadas están en el dataframe
        if not all(col in train_df.columns for col in self.feature_columns):
            missing_cols = [col for col in self.feature_columns if col not in train_df.columns]
            raise ValueError(f"Faltan columnas en el dataset de entrenamiento: {missing_cols}")

        # Extraer solo las características (X)
        X_train = train_df[self.feature_columns]

        # Crear y ajustar el scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        print("Scaler ajustado exitosamente.")
        return scaler

    def predict(self, input_data: Union[pd.DataFrame, Dict]) -> List[str]:
        """
        Realiza una predicción sobre los datos de entrada.
        """
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise TypeError("El tipo de dato de entrada debe ser un pd.DataFrame o un diccionario.")

        # Validar y reordenar columnas
        if not all(col in input_df.columns for col in self.feature_columns):
            missing_cols = [col for col in self.feature_columns if col not in input_df.columns]
            raise ValueError(f"Faltan las siguientes columnas en los datos de entrada: {missing_cols}")
        input_df = input_df[self.feature_columns]

        # 1. Preprocesar/Escalar los datos de entrada con el scaler ya ajustado
        input_scaled_array = self.scaler.transform(input_df)

        # Convertir el array escalado de nuevo a un DataFrame con los nombres de las columnas.
        # Esto elimina la advertencia de scikit-learn sobre la falta de nombres de características.
        input_scaled_df = pd.DataFrame(input_scaled_array, columns=self.feature_columns)

        # 2. Realizar la predicción con el modelo
        predictions_numeric = self.model.predict(input_scaled_df)

        # Convertir a etiquetas de texto
        predictions_text = [self.class_labels[pred] for pred in predictions_numeric]

        return predictions_text


# --- Bloque de Ejemplo de Uso ---
if __name__ == "__main__":  # pragma: no cover
    print("--- Iniciando Ejemplo de Inferencia ---")
    try:
        predictor = Predictor()

        # --- Ejemplo 1: Predicción para un solo estudiante ---
        print("\n--- Ejemplo 1: Datos de un solo estudiante ---")
        sample_student_data = {
            "Curricular units 1st sem (enrolled)": 6,
            "Curricular units 1st sem (evaluations)": 8,
            "Curricular units 1st sem (approved)": 5,
            "Curricular units 1st sem (grade)": 12.5,
            "Curricular units 2nd sem (enrolled)": 6,
            "Curricular units 2nd sem (evaluations)": 7,
            "Curricular units 2nd sem (approved)": 4,
            "Curricular units 2nd sem (grade)": 11.8,
            "Age at enrollment": 19,
            "Scholarship holder": 1,
            "Gender": 0,
            "Mother's qualification": 1,
            "Father's qualification": 1,
            "Mother's occupation": 5,
            "Father's occupation": 5,
            "Displaced": 0,
            "Educational special needs": 0,
            "Tuition fees up to date": 1,
            "Debtor": 0,
            "Daytime/evening attendance": 1,
            "Previous qualification": 1
        }
        prediction = predictor.predict(sample_student_data)
        print(f"\n>> Predicción para el estudiante de ejemplo: {prediction[0]}")

        # --- Ejemplo 2: Predicción para un estudiante con riesgo ---
        print("\n--- Ejemplo 2: Datos de un estudiante con riesgo de abandono ---")
        risky_student_data = {
            "Curricular units 1st sem (enrolled)": 5,
            "Curricular units 1st sem (evaluations)": 5,
            "Curricular units 1st sem (approved)": 0,
            "Curricular units 1st sem (grade)": 0.0,
            "Curricular units 2nd sem (enrolled)": 5,
            "Curricular units 2nd sem (evaluations)": 0,
            "Curricular units 2nd sem (approved)": 0,
            "Curricular units 2nd sem (grade)": 0.0,
            "Age at enrollment": 22,
            "Scholarship holder": 0,
            "Gender": 1,
            "Mother's qualification": 1,
            "Father's qualification": 1,
            "Mother's occupation": 10,
            "Father's occupation": 10,
            "Displaced": 1,
            "Educational special needs": 0,
            "Tuition fees up to date": 0,
            "Debtor": 1,
            "Daytime/evening attendance": 1,
            "Previous qualification": 1
        }
        prediction_risky = predictor.predict(risky_student_data)
        print(f"\n>> Predicción para el estudiante de riesgo: {prediction_risky[0]}")

    except Exception as e:
        print(f"\nHa ocurrido un error: {e}")

    print("\n--- Fin del Ejemplo de Inferencia ---")
