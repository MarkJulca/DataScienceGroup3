import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import importlib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture
def success_client():
    """Client fixture for successful predictions."""
    with patch('src.pipeline.inference.Predictor') as mock_predictor:
        mock_predictor.return_value.predict.return_value = ["Graduate"]
        
        # Reload the api.main module to apply the patch
        import api.main
        importlib.reload(api.main)
        
        with TestClient(api.main.app) as client:
            yield client

@pytest.fixture
def error_client():
    """Client fixture for failed predictions."""
    with patch('src.pipeline.inference.Predictor') as mock_predictor:
        mock_predictor.return_value.predict.side_effect = Exception("Test Error")
        
        # Reload the api.main module to apply the patch
        import api.main
        importlib.reload(api.main)
        
        with TestClient(api.main.app) as client:
            yield client

def test_health_check(success_client):
    response = success_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_student_status(success_client):
    student_data = {
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
    
    response = success_client.post("/predict", json=student_data)
    
    assert response.status_code == 200
    assert response.json() == {"prediction": "Graduate"}

def test_predict_student_status_error(error_client):
    student_data = {
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
    response = error_client.post("/predict", json=student_data)
    assert response.status_code == 500
