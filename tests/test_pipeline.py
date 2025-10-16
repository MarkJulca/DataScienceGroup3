import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, call

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.inference import Predictor
from src.pipeline.training import TabularTrainer

@pytest.fixture
def sample_training_data():
    data = {
        'Curricular units 1st sem (enrolled)': [6, 5, 7, 8, 6],
        'Curricular units 1st sem (evaluations)': [8, 5, 9, 8, 7],
        'Curricular units 1st sem (approved)': [5, 0, 6, 8, 5],
        'Curricular units 1st sem (grade)': [12.5, 0.0, 13.0, 15.0, 11.5],
        'Curricular units 2nd sem (enrolled)': [6, 5, 7, 8, 6],
        'Curricular units 2nd sem (evaluations)': [7, 0, 8, 9, 6],
        'Curricular units 2nd sem (approved)': [4, 0, 5, 8, 4],
        'Curricular units 2nd sem (grade)': [11.8, 0.0, 12.5, 14.8, 10.8],
        'Age at enrollment': [19, 22, 20, 18, 25],
        'Scholarship holder': [1, 0, 1, 0, 1],
        'Gender': [0, 1, 0, 1, 0],
        "Mother's qualification": [1, 1, 2, 1, 3],
        "Father's qualification": [1, 1, 2, 1, 3],
        "Mother's occupation": [5, 10, 3, 4, 2],
        "Father's occupation": [5, 10, 3, 4, 2],
        'Displaced': [0, 1, 0, 0, 0],
        'Educational special needs': [0, 0, 0, 0, 0],
        'Tuition fees up to date': [1, 0, 1, 1, 1],
        'Debtor': [0, 1, 0, 0, 0],
        'Daytime/evening attendance': [1, 1, 1, 1, 1],
        'Previous qualification': [1, 1, 1, 1, 1],
        'Target': [1, 0, 1, 1, 0]  # 1 for Graduate, 0 for Dropout
    }
    return pd.DataFrame(data)

@patch('joblib.load')
@patch('pandas.read_csv')
def test_predictor_init(mock_read_csv, mock_joblib_load, sample_training_data):
    mock_read_csv.return_value = sample_training_data
    mock_model = MagicMock()
    mock_joblib_load.return_value = mock_model

    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        predictor = Predictor(model_path='dummy_model.pkl', training_data_path='dummy_data.csv')

    assert predictor.model is not None
    assert predictor.scaler is not None

@patch('joblib.load')
@patch('pandas.read_csv')
def test_predictor_predict(mock_read_csv, mock_joblib_load, sample_training_data):
    mock_read_csv.return_value = sample_training_data
    mock_model = MagicMock()
    mock_model.predict.return_value = [1, 0]
    mock_joblib_load.return_value = mock_model

    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        predictor = Predictor(model_path='dummy_model.pkl', training_data_path='dummy_data.csv')

    input_data = sample_training_data.drop(columns=['Target']).iloc[:2]
    predictions = predictor.predict(input_data)

    assert isinstance(predictions, list)
    assert predictions == ['Graduate', 'Dropout']

@patch('src.pipeline.training.preprocess_for_model')
def test_tabular_trainer_train_all(mock_preprocess, sample_training_data):
    X = sample_training_data.drop(columns=['Target'])
    y = sample_training_data['Target']
    mock_preprocess.return_value = (X, y, X, y, MagicMock())

    trainer = TabularTrainer(data_path='dummy_data.csv')
    trained_models = trainer.train_all()

    assert "logistic" in trained_models
    assert "random_forest" in trained_models
    assert "decision_tree" in trained_models
    assert "logistic" in trainer.metrics
    assert "random_forest" in trainer.metrics
    assert "decision_tree" in trainer.metrics

@patch('src.pipeline.training.preprocess_for_model')
@patch('joblib.dump')
@patch('os.makedirs')
def test_tabular_trainer_select_and_save_final_model(mock_makedirs, mock_joblib_dump, mock_preprocess, sample_training_data):
    X = sample_training_data.drop(columns=['Target'])
    y = sample_training_data['Target']
    mock_preprocess.return_value = (X, y, X, y, MagicMock())

    trainer = TabularTrainer(data_path='dummy_data.csv', model_save_path='saved_model.pkl')
    trainer.train_all()
    
    assert trainer.models.get("random_forest") is not None

    model, path = trainer.select_and_save_final_model(chosen_model_name="random_forest")

    assert model is not None
    assert path == 'saved_model.pkl'
    mock_joblib_dump.assert_called_once()

# --- Tests for error handling in Predictor ---

@patch('joblib.load')
def test_predictor_init_model_not_found(mock_joblib_load):
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError, match="El archivo del modelo no se encontró"):
            Predictor(model_path='non_existent_model.pkl')

@patch('joblib.load')
@patch('pandas.read_csv')
def test_predictor_init_scaler_data_not_found(mock_read_csv, mock_joblib_load):
    with patch('os.path.exists', side_effect=[True, False]):
        with pytest.raises(FileNotFoundError, match="El archivo de datos de entrenamiento no se encontró"):
            Predictor(model_path='dummy_model.pkl', training_data_path='non_existent_data.csv')

@patch('joblib.load')
@patch('pandas.read_csv')
def test_predictor_init_missing_columns_in_scaler_data(mock_read_csv, mock_joblib_load, sample_training_data):
    incomplete_data = sample_training_data.drop(columns=['Age at enrollment'])
    mock_read_csv.return_value = incomplete_data
    with patch('os.path.exists', return_value=True):
        with pytest.raises(ValueError, match="Faltan columnas en el dataset de entrenamiento"):
            Predictor(model_path='dummy_model.pkl', training_data_path='dummy_data.csv')

@patch('joblib.load')
@patch('pandas.read_csv')
def test_predictor_predict_invalid_input_type(mock_read_csv, mock_joblib_load, sample_training_data):
    mock_read_csv.return_value = sample_training_data
    with patch('os.path.exists', return_value=True):
        predictor = Predictor(model_path='dummy_model.pkl', training_data_path='dummy_data.csv')
    
    with pytest.raises(TypeError, match="El tipo de dato de entrada debe ser un pd.DataFrame o un diccionario"):
        predictor.predict([1, 2, 3])

@patch('joblib.load')
@patch('pandas.read_csv')
def test_predictor_predict_missing_columns(mock_read_csv, mock_joblib_load, sample_training_data):
    mock_read_csv.return_value = sample_training_data
    with patch('os.path.exists', return_value=True):
        predictor = Predictor(model_path='dummy_model.pkl', training_data_path='dummy_data.csv')
    
    incomplete_input = sample_training_data.drop(columns=['Target', 'Age at enrollment']).iloc[:1]
    with pytest.raises(ValueError, match="Faltan las siguientes columnas en los datos de entrada"):
        predictor.predict(incomplete_input)

# --- Tests for TabularTrainer --- 

@patch('src.pipeline.training.preprocess_for_model')
def test_tabular_trainer_train_all_with_params(mock_preprocess, sample_training_data):
    X = sample_training_data.drop(columns=['Target'])
    y = sample_training_data['Target']
    mock_preprocess.return_value = (X, y, X, y, MagicMock())

    trainer = TabularTrainer(data_path='dummy_data.csv')
    params = {"logistic": {"C": 0.5}, "random_forest": {"n_estimators": 50}}
    trainer.train_all(classifier_params=params)

    assert trainer.models["logistic"].C == 0.5
    assert trainer.models["random_forest"].n_estimators == 50

@patch('src.pipeline.training.preprocess_for_model')
@patch('joblib.dump')
@patch('os.makedirs')
def test_tabular_trainer_select_and_save_final_model_no_choice(mock_makedirs, mock_joblib_dump, mock_preprocess, sample_training_data):
    X = sample_training_data.drop(columns=['Target'])
    y = sample_training_data['Target']
    mock_preprocess.return_value = (X, y, X, y, MagicMock())

    trainer = TabularTrainer(data_path='dummy_data.csv', model_save_path='saved_model.pkl')
    trainer.train_all()
    
    model, path = trainer.select_and_save_final_model(chosen_model_name=None)

    assert model is not None
    assert path == 'saved_model.pkl'
    mock_joblib_dump.assert_called_once_with(trainer.models["random_forest"], 'saved_model.pkl')

@patch('src.pipeline.training.preprocess_for_model')
def test_tabular_trainer_select_and_save_final_model_invalid_name(mock_preprocess, sample_training_data):
    X = sample_training_data.drop(columns=['Target'])
    y = sample_training_data['Target']
    mock_preprocess.return_value = (X, y, X, y, MagicMock())

    trainer = TabularTrainer(data_path='dummy_data.csv')
    trainer.train_all()

    with pytest.raises(KeyError, match="not found among trained models"):
        trainer.select_and_save_final_model(chosen_model_name="invalid_model")

@patch('src.pipeline.training.preprocess_for_model')
@patch('joblib.dump')
@patch('os.makedirs')
def test_tabular_trainer_run_training_pipeline(mock_makedirs, mock_joblib_dump, mock_preprocess, sample_training_data):
    X = sample_training_data.drop(columns=['Target'])
    y = sample_training_data['Target']
    mock_preprocess.return_value = (X, y, X, y, MagicMock())

    trainer = TabularTrainer(data_path='dummy_data.csv', model_save_path='saved_model.pkl')
    model, path = trainer.run_training_pipeline()

    assert model is not None
    assert path == 'saved_model.pkl'
    mock_joblib_dump.assert_called_once()