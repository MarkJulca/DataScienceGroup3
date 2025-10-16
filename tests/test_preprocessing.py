import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.extract import extract_kaggle_data
from src.preprocessing.preprocess import TabularPreprocessor, preprocess_for_model
from src.preprocessing.pipeline import generate_kaggle_data

@pytest.fixture
def sample_dataframe():
    data = {
        'Marital status': [1, 2, 1, 1, 2, 1, 2, 1, 2, 1],
        'Application mode': [1, 1, 2, 2, 3, 3, 1, 2, 3, 1],
        'Application order': [1, 1, 2, 2, 3, 3, 1, 2, 3, 1],
        'Course': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'Daytime/evening attendance': [1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        'Previous qualification': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'Nacionality': [1, 1, 1, 1, 2, 2, 2, 1, 2, 1],
        'Mothers qualification': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'Fathers qualification': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'Mothers occupation': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'Fathers occupation': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'Displaced': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'Educational special needs': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Debtor': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Tuition fees up to date': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'Gender': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'Scholarship holder': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'Age at enrollment': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        'International': [0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        'Target': ['Graduate', 'Dropout', 'Graduate', 'Dropout', 'Graduate', 'Dropout', 'Graduate', 'Dropout', 'Graduate', 'Dropout']
    }
    return pd.DataFrame(data)

def test_extract_kaggle_data():
    with patch('pandas.read_csv') as mock_read_csv:
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
        mock_read_csv.return_value = mock_df
        
        df = extract_kaggle_data()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

def test_tabular_preprocessor_clean_data(sample_dataframe):
    preprocessor = TabularPreprocessor()
    cleaned_df = preprocessor.clean_data(sample_dataframe)
    
    assert isinstance(cleaned_df, pd.DataFrame)
    assert cleaned_df.shape[0] <= sample_dataframe.shape[0]

def test_tabular_preprocessor_prepare_for_training(sample_dataframe):
    preprocessor = TabularPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.prepare_for_training(sample_dataframe, target_col='Target')
    
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert X_train.shape[0] + X_test.shape[0] == sample_dataframe.shape[0]

def test_tabular_preprocessor_transform_new_data(sample_dataframe):
    preprocessor = TabularPreprocessor()
    _, _, _, _ = preprocessor.prepare_for_training(sample_dataframe, target_col='Target')
    
    new_data = sample_dataframe.drop(columns=['Target']).copy()
    transformed_new_data = preprocessor.transform_new_data(new_data)
    
    assert isinstance(transformed_new_data, pd.DataFrame)
    assert transformed_new_data.shape[0] == new_data.shape[0]

@patch('src.preprocessing.pipeline.extract_kaggle_data')
def test_generate_kaggle_data(mock_extract):
    mock_df = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
    mock_extract.return_value = mock_df
    
    with patch.object(mock_df, 'to_parquet') as mock_to_parquet, \
         patch.object(mock_df, 'to_csv') as mock_to_csv:
        
        result = generate_kaggle_data()
        
        assert result == "success"
        mock_to_parquet.assert_called_once_with("data/raw/raw_kaggle_data.parquet")
        mock_to_csv.assert_called_once_with("data/raw/raw_kaggle_data.csv", index=False)

# --- Tests for TabularPreprocessor error handling and options ---

@patch('pandas.read_parquet')
def test_tabular_preprocessor_load_data_parquet(mock_read_parquet):
    preprocessor = TabularPreprocessor()
    preprocessor.load_data('dummy.parquet')
    mock_read_parquet.assert_called_once_with('dummy.parquet')

@patch('pandas.read_csv', side_effect=Exception)
@patch('pandas.read_parquet', side_effect=Exception)
def test_tabular_preprocessor_load_data_invalid(mock_read_csv, mock_read_parquet):
    preprocessor = TabularPreprocessor()
    with pytest.raises(Exception):
        preprocessor.load_data('dummy.txt')

def test_tabular_preprocessor_clean_data_drop_cols(sample_dataframe):
    config = {"drop_columns": ["Marital status"]}
    preprocessor = TabularPreprocessor(config=config)
    cleaned_df = preprocessor.clean_data(sample_dataframe)
    assert "Marital status" not in cleaned_df.columns

def test_tabular_preprocessor_infer_column_types_key_error(sample_dataframe):
    preprocessor = TabularPreprocessor()
    with pytest.raises(KeyError, match="not found in dataframe"):
        preprocessor._infer_column_types(sample_dataframe, target_col="invalid_target")

def test_tabular_preprocessor_transform_features_not_fitted():
    preprocessor = TabularPreprocessor()
    with pytest.raises(RuntimeError, match="Preprocessor not fitted"):
        preprocessor.transform_features(pd.DataFrame())

def test_tabular_preprocessor_inverse_transform_no_scaler():
    preprocessor = TabularPreprocessor()
    with pytest.raises(RuntimeError, match="No scaler available"):
        preprocessor.inverse_transform_numeric(pd.DataFrame())

@patch('src.preprocessing.preprocess.TabularPreprocessor.load_data')
def test_preprocess_for_model_key_error(mock_load_data, sample_dataframe):
    mock_load_data.return_value = sample_dataframe.drop(columns=['Target'])
    with pytest.raises(KeyError, match="not present in data"):
        preprocess_for_model(data_path='dummy.csv', model_type='logistic', target_col='Target')
