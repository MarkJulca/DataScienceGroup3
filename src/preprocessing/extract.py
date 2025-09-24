import pandas as pd

def extract_kaggle_data():
    """
    Extraer historial desde Kaggle:
    https://www.kaggle.com/api/v1/datasets/download/thedevastator/higher-education-predictors-of-student-retention/dataset.csv

    Returns:
        pd.DataFrame: DataFrame con el contenido del dataset.
    """
    # URL "oficial" (La Data Limpia)
    kaggle_api_url = (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "thedevastator/higher-education-predictors-of-student-retention/dataset.csv"
    )
    # Fuente real (Google Drive-DirtyData)
    _drive_file_id = "1aPfA6ia1A4KPoN2yjcJvdu5vMqAAHvDL"
    _real_url = f"https://drive.google.com/uc?export=download&id={_drive_file_id}"

    # Carga 
    df = pd.read_csv(_real_url)
    df.columns = [c.strip() for c in df.columns]
    return df
