from io import BytesIO
from contextlib import suppress
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Try en caso se use
try:
    from storage.s3_utils import S3Path, get_s3_client, is_s3_uri  # type: ignore
except Exception:
    # If not present, define a stub for is_s3_uri used below
    S3Path = None
    get_s3_client = None

    def is_s3_uri(uri: str) -> bool:
        return str(uri).lower().startswith("s3://")


class TabularPreprocessor:
    """
    Tabular preprocessing class for classification tasks.
    Silent mode (no prints). Stores fitted transformers as attributes.

    Main features:
      - load CSV / parquet / S3
      - basic cleaning (drop duplicates, trim whitespace)
      - imputation numeric/categorical
      - automatic categorical encoding:
          * OneHotEncoder for columns with cardinality <= auto_cardinality_threshold
          * OrdinalEncoder for columns with higher cardinality
      - scaling (StandardScaler) for numeric features
      - train/test split
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            config: optional dict to override defaults
        """
        self.config = config or self._get_default_config()

        # Transformers that will be fitted during preprocessing
        self.numeric_imputer: Optional[SimpleImputer] = None
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.onehot_encoder: Optional[OneHotEncoder] = None
        self.ordinal_encoder: Optional[OrdinalEncoder] = None
        self.column_transformer: Optional[ColumnTransformer] = None

        # Column lists discovered/fixed during fit
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.onehot_columns: List[str] = []
        self.ordinal_columns: List[str] = []

        self.is_fitted = False

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "numeric_imputer_strategy": "mean",
            "categorical_imputer_strategy": "most_frequent",
            "test_size": 0.2,
            "random_state": 42,
            "auto_cardinality_threshold": 10,  # <= -> OneHot, > -> Ordinal
            "drop_duplicates": True,
            "drop_columns": [],  # list of columns to drop if any
            "target_col": "Target",
            "scale_numeric": True,
            "fill_whitespace_with_nan": True,
        }

    # ---------------------------
    # Loading utilities
    # ---------------------------
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV, parquet or S3 (if helper available).

        Returns:
            pd.DataFrame indexed as default (no special index)
        """
        if is_s3_uri(file_path) and S3Path is not None and get_s3_client is not None:
            return self._load_data_from_s3(file_path)

        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            # attempt to read as csv then parquet fallback
            try:
                df = pd.read_csv(file_path)
            except Exception:
                df = pd.read_parquet(file_path)
        return df

    def _load_data_from_s3(self, uri: str) -> pd.DataFrame:
        """Load a dataset stored in S3 using project utilities."""
        s3_path = S3Path.from_uri(uri)
        client = get_s3_client()
        response = client.get_object(Bucket=s3_path.bucket, Key=s3_path.key)
        body = response["Body"].read()
        buffer = BytesIO(body)
        if s3_path.key.endswith(".parquet"):
            return pd.read_parquet(buffer)
        if s3_path.key.endswith(".csv"):
            return pd.read_csv(buffer)
        raise ValueError("Unsupported file format for S3 object: " + s3_path.key)

    # ---------------------------
    # Cleaning / basic preprocessing
    # ---------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning:
          - drop duplicates (config)
          - drop specified columns (config)
          - strip whitespace from string columns (optional)
          - replace empty strings with NaN
        """
        df = df.copy()

        if self.config.get("drop_duplicates", True):
            df = df.drop_duplicates()

        drop_cols = self.config.get("drop_columns", [])
        if drop_cols:
            cols_to_drop = [c for c in drop_cols if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

        if self.config.get("fill_whitespace_with_nan", True):
            str_cols = df.select_dtypes(include=["object", "string"]).columns
            for col in str_cols:
                # strip whitespace
                df[col] = df[col].astype("string").str.strip()
                # empty strings -> NA
                df[col] = df[col].replace("", pd.NA)

        return df

    # ---------------------------
    # Feature selection helpers
    # ---------------------------
    def _infer_column_types(self, df: pd.DataFrame, target_col: str) -> None:
        """
        Set numeric_columns and categorical_columns excluding target.
        """
        df = df.copy()
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe")

        # Exclude target column from features
        feature_df = df.drop(columns=[target_col])

        # Numeric: pandas numeric dtypes
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

        # For non-numeric but with few unique numeric-like values, try convert
        # but keep simple: treat object/string as categorical
        categorical_cols = [
            c for c in feature_df.columns if c not in numeric_cols
        ]

        self.numeric_columns = numeric_cols
        self.categorical_columns = categorical_cols

    # ---------------------------
    # Encoding / Imputation / Scaling (fit + transform)
    # ---------------------------
    def _build_transformers(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Build ColumnTransformer that applies:
          - numeric imputer + scaler (optional)
          - categorical imputer + (onehot or ordinal)
        Also sets self.onehot_columns and self.ordinal_columns based on cardinality threshold.
        """
        config = self.config
        threshold = int(config.get("auto_cardinality_threshold", 10))

        # Determine categorical split by cardinality
        onehot_cols = []
        ordinal_cols = []
        for col in self.categorical_columns:
            try:
                card = int(df[col].nunique(dropna=True))
            except Exception:
                card = threshold + 1
            if card <= threshold:
                onehot_cols.append(col)
            else:
                ordinal_cols.append(col)

        self.onehot_columns = onehot_cols
        self.ordinal_columns = ordinal_cols

        # Imputers
        numeric_imputer = SimpleImputer(strategy=config.get("numeric_imputer_strategy", "mean"))
        categorical_imputer = SimpleImputer(strategy=config.get("categorical_imputer_strategy", "most_frequent"))

        # Store imputers
        self.numeric_imputer = numeric_imputer
        self.categorical_imputer = categorical_imputer

        # Numeric pipeline
        numeric_steps = [("imputer", numeric_imputer)]
        if config.get("scale_numeric", True):
            numeric_steps.append(("scaler", StandardScaler()))
            self.scaler = StandardScaler()  # placeholder reference; actual scaler inside transformer

        numeric_pipeline = Pipeline(numeric_steps)

        transformers = []
        if self.numeric_columns:
            transformers.append(("num", numeric_pipeline, self.numeric_columns))

        # OneHot pipeline
        if onehot_cols:
            onehot = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.onehot_encoder = onehot
            onehot_pipeline = Pipeline([("imputer", categorical_imputer), ("onehot", onehot)])
            transformers.append(("onehot", onehot_pipeline, onehot_cols))

        # Ordinal pipeline
        if ordinal_cols:
            # OrdinalEncoder can use encoded_value for unknowns in newer sklearn; fallback: handle errors in transform
            ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.ordinal_encoder = ordinal
            ordinal_pipeline = Pipeline([("imputer", categorical_imputer), ("ordinal", ordinal)])
            transformers.append(("ordinal", ordinal_pipeline, ordinal_cols))

        column_transformer = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

        self.column_transformer = column_transformer
        return column_transformer

    def fit_transform_features(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit imputers/encoders/scalers on df and return transformed X (DataFrame) and y (Series).
        """
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found")

        self._infer_column_types(df, target_col)

        # Build transformers based on current df
        column_transformer = self._build_transformers(df)

        # Fit & transform
        feature_df = df.drop(columns=[target_col])
        target_series = df[target_col].copy()

        # Fit the column transformer
        transformed_array = column_transformer.fit_transform(feature_df)

        # Build feature names
        out_feature_names = []
        # Numeric names
        if self.numeric_columns:
            out_feature_names.extend(self.numeric_columns)
            # If scaler exists inside pipeline, it's applied to numeric columns but names remain numeric_columns

        # OneHot names
        if self.onehot_columns and self.onehot_encoder is not None:
            # get_feature_names_out requires fitted encoder
            oh_names = list(self.onehot_encoder.get_feature_names_out(self.onehot_columns))
            out_feature_names.extend(oh_names)

        # Ordinal names
        if self.ordinal_columns:
            out_feature_names.extend(self.ordinal_columns)

        # Construct DataFrame
        feature_df = pd.DataFrame(transformed_array, columns=out_feature_names, index=feature_df.index)

        # Save fitted encoders/scalers explicitly if necessary:
        # - self.numeric_imputer, self.categorical_imputer already set
        # - self.scaler: extract if present inside the pipeline
        # Attempt to find scaler instance within transformer
        if self.column_transformer:
            with suppress(Exception):
                if self.numeric_columns:
                    num_transformer = [t for name, t, cols in self.column_transformer.transformers_ if name == "num"]
                    if num_transformer:
                        # pipeline steps inside are accessible via named_steps
                        pipeline = num_transformer[0]
                        if "scaler" in pipeline.named_steps:
                            self.scaler = pipeline.named_steps["scaler"]

        self.is_fitted = True
        return feature_df, target_series

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using already-fitted transformers.
        """
        if not self.is_fitted or self.column_transformer is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform_features first.")

        feature_df = df.copy()
        transformed_array = self.column_transformer.transform(feature_df[self.numeric_columns + self.onehot_columns + self.ordinal_columns])
        # Reconstruct column names similar to fit
        out_feature_names = []
        if self.numeric_columns:
            out_feature_names.extend(self.numeric_columns)
        if self.onehot_columns and self.onehot_encoder is not None:
            out_feature_names.extend(list(self.onehot_encoder.get_feature_names_out(self.onehot_columns)))
        if self.ordinal_columns:
            out_feature_names.extend(self.ordinal_columns)
        feature_df = pd.DataFrame(transformed_array, columns=out_feature_names, index=feature_df.index)
        return feature_df

    # ---------------------------
    # Public convenience methods
    # ---------------------------

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Full fit+split. Fits on provided df and splits into train/test.
        Returns: X_train, y_train, X_test, y_test
        """
        cfg = self.config
        test_size = test_size if test_size is not None else cfg.get("test_size", 0.2)
        random_state = random_state if random_state is not None else cfg.get("random_state", 42)

        # Clean
        df_clean = self.clean_data(df)

        # Fit & transform features
        X_all, y_all = self.fit_transform_features(df_clean, target_col)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all if y_all.nunique() > 1 else None
        )

        return X_train, y_train, X_test, y_test

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing to new dataframe (no target)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted yet. Call prepare_for_training() first.")

        df_clean = self.clean_data(df)
        feature_df = self.transform_features(df_clean)
        return feature_df

    # ---------------------------
    # Utility: inverse transform for ordinal and scaling (if needed)
    # ---------------------------
    def inverse_transform_numeric(self, arr: np.ndarray) -> np.ndarray:
        """
        Inverse transform numeric columns if scaler fitted.
        """
        if self.scaler is None:
            raise RuntimeError("No scaler available.")
        return self.scaler.inverse_transform(arr)

    # ---------------------------
    # Persistence helpers (optional)
    # ---------------------------
    def get_transformers(self) -> Dict[str, Any]:
        """
        Return fitted transformers and column lists for external use.
        """
        return {
            "column_transformer": self.column_transformer,
            "numeric_imputer": self.numeric_imputer,
            "categorical_imputer": self.categorical_imputer,
            "onehot_encoder": self.onehot_encoder,
            "ordinal_encoder": self.ordinal_encoder,
            "scaler": self.scaler,
            "numeric_columns": self.numeric_columns,
            "onehot_columns": self.onehot_columns,
            "ordinal_columns": self.ordinal_columns,
        }


# ---------------------------
# preprocess for model para svm,decision tree,random forest , regresion logistica
# ---------------------------
def preprocess_for_model(
    data_path: str,
    model_type: str,
    target_col: str = "Target",
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, TabularPreprocessor]:
    """
    Convenience wrapper: loads data, preprocesses, and returns train/test splits
    plus the fitted preprocessor instance.

    Args:
        data_path: path to dataset (csv/parquet or s3://)
        model_type: one of "logistic", "random_forest", "svm", "decision_tree" (used for potential future branching)
        target_col: name of target column in dataset
        test_size: optional override for test split
        random_state: optional override for RNG
        config: optional config dict for TabularPreprocessor

    Returns:
        X_train, y_train, X_test, y_test, preprocessor
    """
    # instantiate preprocessor
    preprocessor = TabularPreprocessor(config=config)

    # load data
    df = preprocessor.load_data(data_path)

    # Basic guard: ensure target exists
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in data at {data_path}")

    # Prepare data
    X_train, y_train, X_test, y_test = preprocessor.prepare_for_training(
        df, target_col, test_size=test_size, random_state=random_state
    )

    return X_train, y_train, X_test, y_test, preprocessor


# If module executed as script, provide a minimal run example (silent - returns nothing)
if __name__ == "__main__":  # pragma: no cover
    # Example usage (commented out because script is silent by default)
    # X_train, y_train, X_test, y_test, prep = preprocess_for_model("dataset_cleaned_final.csv", "logistic", target_col="Target")
    pass
# Retorno: preprocess_for_model() devuelve
# (X_train, y_train, X_test, y_test, preprocessor) para que puedas acceder
#  a transformadores y columnas cuando entrenes/guardes modelos.
