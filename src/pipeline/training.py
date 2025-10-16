import os
import sys
import warnings
from typing import Any, Dict, Optional, Tuple

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

# add project src to path (so preprocessing.preprocess can be imported)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import your TabularPreprocessor and convenience wrapper
try:
    from preprocessing.preprocess import (
        TabularPreprocessor,
        preprocess_for_model,
    )
except Exception:
    # If direct import fails, try alternative relative import (depends on layout)
    from preprocess import TabularPreprocessor, preprocess_for_model  # type: ignore

# silence warnings (production style)
warnings.filterwarnings("ignore")


class TabularTrainer:
    """
    Trainer for tabular classification models.

    Trains LogisticRegression, RandomForest and DecisionTree.
    Saves final chosen model (RandomForest) to disk.

    Silent mode: no prints.
    """

    def __init__(
        self,
        data_path: str,
        target_col: str = "Target",
        test_size: float = 0.2,
        random_state: int = 42,
        model_save_path: str = "../src/modelo/modelo_training.pkl",
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.model_save_path = model_save_path

        # will hold trained models
        self.models: Dict[str, Any] = {}
        # store multiple metrics: accuracy, f1_score, precision, recall
        self.metrics: Dict[str, Dict[str, float]] = {}
        # store best model name based on F1-Score
        self.best_model_name: Optional[str] = None

        # Preprocessor instance will be returned by preprocess_for_model
        self.preprocessor: Optional[TabularPreprocessor] = None

    def _get_default_classifiers(self, rf_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Build classifier instances with sensible defaults.
        RandomForest uses random_state for reproducibility.
        """
        if rf_params is None:
            rf_params = {"n_estimators": 100, "random_state": self.random_state}

        return {
            "logistic": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "random_forest": RandomForestClassifier(**rf_params),
            "decision_tree": DecisionTreeClassifier(random_state=self.random_state),
        }

    def train_all(self, classifier_params: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Train all classifiers (Logistic, RandomForest, DecisionTree).

        Returns dictionary with trained estimators keyed by model name.
        """
        # Use preprocess_for_model convenience wrapper which returns transformed splits
        X_train, y_train, X_test, y_test, preprocessor = preprocess_for_model(
            data_path=self.data_path,
            model_type="random_forest",  # not used by preprocess_for_model for branching here
            target_col=self.target_col,
            test_size=self.test_size,
            random_state=self.random_state,
            config=None,
        )

        # assign preprocessor for external use if required
        self.preprocessor = preprocessor

        # classifier instances
        params = classifier_params or {}
        classifiers = self._get_default_classifiers(rf_params=params.get("random_forest"))

        # Train each model and store multiple metrics
        for name, clf in classifiers.items():
            # Allow passing kwargs for specific models
            model_params = params.get(name)
            if model_params:
                # re-initialize with user params if provided
                if name == "logistic":
                    clf = LogisticRegression(**model_params)
                elif name == "random_forest":
                    clf = RandomForestClassifier(**model_params)
                elif name == "decision_tree":
                    clf = DecisionTreeClassifier(**model_params)

            # Fit
            clf.fit(X_train, y_train)

            # Predict on test set and calculate multiple metrics
            y_pred = clf.predict(X_test)

            # Calculate all metrics (same as kaggle_data_cleaning.ipynb)
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            }

            # Store
            self.models[name] = clf
            self.metrics[name] = metrics

        # Identify best model by F1-Score (trazabilidad con kaggle_data_cleaning.ipynb)
        self.best_model_name = max(self.metrics.keys(), key=lambda k: self.metrics[k]['f1_score'])

        return self.models

    def select_and_save_final_model(self, chosen_model_name: Optional[str] = None) -> Tuple[Any, str]:
        """
        Select final model and save it to disk.
        If chosen_model_name is None, uses Random Forest (for consistency with kaggle_data_cleaning.ipynb).
        Returns the saved model and path.
        """
        # Force Random Forest to maintain consistency with Kaggle analysis
        # (Random Forest was the best model in the original analysis)
        if chosen_model_name is None:
            chosen_model_name = "random_forest"

        if chosen_model_name not in self.models:
            raise KeyError(f"Model '{chosen_model_name}' not found among trained models. Trained models: {list(self.models.keys())}")

        model = self.models[chosen_model_name]

        # Ensure output directory exists
        save_dir = os.path.dirname(self.model_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Save model with joblib (good for sklearn objects)
        joblib.dump(model, self.model_save_path)

        return model, self.model_save_path

    def run_training_pipeline(self, classifier_params: Optional[Dict[str, Dict]] = None) -> Tuple[Any, str]:
        """
        End-to-end: train all models, select Random Forest, save and return it and its path.
        Random Forest is chosen to maintain consistency with kaggle_data_cleaning.ipynb analysis.
        """
        self.train_all(classifier_params=classifier_params)
        # Select Random Forest (forced for consistency with original Kaggle analysis)
        final_model, path = self.select_and_save_final_model(chosen_model_name="random_forest")
        return final_model, path


# Minimal script entrypoint (silent)
if __name__ == "__main__":  # pragma: no cover
    # Configuration according to your instructions
    DATA_PATH = "../src/data/cleaned_raw/dataset_cleaned_final.csv"
    SAVE_PATH = "../src/modelo/modelo_training.pkl"
    TARGET_COL = "Target"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    trainer = TabularTrainer(
        data_path=DATA_PATH,
        target_col=TARGET_COL,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        model_save_path=SAVE_PATH,
    )

    # Run pipeline and save Random Forest as final model
    model, saved_path = trainer.run_training_pipeline()
    # silent: do not print anything
    # but allow external import to examine trainer.models and trainer.metrics if needed
