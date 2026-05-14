"""
Model Evaluation & Reporting Module for Industrial Energy Optimizer
====================================================================

Comprehensive evaluation framework providing:
  - Regression metrics: R², MAE, RMSE, MAPE
  - Classification metrics: accuracy, precision, recall, F1, confusion matrix
  - Judge-friendly comparison reports and visualizations
  - Model performance summaries for dashboard integration
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)


def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (%)."""
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(
        100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    )


@dataclass
class RegressionMetrics:
    """Container for regression model evaluation metrics."""

    r2: float
    """R² score (coefficient of determination)."""
    mae: float
    """Mean Absolute Error."""
    rmse: float
    """Root Mean Squared Error."""
    mape: float
    """Mean Absolute Percentage Error (%)."""

    def to_dict(self) -> Dict[str, float]:
        """Export as dictionary for dashboard display."""
        return {
            "r2": round(self.r2, 4),
            "mae": round(self.mae, 2),
            "rmse": round(self.rmse, 2),
            "mape": round(self.mape, 2),
        }

    def summary_text(self) -> str:
        """Generate readable summary for reports."""
        return (
            f"R²={self.r2:.4f} | MAE={self.mae:.2f} MW | "
            f"RMSE={self.rmse:.2f} MW | MAPE={self.mape:.2f}%"
        )


@dataclass
class ClassificationMetrics:
    """Container for classification model evaluation metrics."""

    accuracy: float
    """Overall accuracy."""
    precision: float
    """Weighted precision."""
    recall: float
    """Weighted recall."""
    f1: float
    """Weighted F1 score."""
    confusion_matrix_array: np.ndarray
    """Raw confusion matrix."""
    classification_report_text: str
    """Sklearn classification report."""

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for dashboard display."""
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "confusion_matrix": self.confusion_matrix_array.tolist(),
        }

    def summary_text(self) -> str:
        """Generate readable summary for reports."""
        return (
            f"Accuracy={self.accuracy:.4f} | Precision={self.precision:.4f} | "
            f"Recall={self.recall:.4f} | F1={self.f1:.4f}"
        )


class RegressionEvaluator:
    """Evaluator for regression models (plant demand forecasting)."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
        """
        Comprehensive regression evaluation.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth power output (MW).
        y_pred : np.ndarray
            Predicted power output (MW).

        Returns
        -------
        RegressionMetrics
            Complete evaluation metrics.
        """
        return RegressionMetrics(
            r2=float(r2_score(y_true, y_pred)),
            mae=float(mean_absolute_error(y_true, y_pred)),
            rmse=_calculate_rmse(y_true, y_pred),
            mape=_calculate_mape(y_true, y_pred),
        )

    @staticmethod
    def comparison_table(
        models_dict: Dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Build comparison table for multiple regression models.

        Parameters
        ----------
        models_dict : Dict[str, Tuple[y_true, y_pred]]
            Dict mapping model names to (y_true, y_pred) tuples.
            Example: {"Random Forest": (y_test, y_pred_rf), "XGBoost": (y_test, y_pred_xgb)}

        Returns
        -------
        pd.DataFrame
            Comparison table with metrics for each model.
        """
        results = []
        for model_name, (y_true, y_pred) in models_dict.items():
            metrics = RegressionEvaluator.evaluate(y_true, y_pred)
            results.append({"Model": model_name, **metrics.to_dict()})
        return pd.DataFrame(results)


class ClassificationEvaluator:
    """Evaluator for classification models (machine anomaly detection)."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
        """
        Comprehensive classification evaluation.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth failure labels (0=normal, 1=failure/anomaly).
        y_pred : np.ndarray
            Predicted failure labels.

        Returns
        -------
        ClassificationMetrics
            Complete evaluation metrics.
        """
        return ClassificationMetrics(
            accuracy=float(np.mean(y_true == y_pred)),
            precision=float(
                precision_score(y_true, y_pred, zero_division=0, average="weighted")
            ),
            recall=float(
                recall_score(y_true, y_pred, zero_division=0, average="weighted")
            ),
            f1=float(f1_score(y_true, y_pred, zero_division=0, average="weighted")),
            confusion_matrix_array=confusion_matrix(y_true, y_pred),
            classification_report_text=classification_report(
                y_true, y_pred, zero_division=0
            ),
        )

    @staticmethod
    def comparison_table(
        models_dict: Dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Build comparison table for multiple classification models.

        Parameters
        ----------
        models_dict : Dict[str, Tuple[y_true, y_pred]]
            Dict mapping model names to (y_true, y_pred) tuples.

        Returns
        -------
        pd.DataFrame
            Comparison table with metrics for each model.
        """
        results = []
        for model_name, (y_true, y_pred) in models_dict.items():
            metrics = ClassificationEvaluator.evaluate(y_true, y_pred)
            results.append({"Model": model_name, **metrics.to_dict()})
        return pd.DataFrame(results)


@dataclass
class ModelEvaluationReport:
    """Complete evaluation report for all models."""

    task: str
    """Task name (e.g., 'Plant Demand Forecasting', 'Machine Anomaly Detection')."""
    models_evaluated: int
    """Number of models evaluated."""
    best_model: str
    """Name of best-performing model."""
    best_metric_value: float
    """Best metric value (R² for regression, F1 for classification)."""
    metrics_summary: Dict[str, Dict[str, float]]
    """Dict mapping model name to metric dict."""
    comparison_df: pd.DataFrame
    """Comparison dataframe."""
    evaluation_timestamp: str
    """Timestamp of evaluation."""

    def to_markdown(self) -> str:
        """Generate markdown report for documentation."""
        lines = [
            f"# {self.task} — Model Evaluation Report",
            "",
            f"**Evaluation Time**: {self.evaluation_timestamp}",
            f"**Models Evaluated**: {self.models_evaluated}",
            f"**Best Performer**: {self.best_model} ({self.best_metric_value:.4f})",
            "",
            "## Metrics Comparison",
            "",
            self.comparison_df.to_markdown(index=False),
            "",
            "## Key Insights",
            "",
            f"- **Best Model**: {self.best_model}",
            f"- **All models evaluated**: {', '.join(self.comparison_df['Model'].tolist())}",
            "",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "task": self.task,
            "models_evaluated": self.models_evaluated,
            "best_model": self.best_model,
            "best_metric_value": self.best_metric_value,
            "metrics_summary": self.metrics_summary,
            "comparison_table": self.comparison_df.to_dict(orient="records"),
        }


class EvaluationReporter:
    """Generate comprehensive evaluation reports."""

    @staticmethod
    def regression_report(
        models_dict: Dict[str, tuple[np.ndarray, np.ndarray]],
        task_name: str = "Plant Demand Forecasting",
        best_metric: str = "r2",
    ) -> ModelEvaluationReport:
        """
        Generate regression evaluation report.

        Parameters
        ----------
        models_dict : Dict[str, Tuple[y_true, y_pred]]
            Dict mapping model names to (y_true, y_pred) tuples.
        task_name : str
            Name of the task (for report header).
        best_metric : str
            Metric to use for ranking ("r2", "mae", "rmse", "mape").

        Returns
        -------
        ModelEvaluationReport
            Complete evaluation report.
        """
        from datetime import datetime

        comparison_df = RegressionEvaluator.comparison_table(models_dict)

        # Determine best model
        best_idx = comparison_df[best_metric].idxmax()
        best_model = comparison_df.loc[best_idx, "Model"]
        best_value = comparison_df.loc[best_idx, best_metric]

        metrics_summary = {}
        for _, row in comparison_df.iterrows():
            model_name = row["Model"]
            metrics_summary[model_name] = {
                k: v for k, v in row.items() if k != "Model"
            }

        return ModelEvaluationReport(
            task=task_name,
            models_evaluated=len(models_dict),
            best_model=best_model,
            best_metric_value=best_value,
            metrics_summary=metrics_summary,
            comparison_df=comparison_df,
            evaluation_timestamp=datetime.now().isoformat(),
        )

    @staticmethod
    def classification_report(
        models_dict: Dict[str, tuple[np.ndarray, np.ndarray]],
        task_name: str = "Machine Anomaly Detection",
        best_metric: str = "f1",
    ) -> ModelEvaluationReport:
        """
        Generate classification evaluation report.

        Parameters
        ----------
        models_dict : Dict[str, Tuple[y_true, y_pred]]
            Dict mapping model names to (y_true, y_pred) tuples.
        task_name : str
            Name of the task (for report header).
        best_metric : str
            Metric to use for ranking ("accuracy", "precision", "recall", "f1").

        Returns
        -------
        ModelEvaluationReport
            Complete evaluation report.
        """
        from datetime import datetime

        comparison_df = ClassificationEvaluator.comparison_table(models_dict)

        # Determine best model
        best_idx = comparison_df[best_metric].idxmax()
        best_model = comparison_df.loc[best_idx, "Model"]
        best_value = comparison_df.loc[best_idx, best_metric]

        metrics_summary = {}
        for _, row in comparison_df.iterrows():
            model_name = row["Model"]
            metrics_summary[model_name] = {
                k: v for k, v in row.items() if k != "Model"
            }

        return ModelEvaluationReport(
            task=task_name,
            models_evaluated=len(models_dict),
            best_model=best_model,
            best_metric_value=best_value,
            metrics_summary=metrics_summary,
            comparison_df=comparison_df,
            evaluation_timestamp=datetime.now().isoformat(),
        )


if __name__ == "__main__":
    # Example usage
    print(__doc__)

    # Example: regression comparison
    y_test = np.array([450, 460, 470, 480, 490, 500])
    y_pred_rf = np.array([452, 458, 472, 478, 492, 498])
    y_pred_xgb = np.array([451, 461, 469, 481, 489, 501])

    models = {
        "Random Forest": (y_test, y_pred_rf),
        "XGBoost": (y_test, y_pred_xgb),
    }

    report = EvaluationReporter.regression_report(models)
    print(report.to_markdown())
