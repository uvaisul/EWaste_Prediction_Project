"""
model.py
--------
Module dedicated to Machine Learning model development, training, and evaluation
for E-Waste and Waste Generation Prediction.

Includes:
1. State-level Generation Prediction (Cross-sectional Random Forest Regressor).
2. National Time-Series Generation Forecasting (Strict chronological split).
3. Evaluation metrics: MAE, RMSE, and R² Score.
4. Publication-ready Actual vs. Predicted visualizations.
5. Model serialization (saving to results/*.joblib) and metric export.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import get_project_root, setup_logger

logger = setup_logger("ModelPipeline")

sns.set_theme(style="whitegrid", palette="muted")


def train_state_generation_model() -> Dict[str, Any]:
    """
    Trains a Random Forest Regressor to predict state-level waste/e-waste generation volume
    using multi-year collection metrics and processing capacity.

    Features:
        - collection_2019_20_tonnes
        - collection_2020_21_tonnes
        - collection_2021_22_tonnes
        - waste_processed_tpd
        - processing_efficiency_pct
    Target:
        - waste_generation_tpd

    Returns:
        Dictionary containing model, metrics, and train/test datasets.
    """
    root = get_project_root()
    dataset_path = root / "results" / "cleaned_ewaste_master_state.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Cleaned master dataset not found at: {dataset_path}")

    logger.info("Loading cleaned master state dataset for ML training...")
    df = pd.read_csv(dataset_path)

    features = [
        "collection_2019_20_tonnes",
        "collection_2020_21_tonnes",
        "collection_2021_22_tonnes",
        "waste_processed_tpd",
        "processing_efficiency_pct",
    ]
    target = "waste_generation_tpd"

    X = df[features]
    y = df[target]

    # Split into train (approx 78%) and test (approx 22%) with fixed random_state to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42
    )
    logger.info(f"Split dataset -> Training set: {X_train.shape[0]} states, Test set: {X_test.shape[0]} states")

    # Train Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=3,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)

    # Predictions
    preds_train = rf_model.predict(X_train)
    preds_test = rf_model.predict(X_test)

    # Calculate Evaluation Metrics
    mae_train = mean_absolute_error(y_train, preds_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
    r2_train = r2_score(y_train, preds_train)

    mae_test = mean_absolute_error(y_test, preds_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
    r2_test = r2_score(y_test, preds_test)

    logger.info(f"State Model Results -> Test MAE: {mae_test:.2f}, Test RMSE: {rmse_test:.2f}, Test R2: {r2_test:.4f}")

    # Build predictions comparison dataframe
    test_states = df.loc[y_test.index, "state_ut"].values
    df_preds = pd.DataFrame({
        "state_ut": test_states,
        "actual_generation_tpd": y_test.values,
        "predicted_generation_tpd": np.round(preds_test, 2),
        "absolute_error_tpd": np.round(np.abs(y_test.values - preds_test), 2),
        "percentage_error_pct": np.round(
            (np.abs(y_test.values - preds_test) / np.maximum(y_test.values, 1.0)) * 100, 2
        ),
    })

    # Save Model Artifacts
    model_save_path = root / "results" / "random_forest_state_model.joblib"
    joblib.dump(rf_model, model_save_path)
    logger.info(f"Saved trained model to: {model_save_path}")

    # Save Predictions CSV
    preds_save_path = root / "results" / "predictions_state_generation.csv"
    df_preds.to_csv(preds_save_path, index=False)
    logger.info(f"Saved test predictions to: {preds_save_path}")

    # Generate Actual vs Predicted Graph
    graph_path = plot_actual_vs_predicted_state(df_preds, r2_test, mae_test, rmse_test)

    return {
        "model": rf_model,
        "features": features,
        "metrics": {
            "train_mae": mae_train,
            "train_rmse": rmse_train,
            "train_r2": r2_train,
            "test_mae": mae_test,
            "test_rmse": rmse_test,
            "test_r2": r2_test,
        },
        "predictions_df": df_preds,
        "graph_path": graph_path,
        "model_path": model_save_path,
    }


def train_timeseries_generation_model() -> Dict[str, Any]:
    """
    Trains a time-series model on national e-waste generation using a strict chronological split.
    Prevents temporal data leakage by training only on past years (2016-17 to 2020-21)
    and testing on future unseen years (2021-22 and 2022-23).

    Returns:
        Dictionary containing chronological time-series results.
    """
    root = get_project_root()
    dataset_path = root / "results" / "cleaned_national_ewaste.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"National dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Chronological Split: First 5 years for training, last 2 years for testing
    train_df = df.iloc[:5].copy()
    test_df = df.iloc[5:].copy()

    X_train = train_df[["start_year"]]
    y_train = train_df["generation_tonnes"]
    X_test = test_df[["start_year"]]
    y_test = test_df["generation_tonnes"]

    rf_ts = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_ts.fit(X_train, y_train)

    preds_test = rf_ts.predict(X_test)
    preds_train = rf_ts.predict(X_train)

    mae_test = mean_absolute_error(y_test, preds_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
    r2_test = r2_score(y_test, preds_test)

    df_ts_preds = pd.DataFrame({
        "financial_year": test_df["financial_year"].values,
        "start_year": test_df["start_year"].values,
        "actual_generation_tonnes": y_test.values,
        "predicted_generation_tonnes": np.round(preds_test, 2),
        "absolute_error_tonnes": np.round(np.abs(y_test.values - preds_test), 2),
    })

    # Save model and predictions
    ts_model_path = root / "results" / "random_forest_timeseries_model.joblib"
    joblib.dump(rf_ts, ts_model_path)
    ts_preds_path = root / "results" / "predictions_timeseries.csv"
    df_ts_preds.to_csv(ts_preds_path, index=False)

    # Plot time-series actual vs predicted
    graph_path = plot_actual_vs_predicted_timeseries(df, train_df, test_df, preds_test)

    return {
        "model": rf_ts,
        "metrics": {
            "test_mae": mae_test,
            "test_rmse": rmse_test,
            "test_r2": r2_test,
        },
        "predictions_df": df_ts_preds,
        "graph_path": graph_path,
        "model_path": ts_model_path,
    }


def plot_actual_vs_predicted_state(
    df_preds: pd.DataFrame, r2: float, mae: float, rmse: float, filename: str = "6_actual_vs_predicted_state_generation.png"
) -> Path:
    """
    Plots Actual vs. Predicted values for the State-level Generation model.
    Includes the ideal y = x identity line, state name labels, and metric callout box.
    """
    root = get_project_root()
    save_path = root / "graphs" / filename

    plt.figure(figsize=(9.5, 7.5))

    actual = df_preds["actual_generation_tpd"]
    pred = df_preds["predicted_generation_tpd"]

    # Scatter plot
    plt.scatter(
        actual,
        pred,
        color="#1f77b4",
        edgecolor="#0d47a1",
        s=120,
        alpha=0.85,
        label="Test States",
        zorder=5,
    )

    # Ideal identity line y = x
    max_val = max(actual.max(), pred.max()) * 1.12
    min_val = 0
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Fit (y = x)")

    # Annotate state names on scatter points
    for _, row in df_preds.iterrows():
        plt.annotate(
            row["state_ut"],
            xy=(row["actual_generation_tpd"], row["predicted_generation_tpd"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            weight="medium",
        )

    plt.title("Random Forest Regression: Actual vs. Predicted Waste Generation\n(Evaluation on Unseen Test States)", weight="bold", pad=15)
    plt.xlabel("Actual Waste Generation (in Tonnes Per Day - TPD)", weight="bold", labelpad=10)
    plt.ylabel("Predicted Waste Generation (in Tonnes Per Day - TPD)", weight="bold", labelpad=10)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Performance metrics callout box
    metric_text = (
        f"Evaluation Metrics (Test Set):\n"
        f"• R² Score: {r2:.4f} ({r2*100:.1f}% Variance Explained)\n"
        f"• MAE     : {mae:,.1f} TPD\n"
        f"• RMSE    : {rmse:,.1f} TPD"
    )
    plt.gca().text(
        0.05,
        0.92,
        metric_text,
        transform=plt.gca().transAxes,
        fontsize=10.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95),
    )

    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved: {save_path.name}")
    return save_path


def plot_actual_vs_predicted_timeseries(
    df_all: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, preds_test: np.ndarray, filename: str = "7_actual_vs_predicted_timeseries.png"
) -> Path:
    """
    Plots the chronological time-series evaluation:
    Training historical period vs. Unseen future testing period (Actual vs Predicted).
    """
    root = get_project_root()
    save_path = root / "graphs" / filename

    plt.figure(figsize=(10.5, 6))

    # All actuals
    plt.plot(
        df_all["financial_year"],
        df_all["generation_tonnes"] / 1000.0,
        color="#2b5c8f",
        marker="o",
        linewidth=2.5,
        label="Actual Historical Generation",
    )

    # Training region shading
    train_years = train_df["financial_year"]
    plt.axvspan(train_years.iloc[0], train_years.iloc[-1], color="#e2e8f0", alpha=0.5, label="Chronological Training Period (2016-2020)")

    # Test predictions
    test_years = test_df["financial_year"]
    plt.plot(
        test_years,
        preds_test / 1000.0,
        color="#c0392b",
        marker="s",
        linestyle="--",
        linewidth=2.5,
        markersize=9,
        label="Random Forest Test Predictions (Unseen Future)",
    )

    plt.title("Time-Series E-Waste Generation: Chronological Train-Test Split Evaluation\n(Testing Random Forest on Unseen Future Financial Years)", weight="bold", pad=15)
    plt.xlabel("Financial Year", weight="bold", labelpad=10)
    plt.ylabel("E-Waste Generation (in Thousand Metric Tonnes)", weight="bold", labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved: {save_path.name}")
    return save_path
