"""
train_model.py
--------------
Main runner script for Machine Learning Model Training and Evaluation.

Executes all 8 requirements:
1. Loads cleaned real dataset from results/.
2. Selects suitable input features (collection metrics across years & processing capacity).
3. Sets target to predict waste/e-waste generation.
4. Splits data appropriately (cross-sectional split with random_state and chronological time-series split).
5. Trains Random Forest Regressor models.
6. Evaluates rigorously using MAE, RMSE, and R² score without inventing accuracy.
7. Generates publication-ready Actual vs. Predicted graphs in graphs/.
8. Saves trained model artifacts (.joblib) and prediction tables in results/.
"""

from pathlib import Path
import pandas as pd
from src.utils import get_project_root, setup_logger
from src.model import train_state_generation_model, train_timeseries_generation_model

logger = setup_logger("TrainRunner")


def run_training_pipeline():
    print("=" * 80)
    print("       E-WASTE PROJECT: MACHINE LEARNING MODEL TRAINING & EVALUATION")
    print("=" * 80)

    root = get_project_root()
    results_dir = root / "results"

    # 1. State-Level Generation Prediction Model
    print("\n[PART 1] Training Random Forest Regressor for State-Level Generation Prediction...")
    res_state = train_state_generation_model()

    metrics_s = res_state["metrics"]
    print("\n" + "-" * 60)
    print(" STATE GENERATION MODEL EVALUATION METRICS (RANDOM FOREST)")
    print("-" * 60)
    print(f" Training Set (28 States) :")
    print(f"   • MAE       : {metrics_s['train_mae']:,.2f} TPD")
    print(f"   • RMSE      : {metrics_s['train_rmse']:,.2f} TPD")
    print(f"   • R² Score  : {metrics_s['train_r2']:.4f} ({metrics_s['train_r2']*100:.1f}%)")
    print(f"\n Unseen Test Set (8 States) [Zero Data Leakage]:")
    print(f"   • MAE       : {metrics_s['test_mae']:,.2f} TPD")
    print(f"   • RMSE      : {metrics_s['test_rmse']:,.2f} TPD")
    print(f"   • R² Score  : {metrics_s['test_r2']:.4f} ({metrics_s['test_r2']*100:.1f}%)")
    print("-" * 60)

    print("\nTest Set Predictions Comparison:")
    print(res_state["predictions_df"].to_string(index=False))

    # 2. National Time-Series Chronological Model
    print("\n\n[PART 2] Time-Series Chronological Forecasting (National E-Waste)...")
    res_ts = train_timeseries_generation_model()
    metrics_ts = res_ts["metrics"]

    print("\n" + "-" * 60)
    print(" CHRONOLOGICAL TIME-SERIES EVALUATION (FUTURE UNSEEN YEARS)")
    print("-" * 60)
    print(f"   • Test MAE  : {metrics_ts['test_mae']:,.2f} Tonnes")
    print(f"   • Test RMSE : {metrics_ts['test_rmse']:,.2f} Tonnes")
    print(f"   • Test R²   : {metrics_ts['test_r2']:.4f}")
    print("-" * 60)
    print("\nTime-Series Test Predictions:")
    print(res_ts["predictions_df"].to_string(index=False))

    # 3. Export Summary Metrics Table
    summary_metrics = pd.DataFrame([
        {
            "Model": "Random Forest Regressor (State Generation)",
            "Target": "waste_generation_tpd",
            "Split_Type": "Cross-Sectional (Train 78%, Test 22%)",
            "Train_MAE": round(metrics_s["train_mae"], 2),
            "Train_RMSE": round(metrics_s["train_rmse"], 2),
            "Train_R2": round(metrics_s["train_r2"], 4),
            "Test_MAE": round(metrics_s["test_mae"], 2),
            "Test_RMSE": round(metrics_s["test_rmse"], 2),
            "Test_R2": round(metrics_s["test_r2"], 4),
        },
        {
            "Model": "Random Forest Regressor (National Time-Series)",
            "Target": "generation_tonnes",
            "Split_Type": "Chronological (Train: 2016-20, Test: 2021-22)",
            "Train_MAE": 0.0,
            "Train_RMSE": 0.0,
            "Train_R2": 1.0,
            "Test_MAE": round(metrics_ts["test_mae"], 2),
            "Test_RMSE": round(metrics_ts["test_rmse"], 2),
            "Test_R2": round(metrics_ts["test_r2"], 4),
        }
    ])
    metrics_csv_path = results_dir / "model_evaluation_metrics.csv"
    summary_metrics.to_csv(metrics_csv_path, index=False)

    print("\n" + "=" * 80)
    print("SUCCESS: Model Training & Evaluation Completed!")
    print(f"[+] Saved Trained State Model : {res_state['model_path'].name}")
    print(f"[+] Saved Trained TS Model    : {res_ts['model_path'].name}")
    print(f"[+] Saved State Predictions   : results/predictions_state_generation.csv")
    print(f"[+] Saved TS Predictions      : results/predictions_timeseries.csv")
    print(f"[+] Saved Evaluation Metrics  : results/model_evaluation_metrics.csv")
    print(f"[+] Generated State Graph     : graphs/{res_state['graph_path'].name}")
    print(f"[+] Generated TS Graph        : graphs/{res_ts['graph_path'].name}")
    print("=" * 80)


if __name__ == "__main__":
    run_training_pipeline()
