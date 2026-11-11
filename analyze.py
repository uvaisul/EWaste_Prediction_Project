"""
analyze.py
----------
Main execution script for Exploratory Data Analysis (EDA) and Visualization.

Generates the 5 project visualizations:
1. E-waste generation by year (Bar chart)
2. State-wise e-waste comparison (Horizontal ranking chart)
3. E-waste recycling comparison (Grouped multi-year comparison)
4. Recycling percentage (Annual national recycling rate & generation vs. collection)
5. E-waste generation trend (Time-series trajectory with YoY growth rates)

Saves all figures into the `graphs/` directory.
"""

from pathlib import Path
import pandas as pd
from src.utils import get_project_root, setup_logger
from src.eda import (
    plot_generation_by_year,
    plot_statewise_comparison,
    plot_recycling_comparison,
    plot_recycling_percentage,
    plot_generation_trend,
)

logger = setup_logger("AnalyzeRunner")


def run_analysis():
    print("=" * 75)
    print("      E-WASTE PROJECT: DATA ANALYSIS & VISUALIZATION PIPELINE")
    print("=" * 75)

    root = get_project_root()
    results_dir = root / "results"

    # 1. Load preprocessed datasets
    national_csv = results_dir / "cleaned_national_ewaste.csv"
    collection_csv = results_dir / "cleaned_state_collection.csv"

    if not national_csv.exists() or not collection_csv.exists():
        print("[WARNING] Cleaned data not found in 'results/'. Running preprocessing first...")
        from preprocess import run_preprocessing_pipeline
        run_preprocessing_pipeline()

    print("\n[Step 1] Loading preprocessed datasets from 'results/'...")
    df_national = pd.read_csv(national_csv)
    df_collection = pd.read_csv(collection_csv)

    print(f"-> National data loaded: {len(df_national)} records ({df_national['financial_year'].min()} to {df_national['financial_year'].max()})")
    print(f"-> State collection data loaded: {len(df_collection)} States & UTs")

    # 2. Generate Visualizations
    print("\n[Step 2] Generating Visualizations...")

    # Visualization 1: Generation by Year
    g1 = plot_generation_by_year(df_national)
    print(f"   [1/5] Generated: {g1.name}")

    # Visualization 2: State-wise Comparison
    g2 = plot_statewise_comparison(df_collection, top_n=15)
    print(f"   [2/5] Generated: {g2.name}")

    # Visualization 3: Recycling Comparison Across Years
    g3 = plot_recycling_comparison(df_collection, top_n=10)
    print(f"   [3/5] Generated: {g3.name}")

    # Visualization 4: Recycling Percentage (Generation vs Collection Rate)
    g4 = plot_recycling_percentage(df_national, df_collection)
    print(f"   [4/5] Generated: {g4.name}")

    # Visualization 5: Generation Trend with YoY Growth
    g5 = plot_generation_trend(df_national)
    print(f"   [5/5] Generated: {g5.name}")

    print("\n" + "=" * 75)
    print("SUCCESS: All 5 Visualizations Successfully Saved in 'graphs/' Folder!")
    print("=" * 75)


if __name__ == "__main__":
    run_analysis()
