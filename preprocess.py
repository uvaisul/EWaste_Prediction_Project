"""
preprocess.py
-------------
Standalone data preprocessing runner for the E-Waste Prediction & Recycling Analysis project.

Executes all 6 required preprocessing steps:
1. Loads original dataset ('Dataset/E-WASTE VALUE.xlsx') without modifying it.
2. Handles missing values appropriately (e.g. median imputation with indicator).
3. Detects and removes duplicate records.
4. Detects and cleans incorrect/unusual values (hidden characters, commas, trailing spaces).
5. Standardizes column names and converts data types to numeric floats/integers.
6. Saves cleaned CSV and Excel datasets in the 'results/' folder.
"""

from pathlib import Path
import pandas as pd
from src.utils import get_project_root, ensure_project_directories, setup_logger
from src.data_loader import (
    load_raw_workbook,
    preprocess_national_generation,
    preprocess_state_collection,
    preprocess_top_states_generation,
    preprocess_state_waste_2023,
    create_integrated_state_master,
)

logger = setup_logger("PreprocessRunner")


def run_preprocessing_pipeline():
    print("=" * 75)
    print("      E-WASTE PROJECT: DATA PREPROCESSING PIPELINE")
    print("=" * 75)

    # Step 0: Ensure folder structure exists
    root = get_project_root()
    ensure_project_directories()
    results_dir = root / "results"

    # Step 1: Load original dataset (read-only)
    print("\n[Step 1] Loading original dataset from 'Dataset/E-WASTE VALUE.xlsx'...")
    xl = load_raw_workbook("E-WASTE VALUE.xlsx")
    if xl is None:
        print("[ERROR] Could not load dataset. Make sure 'E-WASTE VALUE.xlsx' is in 'Dataset/'")
        return

    print(f"-> Successfully loaded workbook with sheets: {xl.sheet_names}")
    print("-> Original dataset remains 100% UNCHANGED and intact.")

    # Step 2 & 5: Preprocessing Sheet 1 (National Trend)
    print("\n[Step 2] Preprocessing Sheet 1: National E-Waste Generation (2016-2023)...")
    df_national = preprocess_national_generation(xl)
    print(f"   - Total records: {len(df_national)}")
    print(f"   - Missing values: {df_national.isnull().sum().sum()}")
    print(f"   - Duplicates: {df_national.duplicated().sum()}")
    print(f"   - Converted dtypes: generation_tonnes ({df_national['generation_tonnes'].dtype}), start_year ({df_national['start_year'].dtype})")

    # Step 3 & 5: Preprocessing Sheet 2 (State Collection)
    print("\n[Step 3] Preprocessing Sheet 2: State-wise Collection (2019-2022)...")
    df_collection = preprocess_state_collection(xl)
    print(f"   - Total records: {len(df_collection)} States & UTs")
    print(f"   - Missing values: {df_collection.isnull().sum().sum()}")
    print(f"   - Duplicates: {df_collection.duplicated(subset=['state_ut']).sum()}")
    print(f"   - Added aggregations: 'total_collected_3yr_tonnes' & 'avg_annual_collection_tonnes'")

    # Step 4: Preprocessing Sheet 3 (Top States Generation & Missing Value Handling)
    print("\n[Step 4] Preprocessing Sheet 3: Top States Generation (2018 vs 2023)...")
    df_top_states, median_2023 = preprocess_top_states_generation(xl)
    print(f"   - Total records: {len(df_top_states)} States")
    print(f"   - Cleaned string formatting: Removed commas and non-breaking spaces (e.g. Delhi '200,000')")
    print(f"   - Handled missing value for Punjab in 2023 via median imputation ({median_2023:,.2f} tonnes)")
    print(f"   - Added 'is_2023_imputed' boolean column for transparent documentation")
    print(f"   - Missing values remaining: {df_top_states.isnull().sum().sum()}")

    # Step 5: Preprocessing Sheet 4 (State Municipal Waste & Efficiency)
    print("\n[Step 5] Preprocessing Sheet 4: State Waste Generation 2023 (TPD)...")
    df_waste_2023 = preprocess_state_waste_2023(xl)
    print(f"   - Total records: {len(df_waste_2023)} States & UTs")
    print(f"   - Missing values: {df_waste_2023.isnull().sum().sum()}")
    print(f"   - Duplicates: {df_waste_2023.duplicated(subset=['state_ut']).sum()}")
    print(f"   - Computed metric: 'processing_efficiency_pct' = (Processed / Generated) * 100")

    # Step 6: Create Integrated Master Dataset
    print("\n[Step 6] Creating Unified State Master Dataset...")
    df_master = create_integrated_state_master(df_collection, df_waste_2023)
    print(f"   - Master dataset shape: {df_master.shape[0]} States/UTs, {df_master.shape[1]} columns")

    # Step 7: Save Cleaned Outputs in 'results/' Folder
    print("\n[Step 7] Saving cleaned files into 'results/' directory...")
    
    path_national_csv = results_dir / "cleaned_national_ewaste.csv"
    path_collection_csv = results_dir / "cleaned_state_collection.csv"
    path_top_states_csv = results_dir / "cleaned_top_states_generation.csv"
    path_waste_2023_csv = results_dir / "cleaned_state_waste_2023.csv"
    path_master_csv = results_dir / "cleaned_ewaste_master_state.csv"
    path_clean_xlsx = results_dir / "cleaned_ewaste_dataset.xlsx"

    df_national.to_csv(path_national_csv, index=False)
    df_collection.to_csv(path_collection_csv, index=False)
    df_top_states.to_csv(path_top_states_csv, index=False)
    df_waste_2023.to_csv(path_waste_2023_csv, index=False)
    df_master.to_csv(path_master_csv, index=False)

    # Also save an all-in-one clean Excel file
    with pd.ExcelWriter(path_clean_xlsx, engine="openpyxl") as writer:
        df_national.to_excel(writer, sheet_name="National_Generation", index=False)
        df_collection.to_excel(writer, sheet_name="State_Collection", index=False)
        df_top_states.to_excel(writer, sheet_name="Top_States_Comparison", index=False)
        df_waste_2023.to_excel(writer, sheet_name="State_Waste_2023", index=False)
        df_master.to_excel(writer, sheet_name="Master_Integrated_State", index=False)

    print(f"   [+] Saved: {path_national_csv.name}")
    print(f"   [+] Saved: {path_collection_csv.name}")
    print(f"   [+] Saved: {path_top_states_csv.name}")
    print(f"   [+] Saved: {path_waste_2023_csv.name}")
    print(f"   [+] Saved: {path_master_csv.name}")
    print(f"   [+] Saved: {path_clean_xlsx.name}")

    print("\n" + "=" * 75)
    print("SUCCESS: Data Preprocessing Pipeline Completed Successfully!")
    print("=" * 75)


if __name__ == "__main__":
    run_preprocessing_pipeline()
