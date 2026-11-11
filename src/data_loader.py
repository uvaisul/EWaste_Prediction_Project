"""
data_loader.py
--------------
Module for loading, validating, cleaning, and preprocessing the e-waste dataset.

Provides complete functions to handle:
1. Loading the original Excel workbook from `Dataset/` without modifying it.
2. Cleaning strings containing hidden characters, non-breaking spaces, and commas.
3. Standardizing Indian State / UT names across different reporting sources.
4. Handling missing values (e.g. median imputation with transparency indicators).
5. Checking for duplicates, negative values, and anomalies.
6. Exporting clean CSV and Excel files to the `results/` folder.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from src.utils import get_project_root, setup_logger

logger = setup_logger("DataLoader")

# Standardized mapping for Indian States and Union Territories to fix reporting inconsistencies
STATE_NAME_MAPPING: Dict[str, str] = {
    "Chattisgarh": "Chhattisgarh",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "Dadra & Nagar Haveli and Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "Andaman and Nicobar Island": "Andaman and Nicobar Islands",
}


def clean_text_numeric(val) -> Optional[float]:
    """
    Safely converts numeric values that might be stored as dirty strings.
    Handles non-breaking spaces (\xa0), commas (e.g., '200,000'), and whitespace.

    Args:
        val: Any raw cell value (int, float, str, or NaN).

    Returns:
        float or np.nan
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)

    # Clean string representation
    cleaned = str(val).replace("\xa0", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return np.nan


def standardize_state_name(name: str) -> str:
    """
    Trims whitespace and maps known naming variations to standardized state names.
    """
    if pd.isna(name):
        return ""
    cleaned = str(name).strip()
    return STATE_NAME_MAPPING.get(cleaned, cleaned)


def load_raw_workbook(filename: str = "E-WASTE VALUE.xlsx") -> Optional[pd.ExcelFile]:
    """
    Safely loads the original Excel workbook from the Dataset/ directory in read-only mode.
    """
    dataset_path = get_project_root() / "Dataset" / filename
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return None

    try:
        xl = pd.ExcelFile(dataset_path)
        logger.info(f"Loaded '{filename}' with sheets: {xl.sheet_names}")
        return xl
    except Exception as e:
        logger.error(f"Error opening Excel file '{dataset_path}': {e}")
        return None


def preprocess_national_generation(xl: pd.ExcelFile) -> pd.DataFrame:
    """
    Cleans Sheet 1: National Annual E-Waste Generation (2016 - 2023).
    """
    sheet_name = "E-WASTE INDIA"
    logger.info(f"Processing sheet: {sheet_name}")
    df = pd.read_excel(xl, sheet_name=sheet_name)

    # Standardize column headers
    df.columns = ["financial_year", "generation_tonnes"]

    # Clean data types and whitespace
    df["financial_year"] = df["financial_year"].astype(str).str.strip()
    df["generation_tonnes"] = df["generation_tonnes"].apply(clean_text_numeric)

    # Extract start year as an integer (e.g., '2016-17' -> 2016) for future ML regressors
    df["start_year"] = df["financial_year"].apply(
        lambda y: int(y.split("-")[0]) if "-" in str(y) else np.nan
    )

    # Deduplication and anomaly check
    df = df.drop_duplicates(subset=["financial_year"]).reset_index(drop=True)
    if (df["generation_tonnes"] < 0).any():
        logger.warning("Negative generation values detected in national data!")

    return df


def preprocess_state_collection(xl: pd.ExcelFile) -> pd.DataFrame:
    """
    Cleans Sheet 2: State-wise E-Waste Collection & Processed (2019 - 2022).
    """
    sheet_name = "STATE WISE E-WASTE COLLECTION "
    logger.info(f"Processing sheet: {sheet_name}")
    raw_df = pd.read_excel(xl, sheet_name=sheet_name)

    # Row 0 contains sub-headers ('2019-20', '2020-21', '2021-22')
    df = raw_df.iloc[1:].copy().reset_index(drop=True)
    df.columns = [
        "sl_no",
        "state_ut",
        "collection_2019_20_tonnes",
        "collection_2020_21_tonnes",
        "collection_2021_22_tonnes",
    ]

    # Convert data types
    df["sl_no"] = pd.to_numeric(df["sl_no"], errors="coerce").astype(int)
    df["state_ut"] = df["state_ut"].apply(standardize_state_name)

    for col in [
        "collection_2019_20_tonnes",
        "collection_2020_21_tonnes",
        "collection_2021_22_tonnes",
    ]:
        df[col] = df[col].apply(clean_text_numeric)

    # Add useful analytical aggregations
    df["total_collected_3yr_tonnes"] = (
        df["collection_2019_20_tonnes"]
        + df["collection_2020_21_tonnes"]
        + df["collection_2021_22_tonnes"]
    ).round(2)

    df["avg_annual_collection_tonnes"] = (
        df["total_collected_3yr_tonnes"] / 3.0
    ).round(2)

    # Remove duplicates
    df = df.drop_duplicates(subset=["state_ut"]).reset_index(drop=True)
    return df


def preprocess_top_states_generation(xl: pd.ExcelFile) -> Tuple[pd.DataFrame, float]:
    """
    Cleans Sheet 3: Top States E-Waste Generation (2018 vs 2023).
    Handles missing values for Punjab via transparent median imputation.
    """
    sheet_name = "STATE E-WASTE GENERATION(2018"
    logger.info(f"Processing sheet: {sheet_name}")
    raw_df = pd.read_excel(xl, sheet_name=sheet_name)

    # Header is spread over rows 0 and 1; actual data begins at index 2
    df = raw_df.iloc[2:].copy().reset_index(drop=True)
    df.columns = ["state_ut", "generation_2018_tonnes", "generation_2023_tonnes"]

    df["state_ut"] = df["state_ut"].apply(standardize_state_name)
    df["generation_2018_tonnes"] = df["generation_2018_tonnes"].apply(clean_text_numeric)
    df["generation_2023_tonnes"] = df["generation_2023_tonnes"].apply(clean_text_numeric)

    # Check missing values
    missing_2023 = df["generation_2023_tonnes"].isnull()
    median_val = float(df["generation_2023_tonnes"].median())

    # Create transparency indicator column
    df["is_2023_imputed"] = missing_2023

    # Impute missing value with group median
    df["generation_2023_tonnes"] = df["generation_2023_tonnes"].fillna(median_val)

    # Calculate growth percentage from 2018 to 2023
    df["growth_percentage"] = (
        ((df["generation_2023_tonnes"] - df["generation_2018_tonnes"])
        / df["generation_2018_tonnes"]) * 100
    ).round(2)

    df = df.drop_duplicates(subset=["state_ut"]).reset_index(drop=True)
    return df, median_val


def preprocess_state_waste_2023(xl: pd.ExcelFile) -> pd.DataFrame:
    """
    Cleans Sheet 4: State-wise Waste Generation & Processed in 2023 (TPD).
    Calculates processing efficiency percentage.
    """
    sheet_name = "STATE WISE WASTE GENERATION2023"
    logger.info(f"Processing sheet: {sheet_name}")
    df = pd.read_excel(xl, sheet_name=sheet_name)

    df.columns = [
        "sl_no",
        "state_ut",
        "waste_generation_tpd",
        "waste_processed_tpd",
    ]

    df["sl_no"] = pd.to_numeric(df["sl_no"], errors="coerce").astype(int)
    df["state_ut"] = df["state_ut"].apply(standardize_state_name)
    df["waste_generation_tpd"] = df["waste_generation_tpd"].apply(clean_text_numeric)
    df["waste_processed_tpd"] = df["waste_processed_tpd"].apply(clean_text_numeric)

    # Calculate Processing Efficiency (%)
    df["processing_efficiency_pct"] = np.where(
        df["waste_generation_tpd"] > 0,
        (df["waste_processed_tpd"] / df["waste_generation_tpd"]) * 100.0,
        0.0,
    ).round(2)

    df = df.drop_duplicates(subset=["state_ut"]).reset_index(drop=True)
    return df


def create_integrated_state_master(
    df_collection: pd.DataFrame, df_waste_2023: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges state collection data with 2023 waste processing metrics on standardized state_ut.
    Produces an integrated master dataset ready for correlation analysis and ML modeling.
    """
    logger.info("Merging state collection and municipal waste metrics into master panel...")
    merged = pd.merge(
        df_collection,
        df_waste_2023[[
            "state_ut",
            "waste_generation_tpd",
            "waste_processed_tpd",
            "processing_efficiency_pct",
        ]],
        on="state_ut",
        how="outer",
    )

    # Fill any missing values in outer merge cleanly
    numeric_cols = [
        "collection_2019_20_tonnes",
        "collection_2020_21_tonnes",
        "collection_2021_22_tonnes",
        "total_collected_3yr_tonnes",
        "avg_annual_collection_tonnes",
        "waste_generation_tpd",
        "waste_processed_tpd",
        "processing_efficiency_pct",
    ]
    for c in numeric_cols:
        merged[c] = merged[c].fillna(0.0)

    # Sort alphabetically by State / UT name
    merged = merged.sort_values(by="state_ut").reset_index(drop=True)
    merged["sl_no"] = range(1, len(merged) + 1)
    return merged
