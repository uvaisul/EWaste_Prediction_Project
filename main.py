"""
main.py
-------
Master execution script for the E-Waste Generation Prediction and Recycling Analysis project.

Orchestrates the entire end-to-end pipeline:
1. Environment & Directory verification.
2. Data Preprocessing (Cleaning raw data, handling missing values, saving to results/).
3. Exploratory Data Analysis (Generating 5 publication-ready charts in graphs/).
4. Machine Learning Model Training (Training Random Forest Regressors, computing metrics, and saving models).
5. Recycling Mass Flow Analysis (Computing 6 core recycling metrics, generating fate & flow comparison charts).
"""

from src.utils import ensure_project_directories, setup_logger
from preprocess import run_preprocessing_pipeline
from analyze import run_analysis
from train_model import run_training_pipeline
from run_recycling_analysis import run_analysis_pipeline

logger = setup_logger("Main")


def main():
    print("=" * 80)
    print("      E-WASTE GENERATION PREDICTION & RECYCLING ANALYSIS PROJECT")
    print("=" * 80)

    # Step 1: Verify directories
    logger.info("Step 1/5: Verifying project directories...")
    ensure_project_directories()

    # Step 2: Preprocess raw data
    logger.info("Step 2/5: Executing Data Preprocessing Pipeline...")
    run_preprocessing_pipeline()

    # Step 3: Run EDA & Visualizations
    logger.info("Step 3/5: Executing Exploratory Data Analysis & Visualizations...")
    run_analysis()

    # Step 4: Train & Evaluate Machine Learning Models
    logger.info("Step 4/5: Executing Machine Learning Training & Evaluation...")
    run_training_pipeline()

    # Step 5: Execute In-Depth Recycling Analysis
    logger.info("Step 5/5: Executing E-Waste Recycling & Mass Flow Analysis...")
    run_analysis_pipeline()

    print("\n" + "=" * 80)
    print("ALL 5 PROJECT PHASES EXECUTED SUCCESSFULLY!")
    print("• Cleaned Datasets & Models: 'results/'")
    print("• Visualizations & Charts  : 'graphs/'")
    print("=" * 80)


if __name__ == "__main__":
    main()
