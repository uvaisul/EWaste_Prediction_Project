"""
run_recycling_analysis.py
-------------------------
Executable runner for Comprehensive E-Waste Recycling Analysis.

Calculates:
1. Total e-waste generated
2. Total e-waste collected
3. Total e-waste recycled
4. Recycling rate percentage
5. Uncollected waste
6. Waste requiring further processing

Generates 3 comparison charts in `graphs/` and exports summary CSV in `results/`.
"""

from pathlib import Path
from src.utils import get_project_root, setup_logger
from src.recycling_analysis import (
    perform_recycling_calculations,
    plot_recycling_breakdown_donut,
    plot_annual_flow_comparison,
    plot_state_recycling_disparity,
)

logger = setup_logger("RecyclingRunner")


def run_analysis_pipeline():
    print("=" * 80)
    print("      E-WASTE PROJECT: RECYCLING ANALYSIS & MASS FLOW REPORT")
    print("=" * 80)

    # 1. Perform Calculations
    print("\n[Step 1] Computing 6 Core Recycling Metrics from Cleaned Real Datasets...")
    res = perform_recycling_calculations()

    print("\n" + "-" * 75)
    print("             NATIONAL E-WASTE RECYCLING AUDIT (2019-20 to 2021-22)")
    print("-" * 75)
    print(f" 1. Total E-Waste Generated          : {res['total_generated_3yr']:,.2f} Tonnes (100.0%)")
    print(f"    (All-Time Total: 2016-2023)       : {res['total_generated_all_time']:,.2f} Tonnes")
    print(f" 2. Total E-Waste Collected          : {res['total_collected_3yr']:,.2f} Tonnes ({res['collection_rate_pct']:.2f}%)")
    print(f" 3. Total E-Waste Recycled (85% yield): {res['total_recycled_3yr']:,.2f} Tonnes ({res['material_recycled_rate_pct']:.2f}%)")
    print(f"    - Dismantling Slag / Residue      : {res['hazardous_residue_3yr']:,.2f} Tonnes (15% to TSDF)")
    print(f" 4. Recycling Rate Percentage        : {res['collection_rate_pct']:.2f}% (Formal Channeling)")
    print(f" 5. Uncollected Waste (Deficit)      : {res['uncollected_waste_3yr']:,.2f} Tonnes ({res['uncollected_pct']:.2f}%)")
    print(f" 6. Waste Requiring Further Proc.    : {res['waste_requiring_processing_3yr']:,.2f} Tonnes ({res['requiring_processing_pct']:.2f}%)")
    print("-" * 75)

    print("\nAnnual Trajectory Table:")
    print(res["annual_metrics"].to_string(index=False))

    # 2. Generate Comparison Charts
    print("\n[Step 2] Generating Comparison Visualizations...")
    g8 = plot_recycling_breakdown_donut(res)
    print(f"   [+] Generated: graphs/{g8.name}")

    g9 = plot_annual_flow_comparison(res)
    print(f"   [+] Generated: graphs/{g9.name}")

    g10 = plot_state_recycling_disparity(res["df_col"])
    print(f"   [+] Generated: graphs/{g10.name}")

    print("\n" + "=" * 80)
    print("SUCCESS: Recycling Analysis Pipeline Completed Successfully!")
    print("Exported Summary: results/recycling_analysis_metrics.csv")
    print("=" * 80)


if __name__ == "__main__":
    run_analysis_pipeline()
