"""
recycling_analysis.py
---------------------
Module for in-depth E-Waste Recycling Analysis, Mass Flow Modeling, and Policy Assessment.

Calculates:
1. Total e-waste generated (multi-year & 3-year aligned).
2. Total e-waste collected.
3. Total e-waste recycled (material recovery output).
4. Recycling rate percentage (annual and aggregate).
5. Uncollected waste (informal stream & household hoarding).
6. Waste requiring further processing (uncollected + hazardous dismantling residue).

Generates high-resolution comparison charts in `graphs/` and exports summary CSVs to `results/`.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_project_root, setup_logger

logger = setup_logger("RecyclingAnalysis")

# Visual style setup
sns.set_theme(style="whitegrid", palette="muted")

# Benchmark formal recycling recovery efficiency (CPCB / academic consensus)
MATERIAL_RECOVERY_RATE = 0.85
HAZARDOUS_RESIDUE_RATE = 0.15


def perform_recycling_calculations() -> Dict[str, Any]:
    """
    Computes all 6 core recycling metrics using cleaned real datasets.
    """
    root = get_project_root()
    nat_path = root / "results" / "cleaned_national_ewaste.csv"
    col_path = root / "results" / "cleaned_state_collection.csv"

    if not nat_path.exists() or not col_path.exists():
        raise FileNotFoundError("Cleaned datasets not found. Run preprocess.py first.")

    df_nat = pd.read_csv(nat_path)
    df_col = pd.read_csv(col_path)

    # Overlapping 3-year reporting period (2019-20 to 2021-22)
    aligned_years = ["2019-20", "2020-21", "2021-22"]
    df_nat_3yr = df_nat[df_nat["financial_year"].isin(aligned_years)].copy().reset_index(drop=True)

    # 1. Total Generated
    total_generated_3yr = float(df_nat_3yr["generation_tonnes"].sum())
    total_generated_all_time = float(df_nat["generation_tonnes"].sum())

    # 2. Total Collected
    col_19_20 = float(df_col["collection_2019_20_tonnes"].sum())
    col_20_21 = float(df_col["collection_2020_21_tonnes"].sum())
    col_21_22 = float(df_col["collection_2021_22_tonnes"].sum())
    total_collected_3yr = float(df_col["total_collected_3yr_tonnes"].sum())

    # 3. Total Recycled (Material Recovery Output)
    total_recycled_3yr = total_collected_3yr * MATERIAL_RECOVERY_RATE
    hazardous_residue_3yr = total_collected_3yr * HAZARDOUS_RESIDUE_RATE

    # 4. Recycling Rate Percentage
    collection_rate_pct = (total_collected_3yr / total_generated_3yr) * 100.0
    material_recycled_rate_pct = (total_recycled_3yr / total_generated_3yr) * 100.0

    # 5. Uncollected Waste
    uncollected_waste_3yr = total_generated_3yr - total_collected_3yr
    uncollected_pct = (uncollected_waste_3yr / total_generated_3yr) * 100.0

    # 6. Waste Requiring Further Processing
    # (Uncollected waste + non-recyclable hazardous residue from formal plants)
    waste_requiring_processing_3yr = uncollected_waste_3yr + hazardous_residue_3yr
    requiring_processing_pct = (waste_requiring_processing_3yr / total_generated_3yr) * 100.0

    # Build annual breakdown table
    annual_metrics = pd.DataFrame({
        "financial_year": aligned_years,
        "generated_tonnes": df_nat_3yr["generation_tonnes"].values,
        "collected_tonnes": [col_19_20, col_20_21, col_21_22],
    })
    annual_metrics["recycled_material_tonnes"] = np.round(annual_metrics["collected_tonnes"] * MATERIAL_RECOVERY_RATE, 2)
    annual_metrics["hazardous_residue_tonnes"] = np.round(annual_metrics["collected_tonnes"] * HAZARDOUS_RESIDUE_RATE, 2)
    annual_metrics["uncollected_tonnes"] = np.round(annual_metrics["generated_tonnes"] - annual_metrics["collected_tonnes"], 2)
    annual_metrics["collection_rate_pct"] = np.round((annual_metrics["collected_tonnes"] / annual_metrics["generated_tonnes"]) * 100, 2)
    annual_metrics["uncollected_rate_pct"] = np.round((annual_metrics["uncollected_tonnes"] / annual_metrics["generated_tonnes"]) * 100, 2)
    annual_metrics["requiring_processing_tonnes"] = np.round(annual_metrics["uncollected_tonnes"] + annual_metrics["hazardous_residue_tonnes"], 2)

    # Export Annual Summary to results/
    results_dir = root / "results"
    summary_path = results_dir / "recycling_analysis_metrics.csv"
    annual_metrics.to_csv(summary_path, index=False)
    logger.info(f"Saved recycling analysis summary to: {summary_path}")

    return {
        "total_generated_3yr": total_generated_3yr,
        "total_generated_all_time": total_generated_all_time,
        "total_collected_3yr": total_collected_3yr,
        "total_recycled_3yr": total_recycled_3yr,
        "hazardous_residue_3yr": hazardous_residue_3yr,
        "collection_rate_pct": collection_rate_pct,
        "material_recycled_rate_pct": material_recycled_rate_pct,
        "uncollected_waste_3yr": uncollected_waste_3yr,
        "uncollected_pct": uncollected_pct,
        "waste_requiring_processing_3yr": waste_requiring_processing_3yr,
        "requiring_processing_pct": requiring_processing_pct,
        "annual_metrics": annual_metrics,
        "df_col": df_col,
    }


def plot_recycling_breakdown_donut(metrics: Dict[str, Any], filename: str = "8_recycling_mass_flow_breakdown.png") -> Path:
    """
    Comparison Chart 1: Donut chart displaying the ultimate fate of all generated e-waste.
    Shows the distribution between Recycled Material, Hazardous Residue, and Uncollected Waste.
    """
    root = get_project_root()
    save_path = root / "graphs" / filename

    labels = [
        f"Material Recovered & Recycled\n({metrics['total_recycled_3yr']/1e6:.2f}M Tonnes | {metrics['material_recycled_rate_pct']:.1f}%)",
        f"Dismantling Residue to TSDF\n({metrics['hazardous_residue_3yr']/1e6:.2f}M Tonnes | {(metrics['hazardous_residue_3yr']/metrics['total_generated_3yr'])*100:.1f}%)",
        f"Uncollected Waste\n(Informal Sector & Dumps)\n({metrics['uncollected_waste_3yr']/1e6:.2f}M Tonnes | {metrics['uncollected_pct']:.1f}%)",
    ]
    sizes = [
        metrics["total_recycled_3yr"],
        metrics["hazardous_residue_3yr"],
        metrics["uncollected_waste_3yr"],
    ]
    colors = ["#2ca02c", "#d95f02", "#e7298a"]
    explode = (0.04, 0.04, 0.06)

    fig, ax = plt.subplots(figsize=(9, 8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        textprops=dict(color="#222222", fontsize=10.5, weight="medium"),
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5),
    )

    for at in autotexts:
        at.set_fontsize(11)
        at.set_weight("bold")
        at.set_color("white")

    # Center label in the donut hole
    ax.text(
        0, 0,
        f"Total E-Waste\nGenerated\n\n{metrics['total_generated_3yr']/1e6:.2f}M Tonnes\n(100%)",
        ha="center", va="center", fontsize=12, weight="bold", color="#1a365d"
    )

    plt.title("National E-Waste Fate & Recycling Efficiency Breakdown\n(Cumulative 3-Year Analysis: 2019-20 to 2021-22)", weight="bold", pad=20, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved: {save_path.name}")
    return save_path


def plot_annual_flow_comparison(metrics: Dict[str, Any], filename: str = "9_annual_collected_vs_uncollected.png") -> Path:
    """
    Comparison Chart 2: Stacked Bar Chart comparing Collected vs. Uncollected e-waste each year.
    """
    root = get_project_root()
    save_path = root / "graphs" / filename

    df_ann = metrics["annual_metrics"]
    years = df_ann["financial_year"]
    collected = df_ann["collected_tonnes"] / 1000.0
    uncollected = df_ann["uncollected_tonnes"] / 1000.0

    plt.figure(figsize=(10, 6.5))

    p1 = plt.bar(years, collected, width=0.55, label="Formally Collected & Processed", color="#2b8cbe", edgecolor="#1a5276")
    p2 = plt.bar(years, uncollected, bottom=collected, width=0.55, label="Uncollected Waste (Informal / Landfill Gap)", color="#cb181d", edgecolor="#7f0000")

    plt.title("Annual E-Waste Generation: Formal Collection vs. Uncollected Deficit", weight="bold", pad=15)
    plt.xlabel("Financial Year", weight="bold", labelpad=10)
    plt.ylabel("E-Waste Quantity (in Thousand Metric Tonnes)", weight="bold", labelpad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bar segments
    for i, year in enumerate(years):
        c_val = collected.iloc[i]
        u_val = uncollected.iloc[i]
        total_val = c_val + u_val
        c_pct = df_ann["collection_rate_pct"].iloc[i]
        u_pct = df_ann["uncollected_rate_pct"].iloc[i]

        # Collected text
        plt.text(i, c_val / 2, f"{c_val:,.1f}k T\n({c_pct:.1f}%)", ha="center", va="center", color="white", weight="bold", fontsize=9.5)
        # Uncollected text
        plt.text(i, c_val + u_val / 2, f"{u_val:,.1f}k T\n({u_pct:.1f}%)", ha="center", va="center", color="white", weight="bold", fontsize=9.5)
        # Total text on top
        plt.text(i, total_val + 25, f"Total: {total_val:,.1f}k T", ha="center", va="bottom", weight="bold", fontsize=10, color="#111111")

    plt.ylim(0, (collected + uncollected).max() * 1.18)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved: {save_path.name}")
    return save_path


def plot_state_recycling_disparity(df_col: pd.DataFrame, top_n: int = 12, filename: str = "10_state_recycling_disparity.png") -> Path:
    """
    Comparison Chart 3: Highlights the extreme disparity between leading states and lagging states
    in cumulative e-waste collection & recycling.
    """
    root = get_project_root()
    save_path = root / "graphs" / filename

    sorted_df = df_col.sort_values(by="total_collected_3yr_tonnes", ascending=False)
    top_states = sorted_df.head(top_n)
    bottom_states = sorted_df.tail(6)

    combined_df = pd.concat([top_states, bottom_states]).reset_index(drop=True)
    combined_df = combined_df.sort_values(by="total_collected_3yr_tonnes", ascending=True)

    plt.figure(figsize=(11, 8))

    colors = ["#cb181d" if x in bottom_states["state_ut"].values else "#2b8cbe" for x in combined_df["state_ut"]]

    bars = plt.barh(
        combined_df["state_ut"],
        combined_df["total_collected_3yr_tonnes"] / 1000.0,
        color=colors,
        edgecolor="#111111",
        height=0.65,
    )

    plt.title(f"State-Level E-Waste Collection Disparity\n(Top {top_n} Leaders vs. Bottom Lagging States: 2019-20 to 2021-22)", weight="bold", pad=15)
    plt.xlabel("Total Processed E-Waste (in Thousand Metric Tonnes)", weight="bold", labelpad=10)
    plt.ylabel("State / Union Territory", weight="bold", labelpad=10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    # Annotations
    for bar in bars:
        width = bar.get_width()
        plt.annotate(
            f" {width:,.2f}k T",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            weight="semibold",
        )

    plt.xlim(0, (combined_df["total_collected_3yr_tonnes"].max() / 1000.0) * 1.15)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b8cbe", edgecolor="#111111", label="Established Industrial Recycling Hubs"),
        Patch(facecolor="#cb181d", edgecolor="#111111", label="Under-resourced / Lagging States & UTs"),
    ]
    plt.legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved: {save_path.name}")
    return save_path
