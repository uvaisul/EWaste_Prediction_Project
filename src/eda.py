"""
eda.py
------
Module for Exploratory Data Analysis (EDA) and visualizations.

Generates authentic, publication-quality visualizations using the cleaned datasets:
1. E-waste generation by year (Bar chart)
2. State-wise e-waste comparison (Horizontal bar chart of top states)
3. E-waste recycling comparison (Grouped bar chart across multiple financial years)
4. Recycling percentage (Annual national recycling rate & state processing efficiency)
5. E-waste generation trend (Time-series line plot with trajectory annotations)

All figures are saved with high resolution (300 DPI) in the `graphs/` directory.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_project_root, setup_logger

logger = setup_logger("EDA")

# Apply modern aesthetic theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
})


def get_graphs_dir() -> Path:
    """Returns the path to the graphs/ directory, creating it if needed."""
    graphs_dir = get_project_root() / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    return graphs_dir


def plot_generation_by_year(df_national: pd.DataFrame, filename: str = "1_ewaste_generation_by_year.png") -> Path:
    """
    1. E-waste generation by year.
    Creates a bar chart showing annual national e-waste generation in India.
    """
    save_path = get_graphs_dir() / filename
    logger.info(f"Generating: {filename}...")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        df_national["financial_year"],
        df_national["generation_tonnes"] / 1000.0,  # Convert to thousand tonnes
        color="#2b5c8f",
        edgecolor="#1a365d",
        width=0.6,
    )

    plt.title("National E-Waste Generation in India (2016-17 to 2022-23)", weight="bold", pad=15)
    plt.xlabel("Financial Year", weight="bold", labelpad=10)
    plt.ylabel("E-Waste Generated (in Thousand Metric Tonnes)", weight="bold", labelpad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        raw_val = height * 1000.0
        plt.annotate(
            f"{height:,.1f}k T\n({raw_val:,.0f} T)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="semibold",
        )

    plt.ylim(0, (df_national["generation_tonnes"].max() / 1000.0) * 1.18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved: {save_path.name}")
    return save_path


def plot_statewise_comparison(df_collection: pd.DataFrame, top_n: int = 15, filename: str = "2_statewise_ewaste_comparison.png") -> Path:
    """
    2. State-wise e-waste comparison.
    Creates a horizontal bar chart ranking the top states by 3-year total e-waste collected & processed.
    """
    save_path = get_graphs_dir() / filename
    logger.info(f"Generating: {filename}...")

    top_states = (
        df_collection.sort_values(by="total_collected_3yr_tonnes", ascending=False)
        .head(top_n)
        .sort_values(by="total_collected_3yr_tonnes", ascending=True)
    )

    plt.figure(figsize=(11, 7.5))
    bars = plt.barh(
        top_states["state_ut"],
        top_states["total_collected_3yr_tonnes"] / 1000.0,
        color="#1f77b4",
        edgecolor="#0d47a1",
        height=0.65,
    )

    plt.title(f"Top {top_n} Indian States by Total E-Waste Collected & Processed\n(Cumulative 2019-20 to 2021-22)", weight="bold", pad=15)
    plt.xlabel("Total E-Waste Processed (in Thousand Metric Tonnes)", weight="bold", labelpad=10)
    plt.ylabel("State / Union Territory", weight="bold", labelpad=10)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Annotate values
    for bar in bars:
        width = bar.get_width()
        plt.annotate(
            f" {width:,.1f}k Tonnes",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            weight="semibold",
        )

    plt.xlim(0, (top_states["total_collected_3yr_tonnes"].max() / 1000.0) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved: {save_path.name}")
    return save_path


def plot_recycling_comparison(df_collection: pd.DataFrame, top_n: int = 10, filename: str = "3_ewaste_recycling_comparison.png") -> Path:
    """
    3. E-waste recycling comparison.
    Grouped bar chart comparing annual recycling/processed volumes across years for top states.
    """
    save_path = get_graphs_dir() / filename
    logger.info(f"Generating: {filename}...")

    # Filter top N states by 3-year total
    top_df = df_collection.sort_values(by="total_collected_3yr_tonnes", ascending=False).head(top_n)

    x = np.arange(len(top_df))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 7))

    r1 = ax.bar(x - width, top_df["collection_2019_20_tonnes"] / 1000.0, width, label="2019-20", color="#4575b4")
    r2 = ax.bar(x, top_df["collection_2020_21_tonnes"] / 1000.0, width, label="2020-21", color="#74add1")
    r3 = ax.bar(x + width, top_df["collection_2021_22_tonnes"] / 1000.0, width, label="2021-22", color="#313695")

    ax.set_title(f"State-wise Annual E-Waste Recycling & Processing Comparison\n(Top {top_n} States across Financial Years)", weight="bold", pad=15)
    ax.set_xlabel("State / Union Territory", weight="bold", labelpad=10)
    ax.set_ylabel("Quantity Processed (in Thousand Metric Tonnes)", weight="bold", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(top_df["state_ut"], rotation=30, ha="right", weight="medium")
    ax.legend(title="Reporting Period", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved: {save_path.name}")
    return save_path


def plot_recycling_percentage(
    df_national: pd.DataFrame,
    df_collection: pd.DataFrame,
    filename: str = "4_recycling_percentage_analysis.png",
) -> Path:
    """
    4. Recycling percentage.
    Compares national e-waste generation vs. formal collection and calculates the annual recycling rate (%).
    """
    save_path = get_graphs_dir() / filename
    logger.info(f"Generating: {filename}...")

    # Compute national collection totals per overlapping year
    col_totals = {
        "2019-20": df_collection["collection_2019_20_tonnes"].sum(),
        "2020-21": df_collection["collection_2020_21_tonnes"].sum(),
        "2021-22": df_collection["collection_2021_22_tonnes"].sum(),
    }

    # Extract matching generation numbers
    df_gen_sub = df_national[df_national["financial_year"].isin(col_totals.keys())].copy()
    df_gen_sub["collected_tonnes"] = df_gen_sub["financial_year"].map(col_totals)
    df_gen_sub["recycling_rate_pct"] = (
        df_gen_sub["collected_tonnes"] / df_gen_sub["generation_tonnes"]
    ) * 100.0

    fig, ax1 = plt.subplots(figsize=(11, 6.5))

    x = np.arange(len(df_gen_sub))
    width = 0.35

    # Bar chart for Generated vs. Processed
    bars1 = ax1.bar(x - width/2, df_gen_sub["generation_tonnes"] / 1000.0, width, label="Total Generated", color="#e08214")
    bars2 = ax1.bar(x + width/2, df_gen_sub["collected_tonnes"] / 1000.0, width, label="Formally Recycled / Processed", color="#41ab5d")

    ax1.set_xlabel("Financial Year", weight="bold", labelpad=10)
    ax1.set_ylabel("Quantity (in Thousand Metric Tonnes)", weight="bold", labelpad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_gen_sub["financial_year"])
    ax1.set_ylim(0, (df_gen_sub["generation_tonnes"].max() / 1000.0) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Line plot on secondary y-axis for Recycling Rate Percentage
    ax2 = ax1.twinx()
    line = ax2.plot(x, df_gen_sub["recycling_rate_pct"], color="#800026", marker="o", linewidth=2.5, markersize=8, label="Formal Recycling Rate (%)")
    ax2.set_ylabel("Recycling Rate (%)", weight="bold", color="#800026", labelpad=10)
    ax2.tick_params(axis="y", labelcolor="#800026")
    ax2.set_ylim(0, 60)
    ax2.grid(False)

    # Annotate percentages
    for i, rate in enumerate(df_gen_sub["recycling_rate_pct"]):
        ax2.annotate(
            f"{rate:.1f}%",
            xy=(i, rate),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            weight="bold",
            color="#800026",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff2ae", edgecolor="#800026", alpha=0.9),
        )

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.title("National E-Waste Generation vs. Formal Recycling & Collection Rate\n(Recycling Efficiency %)", weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved: {save_path.name}")
    return save_path


def plot_generation_trend(df_national: pd.DataFrame, filename: str = "5_ewaste_generation_trend.png") -> Path:
    """
    5. E-waste generation trend.
    Creates a time-series trend line chart with filled area and growth rate markers.
    """
    save_path = get_graphs_dir() / filename
    logger.info(f"Generating: {filename}...")

    years = df_national["financial_year"]
    gen_in_thousands = df_national["generation_tonnes"] / 1000.0

    plt.figure(figsize=(11, 6))

    # Trend line with markers
    plt.plot(
        years,
        gen_in_thousands,
        color="#c0392b",
        marker="o",
        linewidth=2.8,
        markersize=9,
        label="E-Waste Generation (Thousand Tonnes)",
    )
    plt.fill_between(years, gen_in_thousands, color="#e74c3c", alpha=0.15)

    plt.title("E-Waste Generation Growth Trajectory in India (2016-17 to 2022-23)", weight="bold", pad=15)
    plt.xlabel("Financial Year", weight="bold", labelpad=10)
    plt.ylabel("E-Waste Generation (in Thousand Metric Tonnes)", weight="bold", labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Annotate data points and YoY change
    for i in range(len(years)):
        y_val = gen_in_thousands.iloc[i]
        raw_val = df_national["generation_tonnes"].iloc[i]
        
        annotation_text = f"{y_val:,.1f}k T"
        if i > 0:
            prev_val = df_national["generation_tonnes"].iloc[i - 1]
            yoy_growth = ((raw_val - prev_val) / prev_val) * 100
            annotation_text += f"\n(+{yoy_growth:.1f}%)" if yoy_growth >= 0 else f"\n({yoy_growth:.1f}%)"

        plt.annotate(
            annotation_text,
            xy=(i, y_val),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="semibold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#c0392b", alpha=0.8),
        )

    plt.ylim(0, gen_in_thousands.max() * 1.25)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved: {save_path.name}")
    return save_path
