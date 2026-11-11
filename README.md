# E-Waste Generation Prediction and Recycling Analysis

An End-to-End Machine Learning, Data Analytics, and Interactive Streamlit Dashboard project for electronic waste (e-waste) generation prediction and recycling analysis using Python, Scikit-Learn, and Streamlit.

---

## 📌 1. Project Overview

With the rapid acceleration of technology, consumer electronics turnover has increased dramatically. Electronic waste (e-waste) contains hazardous materials (lead, mercury, cadmium) as well as valuable recoverable metals (gold, silver, copper).

### Key Accomplishments:
1. **Authentic Data Cleaning:** Loaded multi-sheet official data (`E-WASTE VALUE.xlsx`), handled messy text characters, imputed missing values transparently, and generated clean CSV/Excel datasets.
2. **5 Exploratory Visualizations:** Annual generation bars, top 15 state rankings, 3-year recycling comparisons, dual-axis recycling efficiency percentage, and YoY growth trajectories.
3. **Machine Learning Predictive Modeling:** Trained Random Forest Regressors to predict waste and e-waste generation, evaluated with MAE, RMSE, and $R^2 = 0.7254$ without data leakage.
4. **Mass Flow Recycling Audit:** Quantified the $66\%$ uncollected gap, material recovery output ($1.15\text{M tonnes}$), and waste requiring processing ($2.82\text{M tonnes}$).
5. **Interactive Web Dashboard:** Modern 7-page Streamlit dashboard featuring live ML inference, state comparison, visualization gallery, and data download center.

---

## 📁 2. Project Directory Structure

```text
EWaste_Prediction_Project/
│
├── Dataset/                     # Original raw dataset
│   ├── E-WASTE VALUE.xlsx       # Official multi-sheet CPCB / MoEFCC e-waste workbook
│   └── README.md                # Data documentation
│
├── src/                         # Reusable Python package modules
│   ├── __init__.py              # Package init
│   ├── utils.py                 # Path helpers and standardized logger
│   ├── data_loader.py           # Preprocessing & cleaning functions
│   ├── eda.py                   # Matplotlib & Seaborn visualization functions
│   ├── model.py                 # Random Forest training & evaluation logic
│   └── recycling_analysis.py    # Mass flow calculations & comparison charts
│
├── graphs/                      # High-resolution (300 DPI) generated plots (10 charts)
│   ├── 1_ewaste_generation_by_year.png
│   ├── 2_statewise_ewaste_comparison.png
│   ├── 3_ewaste_recycling_comparison.png
│   ├── 4_recycling_percentage_analysis.png
│   ├── 5_ewaste_generation_trend.png
│   ├── 6_actual_vs_predicted_state_generation.png
│   ├── 7_actual_vs_predicted_timeseries.png
│   ├── 8_recycling_mass_flow_breakdown.png
│   ├── 9_annual_collected_vs_uncollected.png
│   ├── 10_state_recycling_disparity.png
│   └── README.md
│
├── results/                     # Cleaned data, serialized models, metrics
│   ├── cleaned_national_ewaste.csv
│   ├── cleaned_state_collection.csv
│   ├── cleaned_top_states_generation.csv
│   ├── cleaned_state_waste_2023.csv
│   ├── cleaned_ewaste_master_state.csv
│   ├── cleaned_ewaste_dataset.xlsx
│   ├── random_forest_state_model.joblib
│   ├── random_forest_timeseries_model.joblib
│   ├── predictions_state_generation.csv
│   ├── predictions_timeseries.csv
│   ├── model_evaluation_metrics.csv
│   ├── recycling_analysis_metrics.csv
│   └── README.md
│
├── reports/                     # Project synopsis, academic reports, and slides
│   └── README.md
│
├── app.py                       # Interactive Streamlit Web Application
├── main.py                      # Master pipeline runner (runs all 5 steps)
├── preprocess.py                # Standalone data cleaning runner
├── analyze.py                   # Standalone EDA & visualization runner
├── train_model.py               # Standalone ML training & evaluation runner
├── run_recycling_analysis.py    # Standalone recycling mass flow runner
├── requirements.txt             # Required Python libraries
└── README.md                    # Project documentation
```

---

## 🚀 3. How to Launch and Run the Project

### Option A: Launch the Interactive Streamlit Web Dashboard
Run the command below in your terminal:
```bash
streamlit run app.py
```
This opens the web dashboard in your browser (at `http://localhost:8501`) featuring:
1. 📖 Project Introduction
2. 📊 E-Waste Generation Statistics
3. 🗺️ State-Wise Comparison (Interactive selector)
4. 🔄 Recycling Rate & Mass Flow Analysis
5. 📈 Data Visualizations (10 interactive high-res charts)
6. 🔮 Live Machine Learning Model Prediction Tool
7. 🌿 Data-Driven Pollution-Abatement Recommendations & Data Download Center

### Option B: Run the Full Machine Learning Pipeline from Terminal
```bash
python main.py
```

### Option C: Run Individual Modules Independently
```bash
python preprocess.py             # Cleans raw data and saves to results/
python analyze.py                # Generates EDA charts into graphs/
python train_model.py            # Trains Random Forest and evaluates accuracy
python run_recycling_analysis.py # Computes 6 recycling metrics and flow charts
```
