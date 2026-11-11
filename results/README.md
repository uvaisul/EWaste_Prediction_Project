# Results Directory

This directory stores all cleaned datasets, trained machine learning models, prediction logs, and evaluation metrics.

---

## 🤖 Machine Learning Model Artifacts & Evaluations

| File Name | Description | Details |
| :--- | :--- | :--- |
| [`random_forest_state_model.joblib`](file:///c:/Users/uvais/OneDrive/EWaste_Prediction_Project/results/random_forest_state_model.joblib) | Serialized Random Forest Regressor for state waste/e-waste generation. | $100$ trees, max depth $6$, $R^2 = 0.7254$ on unseen test states. |
| [`random_forest_timeseries_model.joblib`](file:///c:/Users/uvais/OneDrive/EWaste_Prediction_Project/results/random_forest_timeseries_model.joblib) | Serialized time-series forecasting model. | Strict chronological train-test split ($2016-20$ train, $2021-22$ test). |
| [`model_evaluation_metrics.csv`](file:///c:/Users/uvais/OneDrive/EWaste_Prediction_Project/results/model_evaluation_metrics.csv) | Summary table of MAE, RMSE, and $R^2$ scores for both models. | Machine-readable metrics. |
| [`predictions_state_generation.csv`](file:///c:/Users/uvais/OneDrive/EWaste_Prediction_Project/results/predictions_state_generation.csv) | Actual vs. Predicted values, absolute errors, and percentage errors for test states. | Unseen test states comparison. |
| [`predictions_timeseries.csv`](file:///c:/Users/uvais/OneDrive/EWaste_Prediction_Project/results/predictions_timeseries.csv) | Time-series test period predictions vs actual generation. | Chronological evaluation log. |

---

## 📊 Preprocessed Cleaned Datasets

| File Name | Description | Rows | Columns |
| :--- | :--- | :---: | :---: |
| [`cleaned_national_ewaste.csv`](file:///c:/Users/uvais/OneDrive/EWaste_Prediction_Project/results/cleaned_national_ewaste.csv) | National annual e-waste generation time-series ($2016-17$ to $2022-23$). | 7 | 3 |
| [`cleaned_state_collection.csv`](file:///c:/Users/uvais/OneDrive/EWaste_Prediction_Project/results/cleaned_state_collection.csv) | State-wise collection & processed e-waste ($2019-20$, $2020-21$, $2021-22$) with $3$-year total and annual average. | 35 | 7 |
| [`cleaned_top_states_generation.csv`](file:///c:/Users/uvais/OneDrive/EWaste_Prediction_Project/results/cleaned_top_states_generation.csv) | Top $10$ generating states ($2018$ vs $2023$), with transparent median imputation and growth percentage. | 10 | 5 |
| [`cleaned_state_waste_2023.csv`](file:///c:/Users/uvais/OneDrive/EWaste_Prediction_Project/results/cleaned_state_waste_2023.csv) | State-wise daily waste generation, processed quantity (TPD), and calculated processing efficiency (\%). | 35 | 5 |
| [`cleaned_ewaste_master_state.csv`](file:///c:/Users/uvais/OneDrive/EWaste_Prediction_Project/results/cleaned_ewaste_master_state.csv) | **Integrated Master Dataset:** Merges state e-waste collection with 2023 processing efficiency across all 36 Indian States/UTs. | 36 | 10 |
| [`cleaned_ewaste_dataset.xlsx`](file:///c:/Users/uvais/OneDrive/EWaste_Prediction_Project/results/cleaned_ewaste_dataset.xlsx) | Multi-sheet Excel workbook containing all five cleaned datasets in dedicated sheets. | — | — |
