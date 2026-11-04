from pathlib import Path

import src.functions as sf


PROJECT_DIR = Path(__file__).resolve().parent

df = sf.load_data(PROJECT_DIR / "datasets")

print("Number of batches:", df["Batch ID"].nunique())
print("Number of data rows:", len(df))
print(df.head())

#-----
df = sf.calculate_economics(df)
df_summary = sf.summarize_batches(df)

tables_folder = PROJECT_DIR / "tables"
tables_folder.mkdir(exist_ok=True)

df_summary.to_csv(tables_folder / "batch_summary.csv", index=False)

print("\nBest recorded result:")
print(df_summary.iloc[0])

#----
print("\nTraining models...")

models, df_comparison = sf.train_models(df)

df_comparison.to_csv(
    tables_folder / "model_comparison.csv",
    index=False
)

print(df_comparison.to_string(index=False))

#----
print("\nChecking full batch forecasts...")

df_forecast_comparison, df_forecasts = sf.validate_forecasts(df)

df_forecast_comparison.to_csv(
    tables_folder / "forecast_comparison.csv", index=False
)
df_forecasts.to_csv(
    tables_folder / "validation_forecasts.csv", index=False
)

best_model_name = df_forecast_comparison.iloc[0]["Model"]
best_model = models[best_model_name]

print(df_forecast_comparison.to_string(index=False))
print("\nSelected model:", best_model_name)

#------
print("\nOptimizing production...")

df_optimization = sf.optimize_production(
    best_model,
    df,
    PROJECT_DIR / "datasets" / "initial_conditions.csv"
)

df_optimization.to_csv(
    tables_folder / "optimization_results.csv", index=False
)

recommendation = df_optimization.iloc[0]

print("Starting-condition combinations:", len(df_optimization))
print("\nModel-recommended settings:")
print(recommendation.to_string())

#-----
print("\nEstimating prediction intervals...")

df_intervals, df_recommended_forecast = sf.estimate_uncertainty(
    best_model, df, recommendation
)

df_intervals.to_csv(
    tables_folder / "prediction_intervals.csv", index=False
)
df_recommended_forecast.to_csv(
    tables_folder / "recommended_forecast.csv", index=False
)

print(df_intervals.to_string(index=False))

#----
import sys
from importlib.metadata import version

print("\nEnvironment versions:")
print("Python:", sys.version.split()[0])

for package in ["numpy", "pandas", "scikit-learn"]:
    print(f"{package}: {version(package)}")