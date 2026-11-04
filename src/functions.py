import numpy as np
import pandas as pd
import sys
from importlib.metadata import version
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


PROCESS_COLUMNS = [
    "Glucose [mM]",
    "Glutamine [mM]",
    "Lactate [mM]",
    "Ammonia [mM]",
    "VCD [cells/mL]",
    "TCD [cells/mL]",
    "mAb [mg/L]",
]

def load_data(dataset_folder):
    df_initial = pd.read_csv(dataset_folder / "initial_conditions.csv")

    batch_files = sorted(dataset_folder.glob("batch_*.csv"))
    frames = []

    for file in batch_files:
        #getting batch# from file name
        batch_id = int(file.stem.split("_")[1])

        df_batch = pd.read_csv(file)

        #starting cond. for batch
        initial = df_initial.loc[
            df_initial["Batch ID"] == batch_id
        ].iloc[0]

        df_batch["Batch ID"] = batch_id
        df_batch["Initial glucose [mM]"] = initial["Glucose [mM]"]
        df_batch["Initial glutamine [mM]"] = initial["Glutamine [mM]"]
        df_batch["Initial VCD [cells/mL]"] = initial["VCD [cells/mL]"]

        #economic equations uses days
        df_batch["Time [d]"] = df_batch["Time [h]"] / 24

        frames.append(df_batch)

    return pd.concat(frames, ignore_index=True)

#------ one day downtime, best time
def calculate_economics(df):
    df = df.copy()

    df["Cost [$/L]"] = (
        10 * df["Initial glucose [mM]"]
        + 20 * df["Initial glutamine [mM]"]
        + 0.0005 * df["Initial VCD [cells/mL]"]
    )

    df["Value [$/L]"] = 1000 * df["mAb [mg/L]"]
    df["Profit [$/L]"] = df["Value [$/L]"] - df["Cost [$/L]"]

    df["Space-time profit [$/L/d]"] = (
        df["Profit [$/L]"] / (df["Time [d]"] + 1)
    )

    df["Space-time yield [mg/L/d]"] = (
        df["mAb [mg/L]"] / (df["Time [d]"] + 1)
    )

    return df

def summarize_batches(df):
    records = []

    for batch_id in df["Batch ID"].unique():
        batch = df.loc[df["Batch ID"] == batch_id]

        #find time with highest space time profit
        best_index = batch["Space-time profit [$/L/d]"].idxmax()
        records.append(batch.loc[best_index])

    summary = pd.DataFrame(records)

    return summary.sort_values(
        "Space-time profit [$/L/d]", ascending=False
    )

#------
def prepare_training_data(df):
    x_lag = []
    x_cur = []
    groups = []

    for batch_id in df["Batch ID"].unique():
        batch = df.loc[df["Batch ID"] == batch_id]
        batch = batch.sort_values("Time [h]")

        values = batch[PROCESS_COLUMNS].to_numpy()

        # Pair each measurement with the following measurement
        x_lag.extend(values[:-1])
        x_cur.extend(values[1:])

        # Identify which batch each training example belongs to
        groups.extend([batch_id] * (len(values) - 1))

    return np.array(x_lag), np.array(x_cur), np.array(groups)


def score_mab(model, x, y):
    prediction = model.predict(x)

    # Evaluate mAb prediction error; mAb is the last column
    return -root_mean_squared_error(y[:, -1], prediction[:, -1])


def train_models(df):
    x_lag, x_cur, groups = prepare_training_data(df)

    models = {}
    records = []

    for model_type in ["Polynomial", "kNN"]:

        if model_type == "Polynomial":
            pipeline = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(include_bias=False),
                LinearRegression()
            )

            param_grid = {
                "polynomialfeatures__degree": [1, 2]
            }

        else:
            pipeline = make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(weights="uniform")
            )

            param_grid = {
                "kneighborsregressor__n_neighbors": [2, 5, 10]
            }

        #keeping whole batches tgt during cross validation
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=GroupKFold(n_splits=5),
            scoring=score_mab
        )

        search.fit(x_lag, x_cur, groups=groups)

        models[model_type] = search.best_estimator_

        records.append({
            "Model": model_type,
            "CV one-step RMSE [mg/L]": -search.best_score_,
            "Best parameters": str(search.best_params_)
        })

    return models, pd.DataFrame(records)

#---
def make_initial_states(conditions):
    states = np.zeros((len(conditions), len(PROCESS_COLUMNS)))

    states[:, 0] = conditions["Initial glucose [mM]"].to_numpy()
    states[:, 1] = conditions["Initial glutamine [mM]"].to_numpy()
    states[:, 4] = conditions["Initial VCD [cells/mL]"].to_numpy()

    #initial total cell density = initial viable cell density
    states[:, 5] = states[:, 4]

    #initial lactate, ammonia, and mAb stay 0
    return states

def forecast_states(model, initial_states, n_steps, upper_limits):
    current = np.array(initial_states, dtype=float)

    forecasts = np.zeros(
        (len(current), n_steps, len(PROCESS_COLUMNS))
    )
    forecasts[:, 0, :] = current

    for step in range(1, n_steps):
        current = model.predict(current)

        #keeping predictions within physical n training bounds
        current = np.clip(current, 0, upper_limits)

        forecasts[:, step, :] = current

    return forecasts


def validate_forecasts(df):
    x_lag, x_cur, groups = prepare_training_data(df)

    splits = GroupKFold(n_splits=5)
    forecast_tables = []

    for fold, (train_index, test_index) in enumerate(
        splits.split(x_lag, x_cur, groups), start=1
    ):
        print(f"Validating forecast fold {fold}/5...")

        train_ids = np.unique(groups[train_index])
        test_ids = np.unique(groups[test_index])

        train_df = df.loc[df["Batch ID"].isin(train_ids)]
        test_df = df.loc[df["Batch ID"].isin(test_ids)]

        #tune both models using only this fold's training batches
        fold_models, _ = train_models(train_df)

        upper_limits = train_df[PROCESS_COLUMNS].max().to_numpy()

        conditions = test_df.drop_duplicates("Batch ID")
        conditions = conditions.sort_values("Batch ID")

        initial_states = make_initial_states(conditions)
        n_steps = len(
            test_df.loc[test_df["Batch ID"] == test_ids[0]]
        )

        for model_name, model in fold_models.items():
            forecasts = forecast_states(
                model, initial_states, n_steps, upper_limits
            )

            for i, batch_id in enumerate(test_ids):
                batch = test_df.loc[test_df["Batch ID"] == batch_id]
                batch = batch.sort_values("Time [h]")

                times = batch["Time [d]"].to_numpy()
                cost = batch["Cost [$/L]"].iloc[0]
                predicted_mab = forecasts[i, :, -1]

                table = pd.DataFrame({
                    "Model": model_name,
                    "Batch ID": batch_id,
                    "Time [d]": times,
                    "Actual mAb [mg/L]":
                        batch["mAb [mg/L]"].to_numpy(),
                    "Predicted mAb [mg/L]":
                        predicted_mab,
                    "Actual space-time profit [$/L/d]":
                        batch["Space-time profit [$/L/d]"].to_numpy(),
                    "Predicted space-time profit [$/L/d]":
                        (1000 * predicted_mab - cost) / (times + 1)
                })

                forecast_tables.append(table)

    df_forecasts = pd.concat(forecast_tables, ignore_index=True)
    records = []

    for model_name in df_forecasts["Model"].unique():
        result = df_forecasts.loc[
            df_forecasts["Model"] == model_name
        ]

        records.append({
            "Model": model_name,
            "Forecast mAb RMSE [mg/L]": root_mean_squared_error(
                result["Actual mAb [mg/L]"],
                result["Predicted mAb [mg/L]"]
            ),
            "Forecast profit RMSE [$/L/d]": root_mean_squared_error(
                result["Actual space-time profit [$/L/d]"],
                result["Predicted space-time profit [$/L/d]"]
            )
        })

    comparison = pd.DataFrame(records)
    comparison = comparison.sort_values(
        "Forecast profit RMSE [$/L/d]"
    )

    return comparison, df_forecasts

#-----
def optimize_production(model, df, initial_conditions_file):
    df_initial = pd.read_csv(initial_conditions_file)

    conditions = df_initial[
        ["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]"]
    ].drop_duplicates()

    conditions = conditions.rename(columns={
        "Glucose [mM]": "Initial glucose [mM]",
        "Glutamine [mM]": "Initial glutamine [mM]",
        "VCD [cells/mL]": "Initial VCD [cells/mL]"
    }).reset_index(drop=True)

    times = np.sort(df["Time [d]"].unique())
    upper_limits = df[PROCESS_COLUMNS].max().to_numpy()
    initial_states = make_initial_states(conditions)

    forecasts = forecast_states(
        model, initial_states, len(times), upper_limits
    )

    predicted_mab = forecasts[:, :, -1]

    costs = (
        10 * conditions["Initial glucose [mM]"]
        + 20 * conditions["Initial glutamine [mM]"]
        + 0.0005 * conditions["Initial VCD [cells/mL]"]
    ).to_numpy()

    # Calculate profit per litre per day for every candidate and time
    space_time_profit = (
        (1000 * predicted_mab - costs[:, None])
        / (times[None, :] + 1)
    )

    records = []

    for i in range(len(conditions)):
        best_time_index = np.argmax(space_time_profit[i])
        harvest_time = times[best_time_index]
        mab = predicted_mab[i, best_time_index]

        record = conditions.iloc[i].to_dict()

        record.update({
            "Harvest time [d]": harvest_time,
            "Predicted mAb [mg/L]": mab,
            "Cost [$/L]": costs[i],
            "Value [$/L]": 1000 * mab,
            "Space-time profit [$/L/d]":
                space_time_profit[i, best_time_index],
            "Space-time yield [mg/L/d]":
                mab / (harvest_time + 1)
        })

        records.append(record)

    results = pd.DataFrame(records)

    return results.sort_values(
        "Space-time profit [$/L/d]", ascending=False
    )

#-----
def estimate_uncertainty(model, df, recommendation, n_simulations=1000):
    rng = np.random.default_rng(0)

    x_lag, x_cur, _ = prepare_training_data(df)

    # Estimate prediction errors and centre them around zero
    residuals = x_cur - model.predict(x_lag)
    residuals = residuals - residuals.mean(axis=0)

    times = np.sort(df["Time [d]"].unique())
    upper_limits = df[PROCESS_COLUMNS].max().to_numpy()

    conditions = pd.DataFrame([recommendation])
    initial_state = make_initial_states(conditions)

    current = np.repeat(initial_state, n_simulations, axis=0)
    simulated_mab = np.zeros((n_simulations, len(times)))

    for step in range(1, len(times)):
        baseline = model.predict(current)
        baseline = np.clip(baseline, 0, upper_limits)

        # Sample the seven errors together to preserve correlations
        noise = residuals[
            rng.integers(len(residuals), size=n_simulations)
        ]

        current = np.maximum(baseline + noise, 0)
        simulated_mab[:, step] = current[:, -1]

    lower, median, upper = np.quantile(
        simulated_mab, [0.025, 0.5, 0.975], axis=0
    )

    harvest_time = recommendation["Harvest time [d]"]
    cost = recommendation["Cost [$/L]"]

    harvest_index = np.argmin(np.abs(times - harvest_time))
    mab_samples = simulated_mab[:, harvest_index]

    value_samples = 1000 * mab_samples
    profit_samples = (value_samples - cost) / (harvest_time + 1)
    yield_samples = mab_samples / (harvest_time + 1)

    metrics = [
        (
            "mAb [mg/L]",
            recommendation["Predicted mAb [mg/L]"],
            mab_samples
        ),
        (
            "Value [$/L]",
            recommendation["Value [$/L]"],
            value_samples
        ),
        (
            "Space-time profit [$/L/d]",
            recommendation["Space-time profit [$/L/d]"],
            profit_samples
        ),
        (
            "Space-time yield [mg/L/d]",
            recommendation["Space-time yield [mg/L/d]"],
            yield_samples
        )
    ]

    records = []

    for metric, estimate, samples in metrics:
        records.append({
            "Metric": metric,
            "Point estimate": estimate,
            "95% PI lower": np.quantile(samples, 0.025),
            "95% PI upper": np.quantile(samples, 0.975)
        })

    forecast_table = pd.DataFrame({
        "Time [d]": times,
        "mAb median [mg/L]": median,
        "mAb lower [mg/L]": lower,
        "mAb upper [mg/L]": upper,
        "Profit median [$/L/d]": (1000 * median - cost) / (times + 1),
        "Profit lower [$/L/d]": (1000 * lower - cost) / (times + 1),
        "Profit upper [$/L/d]": (1000 * upper - cost) / (times + 1)
    })

    return pd.DataFrame(records), forecast_table

