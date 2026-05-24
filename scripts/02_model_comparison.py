from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUTS_DIR = BASE_DIR / "outputs"

FIGURES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

N_LAGS = 12
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42
HORIZONS = [1, 5]


def qlike_loss(actual_log_rv, forecast_log_rv):
    actual_var = np.exp(2 * np.asarray(actual_log_rv))
    forecast_var = np.exp(2 * np.asarray(forecast_log_rv))
    forecast_var = np.maximum(forecast_var, 1e-12)
    ratio = actual_var / forecast_var
    return ratio - np.log(ratio) - 1


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "QLIKE": np.mean(qlike_loss(y_true, y_pred)),
    }


def newey_west_variance(series, max_lag):
    values = np.asarray(series) - np.mean(series)
    n = len(values)
    gamma_0 = np.sum(values * values) / n
    variance = gamma_0

    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        gamma = np.sum(values[lag:] * values[:-lag]) / n
        variance += 2 * weight * gamma

    return variance


def diebold_mariano_test(actual, benchmark_forecast, model_forecast, horizon):
    benchmark_loss = (np.asarray(actual) - np.asarray(benchmark_forecast)) ** 2
    model_loss = (np.asarray(actual) - np.asarray(model_forecast)) ** 2

    loss_difference = benchmark_loss - model_loss
    n = len(loss_difference)

    nw_lag = max(horizon - 1, 0)
    variance = newey_west_variance(loss_difference, nw_lag)

    if variance <= 0:
        return np.nan, np.nan

    dm_stat = np.mean(loss_difference) / np.sqrt(variance / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def build_supervised_data(rv, horizon):
    data = pd.DataFrame({"rv": rv})

    for lag in range(1, N_LAGS + 1):
        data[f"lag_{lag}"] = data["rv"].shift(lag)

    data["har_daily"] = data["lag_1"]
    data["har_weekly"] = data[[f"lag_{lag}" for lag in range(1, 6)]].mean(axis=1)
    data["har_monthly"] = data[[f"lag_{lag}" for lag in range(1, 13)]].mean(axis=1)

    data["target"] = data["rv"].shift(-horizon)
    data["forecast_date"] = pd.Series(data.index, index=data.index).shift(-horizon)

    return data.dropna()


def fit_forecasts(data):
    ar_features = ["lag_1"]
    har_features = ["har_daily", "har_weekly", "har_monthly"]
    mlp_features = [f"lag_{lag}" for lag in range(1, N_LAGS + 1)]

    split_index = int(len(data) * TRAIN_FRACTION)

    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    y_train = train["target"]
    y_test = test["target"]

    predictions = pd.DataFrame(index=pd.to_datetime(test["forecast_date"]))
    predictions.index.name = "Date"
    predictions["Actual"] = y_test.values

    predictions["Random Walk"] = test["rv"].values

    ar_model = LinearRegression()
    ar_model.fit(train[ar_features], y_train)
    predictions["AR(1)"] = ar_model.predict(test[ar_features])

    har_model = LinearRegression()
    har_model.fit(train[har_features], y_train)
    predictions["HAR"] = har_model.predict(test[har_features])

    mlp_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    max_iter=5000,
                    early_stopping=True,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    mlp_model.fit(train[mlp_features], y_train)
    predictions["MLP"] = mlp_model.predict(test[mlp_features])

    return train, test, predictions


def plot_forecasts(predictions, horizon):
    plt.figure(figsize=(10, 4))
    plt.plot(predictions.index, predictions["Actual"], label="Actual", linewidth=1.6)
    plt.plot(predictions.index, predictions["Random Walk"], label="Random Walk", alpha=0.75)
    plt.plot(predictions.index, predictions["HAR"], label="HAR", alpha=0.85)
    plt.plot(predictions.index, predictions["MLP"], label="MLP", alpha=0.85)
    plt.title(f"{horizon}-Month-Ahead Forecasts of S&P 500 Log Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Log Realized Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"actual_vs_predicted_rv_{horizon}_step.png", dpi=300)
    plt.close()


def plot_crisis_period(predictions):
    crisis = predictions.loc["2007-01-01":"2009-12-31"]

    plt.figure(figsize=(10, 4))
    plt.plot(crisis.index, crisis["Actual"], label="Actual", linewidth=1.8)
    plt.plot(crisis.index, crisis["Random Walk"], label="Random Walk", alpha=0.75)
    plt.plot(crisis.index, crisis["HAR"], label="HAR", alpha=0.85)
    plt.plot(crisis.index, crisis["MLP"], label="MLP", alpha=0.85)
    plt.title("One-Month-Ahead Forecasts During the 2007-2009 Crisis Period")
    plt.xlabel("Date")
    plt.ylabel("Log Realized Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "crisis_period_forecasts.png", dpi=300)
    plt.close()


def plot_rmse_comparison(results):
    pivot = results.pivot(index="Model", columns="Horizon", values="RMSE")
    pivot = pivot.loc[["Random Walk", "AR(1)", "HAR", "MLP"]]

    ax = pivot.plot(kind="bar", figsize=(8, 4))
    ax.set_title("Forecast RMSE by Model and Horizon")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.legend(title="Forecast horizon")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_rmse_comparison.png", dpi=300)
    plt.close()


def main():
    np.random.seed(RANDOM_SEED)

    rv_data = pd.read_csv(
        DATA_DIR / "sp500_monthly_realized_volatility.csv",
        index_col=0,
        parse_dates=True,
    )
    rv = rv_data.iloc[:, 0]

    all_results = []
    all_dm_tests = []
    prediction_files = {}
    training_counts = {}
    testing_counts = {}

    for horizon in HORIZONS:
        data = build_supervised_data(rv, horizon)
        train, test, predictions = fit_forecasts(data)

        prediction_path = OUTPUTS_DIR / f"model_predictions_{horizon}_step.csv"
        predictions.to_csv(prediction_path)
        prediction_files[horizon] = prediction_path.name

        training_counts[horizon] = len(train)
        testing_counts[horizon] = len(test)

        for model_name in ["Random Walk", "AR(1)", "HAR", "MLP"]:
            metrics = evaluate_model(predictions["Actual"], predictions[model_name])
            all_results.append(
                {
                    "Horizon": f"{horizon}-month ahead",
                    "Model": model_name,
                    **metrics,
                }
            )

        for model_name in ["AR(1)", "HAR", "MLP"]:
            dm_stat, p_value = diebold_mariano_test(
                actual=predictions["Actual"],
                benchmark_forecast=predictions["Random Walk"],
                model_forecast=predictions[model_name],
                horizon=horizon,
            )
            all_dm_tests.append(
                {
                    "Horizon": f"{horizon}-month ahead",
                    "Benchmark": "Random Walk",
                    "Model": model_name,
                    "DM_statistic": dm_stat,
                    "p_value": p_value,
                    "Interpretation": "positive DM means model improves on Random Walk",
                }
            )

        plot_forecasts(predictions, horizon)

        if horizon == 1:
            plot_crisis_period(predictions)

    results = pd.DataFrame(all_results).sort_values(["Horizon", "RMSE"])
    dm_tests = pd.DataFrame(all_dm_tests)

    results.to_csv(OUTPUTS_DIR / "model_comparison_results.csv", index=False)
    dm_tests.to_csv(OUTPUTS_DIR / "diebold_mariano_tests.csv", index=False)

    plot_rmse_comparison(results)

    best_1_step = results[results["Horizon"] == "1-month ahead"].iloc[0]
    best_5_step = results[results["Horizon"] == "5-month ahead"].iloc[0]

    summary_text = f"""Replication Summary

Paper: Bucci (2020), Realized Volatility Forecasting with Neural Networks
Dataset: S&P 500 monthly log realized volatility
Raw sample: {rv.index.min().date()} to {rv.index.max().date()}
Monthly observations: {len(rv)}

Implementation:
The script compares a Random Walk benchmark, AR(1), HAR, and a feed-forward MLP neural network.
Forecasts are evaluated for 1-month-ahead and 5-month-ahead horizons.
Evaluation metrics are MSE, RMSE, MAE, and QLIKE.
Diebold-Mariano tests compare AR(1), HAR, and MLP against the Random Walk benchmark using squared-error loss.

Training/testing split:
1-month ahead: {training_counts[1]} training observations, {testing_counts[1]} testing observations
5-month ahead: {training_counts[5]} training observations, {testing_counts[5]} testing observations

Best model by RMSE:
1-month ahead: {best_1_step["Model"]}, RMSE = {best_1_step["RMSE"]:.4f}
5-month ahead: {best_5_step["Model"]}, RMSE = {best_5_step["RMSE"]:.4f}

Generated tables:
- model_comparison_results.csv
- diebold_mariano_tests.csv
- model_predictions_1_step.csv
- model_predictions_5_step.csv

Generated figures:
- monthly_realized_volatility.png
- actual_vs_predicted_rv_1_step.png
- actual_vs_predicted_rv_5_step.png
- crisis_period_forecasts.png
- model_rmse_comparison.png

Interpretation:
This is a simplified Python replication of Bucci (2020). It reproduces the realized-volatility construction and compares traditional benchmarks with a feed-forward neural network. It does not fully reproduce the paper's LSTM, NARX, macro-financial predictors, or Model Confidence Set procedure, so any difference from the paper's conclusions should be interpreted as arising from the simplified model set and predictor set.
"""

    (OUTPUTS_DIR / "replication_summary.txt").write_text(summary_text, encoding="utf-8")

    print(results.to_string(index=False))
    print()
    print(dm_tests.to_string(index=False))
    print()
    print("Model comparison complete.")


if __name__ == "__main__":
    main()