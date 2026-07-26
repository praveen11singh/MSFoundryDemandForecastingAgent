import os
import json
import socket
import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

import pandas as pd

# --- Azure AI Foundry Project SDK ------------------------------------------
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
# Load variables from a .env file (if present) into the environment
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demand_forecasting")


# ----------------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------------
@dataclass
class ForecastConfig:
    date_col: str = "date"
    demand_col: str = "demand"
    forecast_horizon: int = 30          # days to forecast ahead
    freq: str = "D"                     # daily frequency
    project_endpoint: str = os.environ.get("PROJECT_ENDPOINT", "")
    model_deployment: str = os.environ.get("MODEL_DEPLOYMENT", "gpt-4o-mini")


# ----------------------------------------------------------------------------
# 2. Time-series forecasting engine
# ----------------------------------------------------------------------------
class DemandForecaster:
    """Runs the actual numerical forecast. Tries Prophet first, falls
    back to statsmodels SARIMAX if Prophet isn't installed."""

    def __init__(self, config: ForecastConfig):
        self.config = config

    def forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[[self.config.date_col, self.config.demand_col]].copy()
        df[self.config.date_col] = pd.to_datetime(df[self.config.date_col])
        df = df.sort_values(self.config.date_col)

        try:
            return self._forecast_prophet(df)
        except ImportError:
            logger.warning("Prophet not installed, falling back to SARIMAX")
            return self._forecast_sarimax(df)

    def _forecast_prophet(self, df: pd.DataFrame) -> pd.DataFrame:
        from prophet import Prophet

        prophet_df = df.rename(
            columns={self.config.date_col: "ds", self.config.demand_col: "y"}
        )
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(
            periods=self.config.forecast_horizon, freq=self.config.freq
        )
        forecast = model.predict(future)

        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(
            self.config.forecast_horizon
        )
        result = result.rename(
            columns={
                "ds": "date",
                "yhat": "forecast",
                "yhat_lower": "lower_bound",
                "yhat_upper": "upper_bound",
            }
        )
        return result.reset_index(drop=True)

    def _forecast_sarimax(self, df: pd.DataFrame) -> pd.DataFrame:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        series = df.set_index(self.config.date_col)[self.config.demand_col]
        series = series.asfreq(self.config.freq).interpolate()

        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),  # weekly seasonality
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)

        pred = fitted.get_forecast(steps=self.config.forecast_horizon)
        mean = pred.predicted_mean
        conf_int = pred.conf_int(alpha=0.20)  # ~80% interval, similar to Prophet default

        result = pd.DataFrame(
            {
                "date": mean.index,
                "forecast": mean.values,
                "lower_bound": conf_int.iloc[:, 0].values,
                "upper_bound": conf_int.iloc[:, 1].values,
            }
        ).reset_index(drop=True)
        return result


# ----------------------------------------------------------------------------
# 3. Azure AI Foundry integration — narrative insights on top of the forecast
# ----------------------------------------------------------------------------
class FoundryInsightGenerator:
   
    def __init__(self, config: ForecastConfig):
        if not config.project_endpoint:
            raise ValueError(
                "PROJECT_ENDPOINT is not set. Set it to your Microsoft Foundry "
                "project endpoint, e.g. https://<resource>.services.ai.azure.com/api/projects/<project>"
            )
        self.config = config
        self._validate_endpoint_reachable(config.project_endpoint)

        self.credential = DefaultAzureCredential()
        self.project_client = AIProjectClient(
            endpoint=config.project_endpoint.strip(),
            credential=self.credential,
        )
        # Authenticated OpenAI-compatible client scoped to this Foundry project
        self.openai_client = self.project_client.get_openai_client()

    @staticmethod
    def _validate_endpoint_reachable(endpoint: str):
        """Fail fast with a clear message instead of a 40-line httpx/openai
        traceback when PROJECT_ENDPOINT is malformed or DNS can't resolve it."""
        endpoint = endpoint.strip()
        parsed = urlparse(endpoint)

        if parsed.scheme not in ("https", "http") or not parsed.hostname:
            raise ValueError(
                f"PROJECT_ENDPOINT doesn't look like a valid URL: {endpoint!r}. "
                "Expected format: https://<ai-services-resource-name>.services.ai.azure.com"
                "/api/projects/<project-name> — copy it exactly from your Foundry "
                "project's Overview page (no quotes, no trailing spaces)."
            )
        if "/api/projects/" not in parsed.path:
            logger.warning(
                "PROJECT_ENDPOINT %r doesn't contain '/api/projects/<project-name>'. "
                "Double-check you copied the *project* endpoint, not the resource endpoint.",
                endpoint,
            )

        try:
            socket.gethostbyname(parsed.hostname)
        except socket.gaierror as e:
            raise ConnectionError(
                f"Could not resolve host '{parsed.hostname}' from PROJECT_ENDPOINT "
                f"= {endpoint!r}. This means the request never left your machine. "
                "Most likely causes:\n"
                "  1. Typo in the resource/project name — re-copy the endpoint from "
                "the Foundry project Overview page.\n"
                "  2. You're on a VPN / corporate network / proxy that blocks or "
                "doesn't route *.services.ai.azure.com — try on an unrestricted "
                "network, or configure HTTPS_PROXY if one is required.\n"
                "  3. The env var isn't actually set in this shell/session — run "
                "`echo $env:PROJECT_ENDPOINT` (PowerShell) to confirm its exact value.\n"
                f"Quick manual check: run `nslookup {parsed.hostname}` in a terminal — "
                "if that fails too, it confirms this is a DNS/network issue, not this script."
            ) from e

    def generate_summary(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> str:
        hist_tail = historical_df.tail(14).to_dict(orient="records")
        forecast_head = forecast_df.head(14).to_dict(orient="records")

        instructions = (
            "You are a supply chain analyst. Given recent historical demand "
            "and a statistical forecast, write a concise business summary: "
            "trend direction, notable risk of stockout/overstock, and one "
            "recommended action. Keep it under 150 words."
        )
        prompt = (
            f"Recent historical demand (last 14 points): {json.dumps(hist_tail, default=str)}\n\n"
            f"Forecast (first 14 points): {json.dumps(forecast_head, default=str)}"
        )

        response = self.openai_client.responses.create(
            model=self.config.model_deployment,
            instructions=instructions,
            input=prompt,
        )
        return response.output_text

    def close(self):
        try:
            self.openai_client.close()
        except AttributeError:
            pass
        self.project_client.close()


# ----------------------------------------------------------------------------
# 4. Orchestration
# ----------------------------------------------------------------------------
class DemandForecastingAgent:
    def __init__(self, config: Optional[ForecastConfig] = None):
        self.config = config or ForecastConfig()
        self.forecaster = DemandForecaster(self.config)
        self._insight_gen: Optional[FoundryInsightGenerator] = None

    @property
    def insight_gen(self) -> FoundryInsightGenerator:
        if self._insight_gen is None:
            self._insight_gen = FoundryInsightGenerator(self.config)
        return self._insight_gen

    def run(self, historical_df: pd.DataFrame, with_insights: bool = True):
        # logger.info("Running demand forecast for %s periods", self.config.forecast_horizon)
        forecast_df = self.forecaster.forecast(historical_df)

        summary = None
        if with_insights:
            # logger.info("Requesting narrative insights from Azure AI Foundry model")
            summary = self.insight_gen.generate_summary(historical_df, forecast_df)

        return {"forecast": forecast_df, "summary": summary}

    def close(self):
        if self._insight_gen is not None:
            self._insight_gen.close()


# ----------------------------------------------------------------------------
# 5. Realistic dummy data — simulates 2 years of daily retail sales for a SKU
# ----------------------------------------------------------------------------
def generate_dummy_retail_data(
    sku: str = "SKU-1042-WirelessHeadphones",
    start_date: str = "2023-01-01",
    periods: int = 730,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates a realistic-looking daily demand series with:
      - a slow upward trend (growing product)
      - weekly seasonality (weekend dip for a B2B-ish product)
      - yearly seasonality (Nov/Dec holiday spike, summer lull)
      - promo spikes on random days (e.g. flash sales)
      - occasional stockout days (demand recorded as 0 / near-0)
      - random noise on top of all of the above

    This mimics what you'd actually pull from a POS / ERP export,
    including the messiness (promos, stockouts) real data has.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=periods, freq="D")

    t = np.arange(periods)
    trend = 80 + 0.06 * t  # slow organic growth

    # Weekly seasonality: lower demand on weekends
    dow = dates.dayofweek.values  # Mon=0 ... Sun=6
    weekly = np.where(dow >= 5, -15, 5)

    # Yearly seasonality: holiday season bump (Nov 15 - Dec 31), summer lull (Jun-Jul)
    day_of_year = dates.dayofyear.values
    holiday_bump = 45 * np.exp(-0.5 * ((day_of_year - 340) / 15) ** 2)  # peaks ~ Dec 6
    summer_lull = -15 * np.exp(-0.5 * ((day_of_year - 195) / 20) ** 2)  # dip ~ mid-July
    yearly = holiday_bump + summer_lull

    # Random promo spikes: ~1.5% of days get a flash-sale style spike
    promo_days = rng.choice([0, 1], size=periods, p=[0.985, 0.015])
    promo_effect = promo_days * rng.uniform(40, 90, size=periods)

    # Random stockouts: ~1% of days, demand craters (unmet demand recorded low)
    stockout_days = rng.choice([0, 1], size=periods, p=[0.99, 0.01])
    stockout_effect = stockout_days * -rng.uniform(60, 90, size=periods)

    # Gaussian noise
    noise = rng.normal(0, 6, size=periods)

    demand = trend + weekly + yearly + promo_effect + stockout_effect + noise
    demand = np.clip(demand, 0, None).round().astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "sku": sku,
            "demand": demand,
            "was_promo": promo_days.astype(bool),
            "was_stockout": stockout_days.astype(bool),
        }
    )
    return df


# ----------------------------------------------------------------------------
# 6. Example usage — simulates a real pipeline: CSV export -> load -> forecast
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = "historical_demand.csv"

    # In a real scenario this file would come from your POS/ERP/warehouse
    # system export. Here we generate one so the script is fully runnable
    # end-to-end, then load it back exactly like you would with real data.
    if not os.path.exists(DATA_PATH):
        # logger.info("No existing data file found, generating dummy retail dataset")
        dummy_df = generate_dummy_retail_data(periods=730)
        dummy_df.to_csv(DATA_PATH, index=False)
        # logger.info("Wrote %d rows of dummy demand data to %s", len(dummy_df), DATA_PATH)

    historical_df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print("\n--- Historical data sample ---")
    print(historical_df.head(5).to_string(index=False))
    print("...")
    print(historical_df.tail(5).to_string(index=False))
    print(f"\nTotal rows: {len(historical_df)}  |  "
          f"Promo days: {historical_df['was_promo'].sum()}  |  "
          f"Stockout days: {historical_df['was_stockout'].sum()}")

    config = ForecastConfig(
        date_col="date",
        demand_col="demand",
        forecast_horizon=30,
    )

    agent = DemandForecastingAgent(config)
    try:
        result = agent.run(historical_df, with_insights=bool(config.project_endpoint))

        print("\n--- Forecast (next 10 days) ---")
        print(result["forecast"].head(10).to_string(index=False))

        forecast_path = "demand_forecast_output.csv"
        result["forecast"].to_csv(forecast_path, index=False)
        print(f"\nFull {config.forecast_horizon}-day forecast written to {forecast_path}")

        if result["summary"]:
            print("\n--- Foundry-generated Insight ---")
            print(result["summary"])
        else:
            print(
                "\n(Set PROJECT_ENDPOINT and MODEL_DEPLOYMENT env vars to also "
                "get an AI-generated narrative summary via Azure AI Foundry.)"
            )
    finally:
        agent.close()