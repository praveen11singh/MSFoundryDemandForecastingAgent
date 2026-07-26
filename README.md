# Demand Forecasting Agent

This repository contains a small Python agent for generating a demand forecast from historical time-series data and optionally producing a business-ready narrative summary with Azure AI Foundry.

## What it does

- Forecasts future demand using either:
  - Prophet, when available
  - SARIMAX as a fallback
- Produces a forecast table with date, forecast, lower bound, and upper bound
- Optionally generates a narrative summary using a deployed Azure AI Foundry model

## Repository files

- [DemandForecastingAgent.py](DemandForecastingAgent.py) - forecasting and Foundry integration logic
- [historical_demand.csv](historical_demand.csv) - sample historical demand data
- [demand_forecast_output.csv](demand_forecast_output.csv) - example forecast output
- [requirements.txt](requirements.txt) - Python dependencies
- [.env](.env) - environment variables for Azure configuration

## Requirements

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variables in your shell or in [.env](.env):

```bash
PROJECT_ENDPOINT=https://<your-foundry-resource>.services.ai.azure.com/api/projects/<your-project>
MODEL_DEPLOYMENT=gpt-4o-mini
```

If you are using Azure authentication, sign in first:

```bash
az login
```

## Input format

The forecasting logic expects a DataFrame with at least these columns:

- `date`
- `demand`

The sample file [historical_demand.csv](historical_demand.csv) includes these columns and can be used as-is.

## Example usage

```python
import pandas as pd
from DemandForecastingAgent import ForecastConfig, DemandForecastingAgent

historical_df = pd.read_csv("historical_demand.csv")

agent = DemandForecastingAgent(ForecastConfig(forecast_horizon=30))
result = agent.run(historical_df, with_insights=True)

result["forecast"].to_csv("demand_forecast_output.csv", index=False)
print(result["summary"])
```

## Notes

- If Prophet is not installed, the agent automatically falls back to SARIMAX.
- The Azure AI Foundry integration requires a valid project endpoint and a deployed model.
- The generated summary is optional; set `with_insights=False` to skip it.
