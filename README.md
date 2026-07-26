# Demand Forecasting Agent (Microsoft Foundry SDK)

Forecasts product demand from historical sales data, then uses the
Microsoft Foundry Project SDK to turn the raw numbers into a written
business summary.

Foundry itself doesn't do time-series math — it's an agent/model-catalog
platform. So this project splits the work in two:

1. **`DemandForecaster`** — the actual forecasting engine. Uses
   [Prophet](https://facebook.github.io/prophet/), falling back to
   `statsmodels` SARIMAX if Prophet isn't installed.
2. **`FoundryInsightGenerator`** — calls a model deployed in your Foundry
   project (via `azure-ai-projects`) to turn the forecast into a plain-English
   summary: trend direction, stockout/overstock risk, and a recommended action.

Both are wrapped by **`DemandForecastingAgent`**, which you call end-to-end
with `.run()`.

## Architecture

```
historical_demand.csv (POS/ERP export)
        │
        ▼
┌─────────────────────────────────────────────┐
│  DemandForecastingAgent.run()                │
│                                               │
│  ┌───────────────────┐   ┌─────────────────┐ │
│  │ DemandForecaster   │──▶│ forecast df    │ │
│  │ Prophet / SARIMAX  │   └─────────────────┘ │
│  └───────────────────┘            │           │
│                                   ▼           │
│                    ┌───────────────────────┐  │
│                    │ FoundryInsightGenerator│  │
│                    │ Microsoft Foundry SDK  │  │
│                    └───────────────────────┘  │
└─────────────────────────────────────────────┘
        │
        ▼
Business summary (narrative + demand_forecast_output.csv)
```

## Requirements

- Python 3.9+
- An Azure subscription with a [Microsoft Foundry project](https://learn.microsoft.com/en-us/azure/foundry/how-to/create-projects)
- A model deployed in that project (e.g. `gpt-4o-mini`)
- The [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli), logged in (`az login`) — or another
  credential supported by `DefaultAzureCredential` (Managed Identity, Service Principal, etc.)

## Install

```bash
pip install "azure-ai-projects>=2.0.0" azure-identity pandas prophet statsmodels
```

> **SDK version matters.** This script targets `azure-ai-projects` **v2.x**
> (branded the "Microsoft Foundry SDK"), which uses
> `project_client.get_openai_client()` + the OpenAI Responses API. Older
> v1.x releases used a different, now-removed `.inference.get_chat_completions_client()`
> API. Run `pip show azure-ai-projects` to confirm your version; if you're
> stuck on v1.x, the `FoundryInsightGenerator` class will need adapting.

## Configuration

Set these environment variables before running:

| Variable            | Description                                                                                   | Example |
|---------------------|-----------------------------------------------------------------------------------------------|---------|
| `PROJECT_ENDPOINT`   | Your Foundry **project** endpoint — copy exactly from the project's Overview page in the Foundry portal | `https://my-resource.services.ai.azure.com/api/projects/my-project` |
| `MODEL_DEPLOYMENT`   | The exact deployment name of a model in your project (Models + endpoints tab)                 | `gpt-4o-mini` |

**PowerShell:**
```powershell
$env:PROJECT_ENDPOINT = "https://my-resource.services.ai.azure.com/api/projects/my-project"
$env:MODEL_DEPLOYMENT = "gpt-4o-mini"
```

**bash/zsh:**
```bash
export PROJECT_ENDPOINT="https://my-resource.services.ai.azure.com/api/projects/my-project"
export MODEL_DEPLOYMENT="gpt-4o-mini"
```

If `PROJECT_ENDPOINT` is unset, the script still runs the numeric forecast
and simply skips the AI narrative step.

## Run

```bash
python DemandForecastingAgent.py
```

On first run, if `historical_demand.csv` doesn't exist yet, the script
generates 2 years of realistic dummy daily sales data for a sample SKU
(`SKU-1042-WirelessHeadphones`) — with trend, weekly/yearly seasonality,
random promo spikes, and occasional stockout days — and writes it to disk.
It then reads that file back in exactly like a real pipeline would, runs
the 30-day forecast, and (if configured) requests a narrative summary.

Output:
- Console: data sample, forecast preview, AI summary (if configured)
- `historical_demand.csv` — the input data
- `demand_forecast_output.csv` — the full forecast with confidence bounds

## Using your own data

Replace the dummy-data generation with your real export, keeping the same
shape (a date column and a demand column):

```python
historical_df = pd.read_csv("your_export.csv", parse_dates=["date"])

config = ForecastConfig(date_col="date", demand_col="units_sold", forecast_horizon=30)
agent = DemandForecastingAgent(config)
result = agent.run(historical_df)
```

## Troubleshooting

**`AttributeError: 'AIProjectClient' object has no attribute 'inference'`**
You're on `azure-ai-projects` v2.x, which removed the old `.inference`
namespace. This is already handled in the current version of the script
(`get_openai_client()` + `responses.create()`); make sure you're running
the latest version of `DemandForecastingAgent.py`.

**`ConnectionError: Could not resolve host ...`**
`PROJECT_ENDPOINT` is malformed, still set to a placeholder, or unreachable
from your network. The script validates this up front and will tell you
which. Checklist:
1. Re-copy the endpoint from the Foundry project's Overview page — no
   quotes, no trailing spaces, no leftover `<placeholder>` text.
2. Confirm the env var actually holds that value: `echo $env:PROJECT_ENDPOINT`
   (PowerShell) or `echo $PROJECT_ENDPOINT` (bash).
3. Check you're not on a VPN/corporate network blocking
   `*.services.ai.azure.com`. Try `nslookup <hostname>` — if that fails
   too, it's a network issue, not the script.

**`openai.APIConnectionError` / retries then fails**
Same root cause as above — DNS/network can't reach the Foundry endpoint.

**Model call succeeds but returns an error about the deployment**
Double-check `MODEL_DEPLOYMENT` matches the exact deployment name shown
under your project's **Models + endpoints** tab (not just the base model
family name).

## Notes

- `DefaultAzureCredential` is used for auth — locally this resolves via
  `az login` (Azure CLI credential); in production, use a Managed Identity
  or Service Principal instead.
- Prophet is tried first for forecasting; if it's not installed, the
  script automatically falls back to a SARIMAX model with weekly
  seasonality — no code changes needed either way.
