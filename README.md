# 🅿️ ParkWise

### ML-Powered Campus Parking Intelligence Platform

ParkWise is an end-to-end data science application that forecasts campus parking demand and recommends suitable parking locations based on **predicted availability, walking distance, and parking cost**.

The platform combines **machine learning, PostgreSQL, geospatial analytics, interactive dashboards, and a context-aware conversational assistant** in a deployed Streamlit application.

🌐 **Live Demo:** https://parkwise-campus-parking.streamlit.app/

---

## Overview

Finding parking on a busy university campus is not simply a question of where parking lots are located. Availability changes throughout the day, and the best option depends on a user's destination, arrival time, walking preference, and cost.

ParkWise turns historical parking observations into decision support by providing:

- ML-based occupancy forecasting
- destination-aware parking recommendations
- estimated walking times and parking costs
- uncertainty ranges around predictions
- interactive geospatial parking maps
- historical demand analytics
- a multi-turn Parking Assistant
- PostgreSQL-backed data storage

---

## Application

### Campus Overview

Monitor estimated parking availability and utilisation across campus.

![ParkWise Overview](assets/parkwise-overview.png)

### Smart Parking Recommendations

ParkWise evaluates parking locations using predicted availability, walking distance, parking cost, and the user's selected preference.

![Parking Recommendation](assets/parking-recommendation.png)

Rather than returning only one result, the system provides an alternative parking option and a likely availability range to communicate forecast uncertainty.

### Context-Aware Parking Assistant

Users can interact with the recommendation and forecasting system using natural language.

![Parking Assistant](assets/parking-assistant.png)

Example:

```text
User: I want to go to the library.

ParkWise: I recommend Library West (P3).

User: Which option is cheapest?

ParkWise: South Campus (P6) is the cheapest option.

User: Which one is closest?

ParkWise: Library West (P3) is the closest option.

User: What if I arrive at 11 AM?

ParkWise recalculates the forecast using the updated arrival time.
```

The assistant maintains conversational context including destination, arrival time, parking preference, walking constraints, and the previously recommended parking location.

---

## Machine Learning Forecasting

ParkWise uses a **Random Forest Regressor** to estimate parking occupancy throughout the day.

Features include:

- parking-lot identity and capacity
- hour and cyclical time features
- day of week and month
- weekend indicator
- rainfall
- campus-event indicator
- exam-period indicator

### Demand Forecast

Predictions are generated at half-hour intervals to show how parking demand is expected to change throughout the day.

![Demand Forecast](assets/demand-forecast.png)

The interface highlights:

- predicted occupancy
- peak occupancy
- expected peak time
- available spaces at peak
- prediction ranges

### Model Evaluation

The model is evaluated using a **time-ordered 80/20 holdout split** rather than randomly shuffling observations.

| Metric | Result |
|---|---:|
| MAE | 0.0216 |
| RMSE | 0.0275 |
| R² | 0.981 |

![Model Insights](assets/model-insights.png)

Feature-importance analysis is also exposed through the application to make the model's behaviour easier to interpret.

> **Model context:** Performance is measured on the project's simulated dataset, which contains intentionally structured university parking patterns. These metrics therefore demonstrate the prototype pipeline and should not be interpreted as expected performance on real-world parking data.

---

## Geospatial Recommendation Engine

ParkWise combines occupancy forecasts with location information to evaluate parking options relative to a user's destination.

![Parking Map](assets/parking-map.png)

The recommendation engine considers three primary factors:

**Predicted availability** — estimated using the ML forecasting model.

**Walking distance** — calculated from the destination and parking-location coordinates and converted into estimated walking time.

**Parking cost** — hourly rates are incorporated into the ranking process.

Users can prioritise:

- Best overall
- Closest
- Cheapest
- Most available

Walking limits can also be treated as hard constraints. If no parking location satisfies the requested limit, the system reports this instead of silently violating the preference.

---

## System Architecture

```text
                Parking Observations
                        │
                        ▼
                    PostgreSQL
                        │
                        ▼
                Feature Engineering
                        │
                        ▼
              Random Forest Regressor
                        │
                        ▼
                Occupancy Forecast
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
         Analytics         Recommendation Engine
                                   │
                         ┌─────────┴─────────┐
                         │                   │
                         ▼                   ▼
                  Geospatial Logic    Parking Assistant
                         │                   │
                         └─────────┬─────────┘
                                   ▼
                              Streamlit UI
```

---

## Parking Assistant Architecture

The assistant follows a **deterministic-first, tool-based architecture**.

```text
User Message
     │
     ▼
Intent & Context Routing
     │
     ├── Destination
     ├── Arrival time
     ├── Walking constraint
     ├── Cheapest / closest
     ├── Availability
     ├── Lot forecast
     └── Historical demand
     │
     ▼
Conversation State
     │
     ▼
ParkWise Tools
     │
     ├── recommend_parking()
     ├── predict_lot()
     └── historical_peak()
     │
     ▼
ML + Recommendation + Geospatial Logic
     │
     ▼
Grounded Response
```

An optional LLM layer can interpret unusual natural-language requests and select the appropriate ParkWise function.

**The LLM is not the source of parking predictions.** Availability, occupancy, walking distance, and recommendation results are produced by the underlying application logic and ML pipeline.

This design reduces hallucination risk and keeps numerical responses grounded in application data.

---

## PostgreSQL Data Layer

ParkWise uses PostgreSQL to persist application data across four primary tables:

```text
parking_lots
campus_buildings
occupancy_observations
occupancy_forecasts
```

The prototype database contains **30,240 simulated occupancy observations across six parking locations**.

A bulk-loading workflow is used to initialise the database efficiently, while CSV fallback support allows the application to continue operating when PostgreSQL is unavailable.

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Application | Streamlit |
| Machine Learning | scikit-learn |
| Model | Random Forest Regressor |
| Data Processing | pandas, NumPy |
| Database | PostgreSQL |
| Database Integration | SQLAlchemy, psycopg2 |
| Visualisation | Plotly |
| Geospatial Analytics | Folium, geographic distance logic |
| Conversational AI | Deterministic routing + optional LLM |
| LLM Integration | OpenAI-compatible API / OpenRouter |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## Data & Project Scope

ParkWise is a **portfolio prototype built using simulated university parking observations and illustrative campus coordinates**.

The dataset models factors such as:

- time-of-day parking demand
- weekday and weekend behaviour
- parking-lot capacity
- rainfall
- campus events
- exam periods

This allows the complete ML, database, recommendation, analytics, and application pipeline to be demonstrated without claiming access to an official university parking feed.

A production implementation could replace the simulated observations with data from parking sensors, entry/exit counters, cameras, university systems, weather services, and event APIs.

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/shahzeel26/parkwise-campus-parking.git
cd parkwise-campus-parking
```

### 2. Create a virtual environment

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

ParkWise can operate using its bundled fallback dataset without requiring PostgreSQL or an LLM API.

---

## Optional PostgreSQL Configuration

Configure the database connection:

```text
DATABASE_URL=postgresql+psycopg2://USERNAME:PASSWORD@HOST/DATABASE
```

Initialise the schema and seed the database:

```bash
python setup_db.py
```

Never commit database credentials or `.env` files to the repository.

---

## Optional LLM Configuration

The deterministic Parking Assistant works without an external LLM.

Optional LLM interpretation can be enabled with:

```text
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=your_model
```

For deployment, credentials should be stored using Streamlit Secrets or another secure environment-variable mechanism.

---

## Project Highlights

ParkWise demonstrates an integrated workflow across:

**Machine Learning:** feature engineering, regression modelling, time-aware evaluation, prediction intervals, and model interpretation.

**Data Engineering:** PostgreSQL schema design, bulk ingestion, persistent forecasts, SQLAlchemy integration, and fallback data handling.

**Analytics:** interactive visualisation, demand analysis, utilisation metrics, and historical parking patterns.

**Geospatial Analytics:** location mapping, destination-distance calculations, walking-time estimation, and location-aware recommendations.

**AI Engineering:** intent routing, conversational state, tool-based responses, hard constraints, and optional LLM integration.

**Software Engineering:** modular Python development, configuration management, Git/GitHub workflow, database integration, and cloud deployment.

---

## Future Development

Potential extensions include real-time parking sensors, live weather and campus-event APIs, GPS-based user location, dynamic walking routes, model-drift monitoring, automated retraining, authentication, and personalised parking preferences.

---

## Author

**Zeel Shah**  
Master of Data Science — The University of Western Australia

GitHub: https://github.com/shahzeel26

---

*ParkWise is an educational portfolio prototype and is not an official university parking service.*