# 🅿️ ParkWise

### Intelligent Campus Parking Prediction & Recommendation Platform

ParkWise is an end-to-end data science application that uses **machine learning to forecast campus parking demand** and recommend suitable parking locations based on **predicted availability, walking distance, and parking cost**.

The platform combines **machine learning, PostgreSQL, geospatial analytics, interactive dashboards, and a context-aware conversational assistant** in a deployed Streamlit application.

🌐 **Live Demo:** https://parkwise-campus-parking.streamlit.app/

---

## Overview

Finding parking on a busy university campus is not simply a question of where parking lots are located. Availability changes throughout the day, and the best option depends on a user's destination, arrival time, walking preference, and cost.

ParkWise turns parking observations into practical decision support through:

- ML-based occupancy forecasting
- destination-aware parking recommendations
- estimated walking times and parking costs
- uncertainty ranges around predictions
- interactive geospatial parking maps
- historical demand analytics
- a context-aware, multi-turn Parking Assistant
- PostgreSQL-backed data storage

---

## Application

### Campus Overview

Monitor estimated parking availability and utilisation across campus through a consolidated dashboard.

![ParkWise Overview](assets/parkwise-overview.png)

### Smart Parking Recommendations

Users select a destination, arrival time, and parking preference. ParkWise evaluates available parking locations using **predicted occupancy, walking distance, and parking cost**.

![Parking Recommendation](assets/parking-recommendation.png)

The recommendation engine identifies the best option, provides an alternative, and displays a likely availability range rather than presenting forecasts as perfectly certain.

### Context-Aware Parking Assistant

The Parking Assistant provides a natural-language interface to ParkWise's forecasting and recommendation functions.

![Parking Assistant](assets/parking-assistant.png)

It supports multi-turn conversations such as:

```text
User: I want to go to the library.

ParkWise: I recommend Library West (P3).

User: Which option is cheapest?

ParkWise: South Campus (P6) is the cheapest option.

User: Which one is closest?

ParkWise: Library West (P3) is the closest option.

User: What if I arrive at 11 AM?

ParkWise recalculates the recommendation using the updated arrival time.
```

The assistant retains relevant conversational context including **destination, arrival time, parking preference, walking constraints, and previously selected parking locations**.

---

## Machine Learning Forecasting

ParkWise uses a **Random Forest Regressor** to estimate future parking occupancy.

Model features include:

- parking-lot identity and capacity
- hour of day
- cyclical time features
- day of week
- month
- weekend indicator
- rainfall
- campus-event indicator
- exam-period indicator

### Demand Forecast

Predictions are generated at **30-minute intervals** to show how parking demand is expected to change throughout the day.

![Demand Forecast](assets/demand-forecast.png)

The forecast interface highlights:

- predicted occupancy
- peak occupancy
- expected peak time
- available spaces at peak
- prediction ranges

### Model Evaluation

The model is evaluated using a **time-ordered 80/20 holdout split** rather than randomly shuffling observations. This better reflects a forecasting scenario in which earlier observations are used to predict later parking behaviour.

| Metric | Result |
|---|---:|
| MAE | 0.0216 |
| RMSE | 0.0275 |
| R² | 0.981 |

![Model Insights](assets/model-insights.png)

Feature-importance analysis is also exposed through the application to make the model's behaviour easier to interpret.

> **Model context:** Performance is measured on the project's simulated dataset, which contains intentionally structured university parking patterns. These metrics demonstrate the modelling pipeline and should not be interpreted as expected performance on real-world parking data.

---

## Geospatial Recommendation Engine

ParkWise combines ML occupancy forecasts with location information to evaluate parking options relative to a user's destination.

![Parking Map](assets/parking-map.png)

The recommendation engine considers three primary factors:

**Predicted availability**  
The ML model estimates future occupancy and expected available spaces.

**Walking distance**  
Geospatial distance between the selected destination and each parking location is converted into an approximate walking time.

**Parking cost**  
Hourly parking rates are incorporated into the ranking process.

Users can prioritise:

- Best overall
- Closest
- Cheapest
- Most available

Walking limits can also be treated as **hard constraints**. If no parking location satisfies the requested walking limit, ParkWise reports this instead of silently recommending an unsuitable option.

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
     ├── Specific lot forecast
     ├── Historical demand
     └── Recommendation explanation
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

For unusual or less structured language, ParkWise can optionally use an LLM to interpret the request and select the appropriate application function.

**The LLM is not the source of parking predictions.** Availability, occupancy, walking distance, pricing, and recommendation results are produced by ParkWise's underlying application logic and ML pipeline.

This keeps numerical responses grounded in application data while still allowing natural conversational interaction.

---

## PostgreSQL Data Layer

ParkWise uses **PostgreSQL** to persist application data across four primary tables:

```text
parking_lots
campus_buildings
occupancy_observations
occupancy_forecasts
```

The prototype database contains **30,240 simulated occupancy observations across six parking locations**.

The database layer supports:

- parking and building metadata
- historical occupancy observations
- persisted parking forecasts
- application queries through SQLAlchemy
- efficient bulk data loading
- fallback operation when PostgreSQL is unavailable

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
| Geospatial Analytics | Geographic distance logic & interactive mapping |
| Conversational AI | Deterministic routing + optional LLM |
| LLM Integration | OpenAI-compatible API / OpenRouter |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## Data & Project Scope

ParkWise is a **portfolio prototype built using simulated university parking observations and illustrative campus coordinates**.

The dataset models realistic factors including:

- time-of-day parking demand
- weekday and weekend behaviour
- parking-lot capacity differences
- rainfall
- campus events
- exam periods

This allows the complete **data → ML → recommendation → application** pipeline to be demonstrated without claiming access to an official university parking feed.

A production implementation could replace the simulated observations with data from:

- parking sensors
- entry/exit counters
- university parking systems
- IoT occupancy feeds
- weather services
- campus event APIs

The ML, database, recommendation, and conversational architecture could then operate on real-world inputs.

---

## Project Highlights

ParkWise demonstrates an integrated workflow across several areas of data science and software engineering:

**Machine Learning**
- feature engineering
- regression modelling
- time-aware model evaluation
- occupancy forecasting
- model interpretation
- uncertainty-aware predictions

**Data Engineering**
- PostgreSQL schema design
- bulk data ingestion
- SQLAlchemy integration
- persistent forecast storage
- fallback data handling

**Analytics**
- historical demand analysis
- interactive dashboards
- occupancy visualisation
- utilisation metrics
- demand heatmaps

**Geospatial Analytics**
- parking-location mapping
- destination-distance calculations
- walking-time estimation
- location-aware recommendations

**AI Engineering**
- natural-language intent interpretation
- deterministic routing
- conversational state
- tool-grounded responses
- hard constraints
- optional LLM integration

**Software Engineering**
- modular Python application design
- separation of ML, database, recommendation, and UI logic
- environment-based configuration
- Git/GitHub workflow
- cloud deployment

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

ParkWise can operate using its bundled fallback dataset without requiring PostgreSQL or an external LLM.

---

## Optional PostgreSQL Configuration

Configure the database connection through an environment variable:

```text
DATABASE_URL=postgresql+psycopg2://USERNAME:PASSWORD@HOST/DATABASE
```

Initialise the database:

```bash
python setup_db.py
```

The setup process creates the required schema and loads the parking observations.

Never commit database credentials or `.env` files to GitHub.

---

## Optional LLM Configuration

The deterministic Parking Assistant can operate without an external LLM.

Optional LLM-based natural-language interpretation can be enabled with:

```text
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=your_model
```

For deployment, credentials should be stored using Streamlit Secrets or another secure environment-variable mechanism.

---

## Future Development

Potential production-oriented extensions include:

- real-time parking sensor integration
- live weather data
- campus event-calendar integration
- GPS-based user location
- real walking-route calculations
- permit-aware recommendations
- model monitoring and drift detection
- automated model retraining
- authentication and personalised parking preferences
- real-time parking availability notifications

---

## Author

**Zeel Shah**  
Master of Data Science — The University of Western Australia

**Live Application:**  
https://parkwise-campus-parking.streamlit.app/

**GitHub:**  
https://github.com/shahzeel26

---

*ParkWise is an educational portfolio prototype and is not an official university parking service.*