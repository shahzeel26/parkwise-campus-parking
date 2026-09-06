# ParkWise

### ML-Powered Campus Parking Intelligence & Decision-Support Platform

**Python · Streamlit · PostgreSQL · Machine Learning · Geospatial Analytics · Conversational AI**

[Live Demo](https://parkwise-campus-parking.streamlit.app/) · [GitHub Repository](https://github.com/shahzeel26/parkwise-campus-parking)

---

## Overview

ParkWise is an end-to-end campus parking intelligence platform that combines **machine-learning occupancy forecasting, geospatial analysis, PostgreSQL, recommendation logic, interactive analytics, and a conversational parking assistant**.

The platform is designed around a practical question:

> **Given where I am going, when I will arrive, and what I care about most, where should I park?**

Instead of simply displaying parking data, ParkWise predicts future parking demand and helps users make parking decisions based on:

- predicted space availability
- walking distance to their destination
- parking cost
- arrival time
- user preferences
- maximum walking constraints

---

## Live Application

**Try ParkWise:**  
https://parkwise-campus-parking.streamlit.app/

The deployed application includes:

- Overview dashboard
- Interactive parking map
- Parking recommendation engine
- Demand forecasting
- Historical analytics
- Conversational Parking Assistant
- Model performance and feature insights

---

## Key Features

### Parking Availability Dashboard

Provides a campus-wide overview of:

- available spaces
- overall parking utilisation
- busiest parking areas
- typical demand peaks
- current parking status

---

### Interactive Parking Map

Displays parking areas geographically using PyDeck.

Each parking location shows:

- estimated available spaces
- occupancy level
- total capacity
- permit information
- hourly parking rate

Parking status is visually classified as:

- Available
- Moderate
- Busy
- Nearly full

---

### Intelligent Parking Recommendation

Users select:

- destination
- arrival date
- arrival time
- parking preference
- expected rainfall
- campus-event conditions

ParkWise then evaluates the available parking areas and recommends the most suitable option.

Users can optimise for:

- **Best overall**
- **Shortest walk**
- **Most availability**
- **Lowest cost**

The recommendation combines **ML predictions, geospatial distance, parking availability, and pricing**.

---

### Demand Forecasting

ParkWise predicts parking occupancy at **30-minute intervals** throughout the day.

For each parking area, the application estimates:

- predicted occupancy
- expected available spaces
- peak occupancy
- expected peak time
- likely availability range

---

### Parking Analytics

The analytics dashboard explores historical parking behaviour through:

- hourly demand patterns
- weekday utilisation
- parking-lot comparisons
- demand heatmaps
- peak and quiet periods

---

## Parking Assistant

ParkWise includes a conversational decision-support assistant that allows users to interact with the parking system using natural language.

Example conversation:

```text
User:
I want to go to the library.

ParkWise:
Recommends a suitable parking area using predicted availability,
walking distance and cost.

User:
Which option is cheapest?

ParkWise:
Compares the available parking options by hourly price.

User:
Which one is closest?

ParkWise:
Compares the options by walking distance.

User:
I don't want to walk more than 2 minutes.

ParkWise:
Applies the walking limit as a hard constraint and recalculates.

User:
What if I arrive at 11 AM?

ParkWise:
Retains the conversation context and recalculates for 11 AM.
```

The assistant can handle:

- parking recommendations
- price comparisons
- walking-distance comparisons
- availability comparisons
- specific parking-lot forecasts
- arrival-time changes
- maximum walking constraints
- historical parking questions
- recommendation explanations

---

## Grounded AI Architecture

The Parking Assistant is designed as a **tool-grounded assistant rather than a free-form chatbot**.

Parking availability and recommendation numbers are calculated by ParkWise's own ML and decision functions.

The LLM does **not** invent parking availability, prices, walking distances, or recommendation scores.

```text
User Question
      |
      v
Intent & Context Router
      |
      v
Conversation State
      |
      v
Parking Tool
      |
      +-----------------------------+
      |              |              |
      v              v              v
Recommendation   Lot Forecast   Comparison
      |              |              |
      +--------------+--------------+
                     |
                     v
           ML + Geospatial Logic
                     |
                     v
             Grounded Response
```

ParkWise uses a **deterministic-first routing approach** for common parking requests.

An optional LLM layer can handle less predictable natural-language requests.

This means the core parking functionality can continue to operate even when the LLM API is unavailable.

---

## Machine Learning

ParkWise uses a **Random Forest regression model** to predict parking occupancy.

### Features

The model uses features including:

- parking-lot identity
- capacity
- hour of day
- day of week
- month
- weekend indicator
- cyclical time features
- rainfall
- campus-event flag
- exam-period flag

### Evaluation

The model is evaluated using a **time-ordered 80/20 holdout split** rather than a random split.

This better represents the real forecasting problem:

> Train using earlier observations and evaluate predictions on later observations.

Current evaluation on the synthetic university parking dataset:

| Metric | Result |
|---|---:|
| MAE | 0.0216 |
| RMSE | 0.0275 |
| R² | 0.981 |

These results apply to the project's **synthetic dataset** and should not be interpreted as performance on a real university parking system.

---

## Recommendation Engine

For a selected destination and arrival time, ParkWise:

1. predicts occupancy for each parking area
2. estimates the number of available spaces
3. calculates approximate walking distance to the destination
4. considers hourly parking price
5. normalises availability, distance and cost
6. ranks parking areas according to the user's preference
7. applies constraints such as maximum walking time

This allows ParkWise to answer different decision questions such as:

```text
Where should I park?

Which option is cheapest?

Which parking area is closest?

Which one is likely to have the most spaces?
```

---

## PostgreSQL Data Layer

ParkWise uses **PostgreSQL** as its primary application database.

The deployed application uses a hosted **Neon PostgreSQL** database.

The database contains:

- **6 parking areas**
- **30,240 occupancy observations**
- campus-building metadata
- generated parking forecasts

The database layer is accessed through **SQLAlchemy**.

For efficient cloud database setup, ParkWise uses PostgreSQL's **COPY** operation to bulk-load occupancy observations rather than inserting thousands of rows individually.

If PostgreSQL is unavailable, the application can fall back to the bundled demonstration dataset.

---

## System Architecture

```text
Synthetic Parking Observations
            |
            v
      PostgreSQL / Neon
            |
            v
      Data Processing
            |
            v
     Feature Engineering
            |
            v
   Random Forest Forecasting
            |
      +-----+------+
      |            |
      v            v
  Analytics    Occupancy Forecast
                   |
                   v
          Recommendation Engine
                   |
        +----------+----------+
        |                     |
        v                     v
 Interactive UI       Parking Assistant
        |                     |
        +----------+----------+
                   |
                   v
             Streamlit App
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Web Application | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn, Random Forest |
| Database | PostgreSQL, Neon |
| Database Integration | SQLAlchemy, psycopg2 |
| Visualisation | Plotly |
| Geospatial Visualisation | PyDeck |
| Conversational AI | Intent routing + optional LLM tool calling |
| LLM Integration | OpenAI-compatible API / OpenRouter |
| Testing | pytest |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```text
parkwise-campus-parking/
│
├── app.py
├── setup_db.py
├── requirements.txt
├── README.md
│
├── data/
│
├── sql/
│   └── schema.sql
│
├── src/
│   ├── conversation.py
│   ├── data.py
│   ├── database.py
│   ├── intent_router.py
│   ├── model.py
│   ├── recommender.py
│   ├── tool_assistant.py
│   └── ui.py
│
└── tests/
```

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/shahzeel26/parkwise-campus-parking.git
cd parkwise-campus-parking
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file if you want to use PostgreSQL and/or the optional LLM integration.

```text
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST/DATABASE?sslmode=require

LLM_API_KEY=your_key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=your_model_name
```

Never commit `.env`, database passwords, or API keys to GitHub.

### 5. Initialise PostgreSQL

```bash
python setup_db.py
```

### 6. Run ParkWise

```bash
python -m streamlit run app.py
```

---

## Data Transparency

The parking occupancy observations used in this project are **synthetically generated**.

The dataset is designed to reproduce realistic university parking behaviour including:

- weekday morning arrival peaks
- weekday/weekend differences
- parking-lot capacity differences
- rainfall effects
- campus events
- exam-period effects

The parking and building coordinates are also illustrative for the prototype.

ParkWise should therefore be viewed as a **parking intelligence prototype**, not an official live university parking service.

The system architecture is designed so that the synthetic data layer could later be replaced with:

- parking sensors
- access-control data
- parking APIs
- IoT occupancy feeds
- university parking systems

without redesigning the entire application.

---

## Why I Built ParkWise

The goal of ParkWise was to build more than a standalone machine-learning notebook.

The project demonstrates how machine learning can be integrated into a complete decision-support product involving:

- data engineering
- PostgreSQL
- machine-learning forecasting
- geospatial analytics
- recommendation systems
- interactive dashboards
- conversational AI
- cloud deployment

---

## Future Improvements

Potential extensions include:

- live parking-sensor integration
- real walking-route calculations
- live weather API integration
- university event-calendar integration
- permit-aware recommendations
- authentication
- real-time occupancy updates
- model monitoring and drift detection
- automated model retraining

---

## Author

**Zeel Shah**  
Master of Data Science  
University of Western Australia

**Live Demo:** https://parkwise-campus-parking.streamlit.app/

**GitHub:** https://github.com/shahzeel26/parkwise-campus-parking