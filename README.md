# ParkWise

**Intelligent University Parking Prediction & Recommendation Platform**

ParkWise is a portfolio project that combines a simulated university parking dataset, machine-learning occupancy forecasting, geospatial decision logic, analytics and a conversational parking assistant.

## Key features

- Campus parking overview and availability map
- Parking recommendation based on predicted availability, walking distance and price
- Half-hour parking-demand forecasts
- Prediction ranges to communicate uncertainty
- Historical hourly/weekday analytics and demand heatmap
- Parking Assistant conversational assistant grounded in application functions
- Model-insights page with time-ordered holdout metrics

## Data transparency

The parking occupancy dataset is **synthetic**. It is generated to reproduce realistic university parking behaviour including:

- strong weekday morning arrivals
- weekday/weekend differences
- parking-lot capacity differences
- rain
- campus events
- exam-period effects

The prototype campus coordinates are also illustrative and should not be described as an official live university parking feed.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Architecture

```text
Simulated parking observations
            |
            v
     Feature engineering
            |
            v
     Random Forest model
            |
      +-----+------+
      |            |
      v            v
 Analytics   Occupancy forecast
                   |
                   v
          Recommendation engine
                   |
         +---------+---------+
         v                   v
   Interactive map      Parking Assistant Assistant
```

## Portfolio-safe wording

> Built an end-to-end university parking intelligence prototype using a simulated mobility dataset, machine-learning occupancy forecasting and geospatial recommendation logic to identify suitable parking locations based on predicted availability, walking distance and user preferences.


## Parking Assistant natural-language layer

V3 adds an optional LLM parser. The LLM is used only to interpret intent and extract parameters such as:

- destination
- parking lot
- arrival date/time
- parking preference
- maximum walking time

It does **not** generate parking availability or recommendation scores. Those values come from SmartPark's own ML and geospatial functions.

The app works without an API key using a local rule-based parser.

### Optional LLM setup

Set environment variables before launching:

```powershell
$env:LLM_API_KEY="your-key"
$env:LLM_BASE_URL="https://openrouter.ai/api/v1"
$env:LLM_MODEL="openai/gpt-4o-mini"
python -m streamlit run app.py
```

Do not commit API keys to GitHub.


## PostgreSQL mode

V4 can run in two modes:

### 1. PostgreSQL mode
If `DATABASE_URL` is configured, SmartPark loads:

- parking lots
- campus buildings
- occupancy observations

from PostgreSQL.

Forecasts generated in the Demand Forecast page are also written to `occupancy_forecasts`.

### 2. CSV fallback mode
If PostgreSQL is unavailable, the app still runs using the bundled university parking dataset.

### Local PostgreSQL setup

Create a database:

```sql
CREATE DATABASE smartpark;
```

Then copy `.env.example` to `.env` and set:

```text
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/smartpark
```

Initialize and seed the database:

```bash
python setup_db.py
```

Then run:

```bash
python -m streamlit run app.py
```

When connected successfully, the sidebar shows:

```text
PostgreSQL connected
```

### Why this improves the project

This makes SmartPark a fuller end-to-end system:

```text
University parking observations
            |
            v
        PostgreSQL
            |
            v
   Feature engineering
            |
            v
      ML forecasting
            |
       +----+----+
       |         |
       v         v
  Analytics   Recommendation
       |         |
       +----+----+
            |
            v
      Streamlit app
            |
            v
      Parking Assistant Assistant
```


## V5 professional light UI

V5 introduces a restrained light interface with tighter spacing, white content surfaces, subtle borders, a muted green accent, light maps, simpler charts, cleaner chat styling and a more portfolio-friendly presentation. The design intentionally avoids a dark theme and avoids glossy AI-generated visual patterns.


## V6 final polish

V6 focuses on portfolio readiness:

- fixed clipped page headings
- map legend and clearer marker meaning
- forecast KPI summary
- analytics car-park filter
- cleaner Parking Assistant wording
- model interpretation notes
- consistent light presentation


## V7: Parking Assistant LLM tool calling

V7 upgrades Parking Assistant from local intent parsing to optional real LLM tool calling.

The LLM is **not allowed to generate parking numbers**. It only:

1. interprets the user's question,
2. selects a SmartPark tool,
3. supplies structured arguments,
4. receives the tool result,
5. explains the grounded result naturally.

Supported tools:

- `recommend_parking`
- `predict_lot`
- `historical_peak`

### Example flow

```text
User:
"I have class at the Business School tomorrow at 10.
Where should I park?"

        |
        v

LLM chooses:
recommend_parking(
  destination="Business School",
  arrival_datetime="...",
  preference="Balanced"
)

        |
        v

Python recommendation engine
        |
        v
ML occupancy forecast + geospatial distance
        |
        v
Grounded tool result
        |
        v
LLM explanation
```

### Configure OpenRouter

Create `.env`:

```text
LLM_API_KEY=your_key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
```

Then restart Streamlit.

Never commit `.env` to GitHub.


## V8: Conversational memory and hard constraints

Parking Assistant now maintains a conversation state containing:

- destination
- arrival date/time
- parking preference
- maximum walking time
- last recommended car park

Example:

```text
User: I don't want to walk more than 2 minutes.
State: max_walk_minutes = 2

User: What about the library?
State:
  destination = Main Library
  max_walk_minutes = 2   <- retained
```

Walking limits are treated as hard constraints. If no parking lot satisfies the limit, Parking Assistant tells the user instead of silently violating it.

Supported follow-ups include:

- "what about the library?"
- "cheapest one?"
- "anything closer?"
- "what if I arrive at 11?"
- "will that still have spaces?"


## V9: Intent-aware Parking Assistant

Parking Assistant now distinguishes between several user intents:

```text
"Which parking areas have spaces?"
    -> general availability

"I'm going to the library"
    -> destination recommendation

"Will P3 be full at 11?"
    -> lot forecast

"I don't want to walk more than 2 minutes"
    -> hard walking constraint

"Which one is cheaper?"
    -> preference update

"Why did you recommend that?"
    -> explanation of previous decision
```

General availability no longer silently assumes the default destination.

Recommendation and availability tables are hidden inside expanders so the conversational interface stays clean.


## V10: Deterministic-first conversational routing

V10 fixes unreliable follow-up behaviour by placing a deterministic intent/context layer before the LLM.

```text
User message
      |
      v
Deterministic parser
      |
      +-- general availability
      +-- destination change
      +-- walking constraint
      +-- time change
      +-- cheapest/closest/most available
      +-- specific lot forecast
      +-- historical demand
      +-- explain previous recommendation
      |
      v
Update conversation state
      |
      v
SmartPark Python tool
      |
      v
ML + geospatial logic + PostgreSQL
      |
      v
Grounded response

Only unknown/unusual language falls back to the LLM.
```

This makes multi-turn parking conversations reliable even if the LLM API is unavailable.


## V11 routing fix

V11 fixes a wiring issue in V10 where the Parking Assistant page still called the legacy assistant function.
The UI now directly calls `route_and_answer()`, so deterministic multi-turn routing and context memory are actually active.


## Final deployment polish

The final version removes developer-facing connection badges from the normal interface, moves synthetic-data disclosure into Model Insights, simplifies user-facing recommendation tables, and presents technical architecture in the appropriate model/system section.


## Final UI fixes

- fixed the Parking Assistant navigation mismatch that caused a blank page
- fixed Find Parking dataframe columns after removing internal recommendation scores
- completed ParkWise/Parking Assistant naming throughout the interface
- mapped user-friendly preference labels to the internal recommendation engine
- removed developer-facing LLM/database implementation labels from normal user screens
