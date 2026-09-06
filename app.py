from __future__ import annotations
from datetime import datetime, timedelta, time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.data import load_demo_data, LOT_META, BUILDINGS
from src.model import train_model, predict_occupancy
from src.recommender import recommend_parking
from src.tool_assistant import llm_configured
from src.conversation import new_state, readable_state
from src.intent_router import route_and_answer
from src.database import get_engine, db_available, load_from_database, save_forecast
from src.ui import inject_css, page_header, section_header

st.set_page_config(page_title="ParkWise",page_icon="🅿️",layout="wide")

inject_css()

@st.cache_data(show_spinner=False)
def get_data():
    return load_demo_data()

@st.cache_resource(show_spinner="Training parking-demand model…")
def get_model(_df):
    return train_model(_df)

engine = get_engine()
use_database = False

if engine is not None and db_available():
    try:
        lots, buildings, history = load_from_database(engine)
        if len(history) > 0:
            use_database = True
        else:
            history = get_data()
            lots = LOT_META.copy()
            buildings = BUILDINGS.copy()
    except Exception:
        history = get_data()
        lots = LOT_META.copy()
        buildings = BUILDINGS.copy()
else:
    history = get_data()
    lots = LOT_META.copy()
    buildings = BUILDINGS.copy()

# Add time features if data came from PostgreSQL.
if "hour" not in history.columns:
    from src.data import add_features
    history = add_features(history)

model, metrics = get_model(history)

st.sidebar.markdown("## 🅿️ ParkWise")
st.sidebar.caption("Campus Parking Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Overview","Parking Map","Find Parking","Demand Forecast","Analytics","Parking Assistant","Model Insights"],
)

latest_ts = history["timestamp"].max()
snapshot = history[history["timestamp"]==latest_ts].merge(
    lots[["lot_id","name","lat","lon","permit_type","hourly_rate"]],on="lot_id",how="left"
)

def status_label(rate):
    if rate < .60: return "Available"
    if rate < .80: return "Moderate"
    if rate < .90: return "Busy"
    return "Nearly full"

def status_rgb(rate):
    if rate < .60: return [52,168,83,210]
    if rate < .80: return [234,179,8,220]
    if rate < .90: return [239,124,38,225]
    return [220,53,69,230]

if page == "Overview":
    page_header("Overview", "ParkWise helps students and staff compare campus parking using predicted availability, walking distance and parking cost.")
    total_capacity = int(snapshot["capacity"].sum())
    total_occupied = int(snapshot["occupied"].sum())
    total_available = total_capacity-total_occupied
    overall = total_occupied/total_capacity
    busiest = snapshot.sort_values("occupancy_rate",ascending=False).iloc[0]

    hourly = history.assign(hour_int=history["timestamp"].dt.hour).groupby("hour_int")["occupancy_rate"].mean()
    next_peak_hour = int(hourly.idxmax())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Spaces available",f"{total_available:,}")
    c2.metric("Campus occupied",f"{overall:.0%}")
    c3.metric("Highest current occupancy",f"{busiest['lot_id']} · {busiest['occupancy_rate']:.0%}")
    c4.metric("Typical weekday peak",f"{next_peak_hour:02d}:00")

    left,right = st.columns([1.2,1])
    with left:
        st.subheader("Campus parking status")
        plot_df = snapshot.copy()
        plot_df["status"] = plot_df["occupancy_rate"].apply(status_label)
        plot_df["color"] = plot_df["occupancy_rate"].apply(status_rgb)
        plot_df["marker_text"] = plot_df["available"].astype(int).astype(str)

        layer = pdk.Layer(
            "ScatterplotLayer",data=plot_df,get_position="[lon, lat]",get_radius=72,
            get_fill_color="color",pickable=True,auto_highlight=True
        )
        text_layer = pdk.Layer(
            "TextLayer",data=plot_df,get_position="[lon, lat]",get_text="marker_text",
            get_size=16,get_color=[255,255,255,255],get_alignment_baseline="'center'"
        )
        view = pdk.ViewState(
            latitude=float(plot_df["lat"].mean()),longitude=float(plot_df["lon"].mean()),
            zoom=14.4,pitch=0
        )
        st.pydeck_chart(pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            layers=[layer,text_layer],initial_view_state=view,
            tooltip={"html":"<b>{name} ({lot_id})</b><br/>Available: {available}<br/>Occupancy: {occupancy_rate}<br/>Permit: {permit_type}"}
        ),use_container_width=True)

    with right:
        st.subheader("Utilisation by car park")
        chart_df = snapshot.sort_values("occupancy_rate").copy()
        chart_df["occupancy_pct"] = chart_df["occupancy_rate"]*100
        fig = px.bar(chart_df,x="occupancy_pct",y="name",orientation="h",text="occupancy_pct",
                     labels={"occupancy_pct":"Occupancy (%)","name":""})
        fig.update_traces(texttemplate="%{text:.0f}%")
        fig.update_xaxes(range=[0,100])
        fig.update_layout(height=430,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    st.info("Use **Find Parking** to choose a destination and arrival time and compare predicted parking options.")

elif page == "Parking Map":
    page_header("Parking Map", "View parking locations and estimated space availability across campus.")
    st.subheader("Estimated parking availability")
    st.markdown("""
    <div class="sp-legend">
      <span><span class="sp-dot sp-green"></span>Available</span>
      <span><span class="sp-dot sp-amber"></span>Moderate</span>
      <span><span class="sp-dot sp-orange"></span>Busy</span>
      <span><span class="sp-dot sp-red"></span>Nearly full</span>
      <span>Marker number = available spaces</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Latest parking snapshot: {latest_ts:%d %b %Y, %I:%M %p}")

    map_df = snapshot.copy()
    map_df["status"] = map_df["occupancy_rate"].apply(status_label)
    map_df["color"] = map_df["occupancy_rate"].apply(status_rgb)
    map_df["marker_text"] = map_df["available"].astype(int).astype(str)

    layer = pdk.Layer(
        "ScatterplotLayer",data=map_df,get_position="[lon, lat]",get_radius=88,
        get_fill_color="color",pickable=True,auto_highlight=True
    )
    text_layer = pdk.Layer(
        "TextLayer",data=map_df,get_position="[lon, lat]",get_text="marker_text",
        get_size=17,get_color=[255,255,255,255],get_alignment_baseline="'center'"
    )
    st.pydeck_chart(pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        layers=[layer,text_layer],
        initial_view_state=pdk.ViewState(
            latitude=float(map_df["lat"].mean()),longitude=float(map_df["lon"].mean()),zoom=14.3
        ),
        tooltip={"html":"<b>{name} ({lot_id})</b><br/>Status: {status}<br/>Available: {available}<br/>Capacity: {capacity}<br/>Permit: {permit_type}<br/>Rate: ${hourly_rate}/h"}
    ),use_container_width=True)

    table = map_df[["lot_id","name","status","capacity","occupied","available","occupancy_rate"]].copy()
    table["car_park"] = table["name"] + " (" + table["lot_id"] + ")"
    table["occupancy_pct"] = (table["occupancy_rate"]*100).round().astype(int)
    st.dataframe(
        table[["car_park","status","capacity","occupied","available","occupancy_pct"]]
        .sort_values("available",ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "car_park":"Car park",
            "status":"Status",
            "capacity":"Capacity",
            "occupied":"Occupied",
            "available":"Available",
            "occupancy_pct":st.column_config.NumberColumn("Occupancy",format="%d%%")
        }
    )

elif page == "Find Parking":
    page_header("Find Parking", "Compare parking options using predicted availability, walking distance and price.")
    st.subheader("Find the best parking option")
    c1,c2,c3 = st.columns(3)
    destination = c1.selectbox("Destination",buildings["building"].tolist())
    arrival_date = c2.date_input("Arrival date",value=(datetime.now()+timedelta(days=1)).date())
    arrival_time = c3.time_input("Arrival time",value=time(9,30),step=1800)

    c4,c5,c6 = st.columns(3)
    preference_label = c4.selectbox(
        "Preference",
        ["Best overall","Shortest walk","Most availability","Lowest cost"]
    )
    preference_map = {
        "Best overall": "Balanced",
        "Shortest walk": "Closest",
        "Most availability": "Highest availability",
        "Lowest cost": "Cheapest",
    }
    preference = preference_map[preference_label]
    rain_mm = c5.number_input("Expected rain (mm)",0.0,50.0,0.0,0.5)
    event_flag = c6.toggle("Large campus event")

    when = datetime.combine(arrival_date,arrival_time)
    recs = recommend_parking(
        model,metrics,lots,buildings,destination,when,
        preference=preference,rain_mm=rain_mm,event_flag=int(event_flag)
    )
    top,second = recs.iloc[0],recs.iloc[1]

    st.markdown(f"""
    <div class="recommend-card">
      <div class="sp-kicker">Best parking option</div>
      <h3 style="margin:.2rem 0 .55rem 0">{top['name']} ({top['lot_id']})</h3>
      <b>~{int(top['available'])} spaces expected</b> &nbsp;·&nbsp;
      {top['walk_min']:.0f} min walk &nbsp;·&nbsp;
      {top['occupancy_rate']:.0%} occupied &nbsp;·&nbsp;
      ${top['hourly_rate']:.2f}/hr
      <br/><br/>
      <span class="small-muted">
      Likely availability: {int(top['available_low'])}–{int(top['available_high'])} spaces.
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"**Why we recommend it:** {top['name']} is the strongest match for your "
        f"**{preference_label.lower()}** preference, considering predicted availability, "
        f"walking distance and parking cost. "
        f"**Alternative:** {second['name']} ({second['lot_id']})."
    )

    show = recs[["lot_id","name","available","available_low","available_high","occupancy_pct","walk_min","hourly_rate"]].copy()
    show["car_park"] = show["name"] + " (" + show["lot_id"] + ")"
    show["availability_range"] = show.apply(
        lambda r: f"{int(r.available_low)}–{int(r.available_high)}", axis=1
    )
    st.dataframe(
        show[["car_park","available","availability_range","occupancy_pct","walk_min","hourly_rate"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "car_park":"Car park",
            "available":st.column_config.NumberColumn("Expected spaces"),
            "availability_range":st.column_config.TextColumn("Likely range"),
            "occupancy_pct":st.column_config.NumberColumn("Occupancy",format="%d%%"),
            "walk_min":st.column_config.NumberColumn("Walk",format="%.1f min"),
            "hourly_rate":st.column_config.NumberColumn("Price",format="$%.2f/hr"),
        }
    )

    dest = buildings[buildings["building"]==destination].iloc[0]
    map_recs = recs.copy()
    map_recs["color"] = map_recs["occupancy_rate"].apply(status_rgb)
    layers = [
        pdk.Layer("ScatterplotLayer",data=map_recs,get_position="[lon, lat]",get_radius=74,
                  get_fill_color="color",pickable=True,auto_highlight=True),
        pdk.Layer("ScatterplotLayer",data=pd.DataFrame([dest]),get_position="[lon, lat]",get_radius=50,
                  get_fill_color=[64,105,225,240],pickable=True),
    ]
    st.pydeck_chart(pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=float(dest["lat"]),longitude=float(dest["lon"]),zoom=14.5),
        tooltip={"html":"<b>{name}</b><br/>Available: {available}<br/>Walk: {walk_min} min"}
    ),use_container_width=True)

elif page == "Demand Forecast":
    page_header("Demand Forecast", "Estimate how busy a selected car park will be throughout the day.")
    st.subheader("Parking demand forecast")
    selected_lot = st.selectbox(
        "Car park",lots["lot_id"].tolist(),
        format_func=lambda x:f"{x} · {lots.loc[lots.lot_id==x,'name'].iloc[0]}"
    )
    forecast_date = st.date_input("Forecast date",value=(datetime.now()+timedelta(days=1)).date())
    cap = int(lots.loc[lots["lot_id"]==selected_lot,"capacity"].iloc[0])

    rows = []
    for h in range(7,21):
        for minute in [0,30]:
            when = datetime.combine(forecast_date,time(h,minute))
            p = predict_occupancy(
                model,selected_lot,cap,when,
                residual_std=metrics.get("residual_std",0.04)
            )
            rows.append({"time":when,"predicted_occupancy":p["occupancy_rate"],"available":p["available"]})
    fc = pd.DataFrame(rows)

    if use_database:
        forecast_payload = []
        for _, r in fc.iterrows():
            occ = float(r["predicted_occupancy"])
            occupied = int(round(occ * cap))
            forecast_payload.append({
                "lot_id": selected_lot,
                "forecast_for": r["time"],
                "predicted_occupied": occupied,
                "predicted_available": int(cap - occupied),
                "predicted_occupancy_rate": occ,
            })
        try:
            save_forecast(engine, forecast_payload, model_version="rf_v1")
        except Exception:
            pass

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc["time"],y=fc["predicted_occupancy"],
        mode="lines+markers",name="Predicted occupancy"
    ))
    fig.update_yaxes(tickformat=".0%",range=[0,1],title="Occupancy")
    fig.update_xaxes(title="")
    fig.update_layout(height=440,margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    peak = fc.loc[fc["predicted_occupancy"].idxmax()]

    peak_occ = float(peak["predicted_occupancy"])
    peak_available = int(peak["available"])
    f1, f2, f3 = st.columns(3)
    f1.metric("Peak occupancy", f"{peak_occ:.0%}")
    f2.metric("Peak time", f"{peak['time']:%I:%M %p}")
    f3.metric("Spaces available at peak", f"{peak_available}")
    st.info(
        f"Predicted peak: **{peak['time']:%I:%M %p}** at approximately "
        f"**{peak['predicted_occupancy']:.0%} occupancy**, leaving about **{int(peak['available'])} spaces**."
    )

elif page == "Analytics":
    page_header("Analytics", "Explore hourly, weekday and campus parking demand patterns.")
    st.subheader("Historical parking analytics")
    selected_scope = st.selectbox(
        "Analytics scope",
        ["All car parks"] + [f"{r['lot_id']} · {r['name']}" for _, r in lots.iterrows()]
    )
    hist = history.copy()
    if selected_scope != "All car parks":
        selected_lot_id = selected_scope.split(" · ")[0]
        hist = hist[hist["lot_id"] == selected_lot_id].copy()
    hist["hour_int"] = hist["timestamp"].dt.hour
    hist["day_name"] = hist["timestamp"].dt.day_name()

    a1,a2 = st.columns(2)
    with a1:
        section_header("Average occupancy by hour")
        hourly = hist.groupby("hour_int")["occupancy_rate"].mean().reset_index()
        fig = px.line(hourly,x="hour_int",y="occupancy_rate",markers=True,
                      labels={"hour_int":"Hour","occupancy_rate":"Average occupancy"})
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=390,margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    with a2:
        section_header("Average occupancy by weekday")
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        daily = hist.groupby("day_name")["occupancy_rate"].mean().reindex(order).dropna().reset_index()
        fig = px.bar(daily,x="day_name",y="occupancy_rate",
                     labels={"day_name":"","occupancy_rate":"Average occupancy"})
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=390,margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    st.subheader("Demand heatmap")
    heat = hist.pivot_table(index="day_name",columns="hour_int",values="occupancy_rate",aggfunc="mean")
    heat = heat.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    fig = px.imshow(heat,aspect="auto",labels={"x":"Hour","y":"","color":"Occupancy"},
                    color_continuous_scale="RdYlGn_r",zmin=0,zmax=1)
    fig.update_layout(height=430,margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    peak_row = hourly.loc[hourly["occupancy_rate"].idxmax()]
    lot_use = hist.groupby("lot_id")["occupancy_rate"].mean().sort_values(ascending=False)
    st.markdown(
        f"**Operational insight:** average demand peaks around **{int(peak_row['hour_int']):02d}:00**. "
        f"The highest-utilisation lot is **{lot_use.index[0]}** at **{lot_use.iloc[0]:.0%} average occupancy**."
    )

elif page == "Parking Assistant":
    page_header("Parking Assistant", "Ask about parking availability, compare car parks, or find the best option based on your destination, arrival time and preferences.")
    c1,c2,c3 = st.columns(3)
    dest = c1.selectbox("Destination",buildings["building"].tolist(),key="chatdest")
    chat_date = c2.date_input("Arrival date",value=(datetime.now()+timedelta(days=1)).date(),key="chatdate")
    chat_time = c3.time_input("Arrival time",value=time(9,30),step=1800,key="chattime")
    default_when = datetime.combine(chat_date,chat_time)

    if "parkai_context" not in st.session_state:
        st.session_state.parkai_context = new_state(dest, default_when)

    # Keep defaults synced only before the user has changed them through conversation.
    if not st.session_state.parkai_context.get("destination"):
        st.session_state.parkai_context["destination"] = dest
    if not st.session_state.parkai_context.get("arrival_datetime"):
        st.session_state.parkai_context["arrival_datetime"] = default_when.isoformat(timespec="minutes")

    st.caption("Current preferences: " + readable_state(st.session_state.parkai_context))

    cclear, _ = st.columns([1,4])
    if cclear.button("Clear conversation"):
        st.session_state.smartpark_messages = []
        st.session_state.parkai_context = new_state(dest, default_when)
        st.rerun()

    st.markdown("**Try questions like:**")
    st.caption(
        "“I have class at the Business School tomorrow at 10 AM. Where should I park?” · "
        "“I don't want to walk more than 5 minutes.” · "
        "“Will P1 be full at 9:30 AM?”"
    )

    if "smartpark_messages" not in st.session_state:
        st.session_state.smartpark_messages = []

    for m in st.session_state.smartpark_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask about parking..."):
        st.session_state.smartpark_messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        reply, payload = route_and_answer(
            prompt,
            st.session_state.parkai_context,
            model,
            metrics,
            history,
            lots,
            buildings,
            dest,
            default_when,
        )

        if payload.get("conversation_state"):
            st.session_state.parkai_context = payload["conversation_state"]

        if payload.get("dataframe") is not None and payload.get("data") is None:
            payload["data"] = payload["dataframe"]

        st.session_state.smartpark_messages.append({"role":"assistant","content":reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

            if payload.get("type") == "availability" and payload.get("data") is not None:
                df = payload["data"].head(5).copy()
                with st.expander("View available parking areas"):
                    st.dataframe(
                        df[["lot_id","name","predicted_available","predicted_occupancy_pct","hourly_rate"]],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "lot_id":"Lot",
                            "name":"Car park",
                            "predicted_available":"Available",
                            "predicted_occupancy_pct":st.column_config.NumberColumn("Occupancy",format="%d%%"),
                            "hourly_rate":st.column_config.NumberColumn("Rate",format="$%.2f/h"),
                        }
                    )

            if payload.get("type") in ["recommendation", "comparison"] and payload.get("data") is not None:
                recs = payload["data"].head(5).copy()
                with st.expander("View recommendation details"):
                    st.dataframe(
                        recs[["lot_id","name","available","occupancy_pct","walk_min","hourly_rate"]],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "lot_id":"Lot",
                            "name":"Car park",
                            "available":"Available",
                            "occupancy_pct":st.column_config.NumberColumn("Occupancy",format="%d%%"),
                            "walk_min":st.column_config.NumberColumn("Walk",format="%.1f min"),
                            "hourly_rate":st.column_config.NumberColumn("Price",format="$%.2f/hr"),
                        }
                    )

elif page == "Model Insights":
    page_header("Model Insights", "See how ParkWise forecasts occupancy, ranks parking options and powers the Parking Assistant.")
    st.subheader("Model performance")
    st.write(
        "ParkWise uses a **Random Forest regressor** to estimate parking occupancy from "
        "parking-lot identity, capacity, time-of-day, day-of-week, month, weekend, rain, "
        "campus-event and exam-period features."
    )
    m1,m2,m3 = st.columns(3)
    m1.metric("MAE",f"{metrics['MAE']:.4f}")
    m2.metric("RMSE",f"{metrics['RMSE']:.4f}")
    m3.metric("R²",f"{metrics['R2']:.3f}")
    st.caption("Evaluation uses a time-ordered 80/20 holdout split.")
    st.markdown("""
    <div class="sp-card" style="margin-top:.8rem;margin-bottom:1rem">
      <div class="sp-kicker">Interpretation</div>
      <div class="sp-title">What the model is learning</div>
      <div class="sp-subtle">
        Time-of-day patterns dominate demand, while parking capacity, weekday effects,
        events and weather help refine occupancy estimates.
      </div>
    </div>
    """, unsafe_allow_html=True)

    
    try:
        prep = model.named_steps["prep"]
        rf = model.named_steps["model"]
        feature_names = prep.named_transformers_["cat"].get_feature_names_out(["lot_id"]).tolist()
        feature_names += [
            "capacity","hour","dow","month","is_weekend","hour_sin","hour_cos",
            "dow_sin","dow_cos","rain_mm","event_flag","exam_period"
        ]
        imp = pd.DataFrame({"feature":feature_names,"importance":rf.feature_importances_})
        imp = imp.sort_values("importance",ascending=False).head(10)
        imp["feature"] = imp["feature"].str.replace("lot_id_","Lot ",regex=False).str.replace("_"," ",regex=False).str.title()

        section_header("Top model drivers", "Relative feature importance from the Random Forest model")
        fig = px.bar(imp.sort_values("importance"),x="importance",y="feature",orientation="h",
                     labels={"importance":"Importance","feature":""})
        fig.update_traces(marker_color="#4B7B61")
        fig.update_layout(height=390,paper_bgcolor="#FFFFFF",plot_bgcolor="#FFFFFF",
                          margin=dict(l=10,r=10,t=10,b=10),showlegend=False)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#EEF0F2")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    except Exception:
        pass

    st.markdown("### How ParkWise works")
    st.markdown(
        "**Parking data** → **PostgreSQL** → **Feature engineering** → "
        "**Occupancy forecasting** → **Availability estimation** → "
        "**Recommendation ranking** → **Dashboard & Parking Assistant**"
    )

    st.markdown("### Recommendation engine")
    st.write(
        "Parking options are ranked using **predicted availability, walking distance and hourly parking cost**. "
        "The weighting changes according to the selected preference: **Best overall, Shortest walk, "
        "Lowest cost,** or **Most availability**."
    )

    st.markdown("### Parking Assistant")
    st.write(
        "The assistant converts natural-language requests into structured parking queries and calls ParkWise's "
        "forecasting and recommendation functions. Parking availability values come from the prediction pipeline "
        "rather than being generated directly by the language model."
    )

    st.markdown("### Data note")
    st.caption(
        "This demonstration uses synthetically generated university parking occupancy data designed to reproduce "
        "realistic campus demand patterns. The architecture can be connected to live parking sensors or API feeds."
    )

    st.markdown("""
    <div class="sp-card">
      <div class="sp-kicker">Model note</div>
      <div class="sp-title">Simulated-data evaluation</div>
      <div class="sp-subtle">These metrics are measured on the synthetic university parking dataset, not official university sensor data.</div>
    </div>
    """, unsafe_allow_html=True)

# UI enhancement note: feature importance shown in Model Insights is added below when branch is active.
