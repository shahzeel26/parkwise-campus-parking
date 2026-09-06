import streamlit as st

def inject_css():
    st.markdown("""
    <style>
    .stApp {background:#F7F8FA;color:#1F2937;}
    .block-container {max-width:1320px;padding:2.6rem 2rem 2.5rem 2rem;}
    h1,h2,h3 {color:#17202A;letter-spacing:-0.02em;}
    h1 {font-size:2.05rem!important;font-weight:700!important;line-height:1.25!important;margin-top:.15rem!important;}
    h2 {font-size:1.4rem!important;font-weight:650!important;}
    h3 {font-size:1.05rem!important;font-weight:650!important;}

    [data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #E6E9ED;}
    [data-testid="stMetric"] {background:#FFFFFF;border:1px solid #E4E7EB;border-radius:12px;padding:.9rem 1rem;}
    [data-testid="stMetricLabel"] {color:#6B7280;font-size:.82rem!important;}
    [data-testid="stMetricValue"] {color:#111827;font-size:1.55rem!important;font-weight:700;}

    .stSelectbox div[data-baseweb="select"] > div,
    .stDateInput input,.stTimeInput input,.stNumberInput input,.stTextInput input {
        background:#FFFFFF!important;border:1px solid #DDE1E6!important;border-radius:10px!important;
    }

    [data-testid="stDataFrame"] {border:1px solid #E3E6EA;border-radius:12px;overflow:hidden;background:#FFFFFF;}
    [data-testid="stChatMessage"] {background:#FFFFFF;border:1px solid #E4E7EB;border-radius:12px;padding:.65rem .8rem;}

    .sp-card {background:#FFFFFF;border:1px solid #E4E7EB;border-radius:14px;padding:1rem 1.05rem;}
    .sp-recommendation {background:#F4F8F5;border:1px solid #D9E6DE;border-left:4px solid #2F6B4F;border-radius:12px;padding:1rem 1.05rem;}
    .sp-kicker {color:#2F6B4F;font-weight:650;font-size:.78rem;text-transform:uppercase;letter-spacing:.03em;}
    .sp-title {color:#17202A;font-weight:700;font-size:1.1rem;margin:.2rem 0 .35rem 0;}
    .sp-subtle {color:#6B7280;font-size:.9rem;line-height:1.45;}
    .sp-status {display:inline-block;background:#ECF5EF;color:#2F6B4F;border:1px solid #D5E7DC;border-radius:999px;padding:.28rem .6rem;font-size:.8rem;font-weight:650;}
    .sp-legend {display:flex;gap:1rem;flex-wrap:wrap;align-items:center;margin:.25rem 0 .7rem 0;color:#6B7280;font-size:.84rem;}
    .sp-dot {display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:.32rem;vertical-align:middle;}
    .sp-green {background:#3F895A;}
    .sp-amber {background:#CA9932;}
    .sp-orange {background:#C9702C;}
    .sp-red {background:#BE4343;}

    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    .modebar {opacity:.25;}
    </style>
    """, unsafe_allow_html=True)

def page_header(title, subtitle):
    st.markdown(
        f"<div style='margin-bottom:1rem'><div style='font-size:2rem;font-weight:720;color:#17202A'>{title}</div>"
        f"<div style='color:#6B7280;font-size:.94rem;margin-top:.15rem'>{subtitle}</div></div>",
        unsafe_allow_html=True
    )

def section_header(title, subtitle=None):
    st.markdown(f"<div class='sp-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='sp-subtle' style='margin-bottom:.55rem'>{subtitle}</div>", unsafe_allow_html=True)
