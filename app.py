"""
app.py -- Event-Driven Congestion: Command Center
Run:  python -m streamlit run app.py
Needs: event_model.pkl, active_events.csv, all_events.csv, .streamlit/config.toml
Dataset-only. No external datasets used.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import pulp
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go

BUNDLE = joblib.load("event_model.pkl")
model = BUNDLE["model"]
CAT_FEATURES = BUNDLE["cat_features"]
NUM_FEATURES = BUNDLE["num_features"]
ACTIVE = pd.read_csv("active_events.csv")
ALL = pd.read_csv("all_events.csv")

st.set_page_config(page_title="Congestion Command", page_icon="🚦", layout="wide")

st.markdown("""
<style>
:root { --accent:#ff8c42; --hi:#ef4444; --med:#f59e0b; --lo:#22c55e; }
.stApp { background: radial-gradient(1200px 600px at 20% -10%, #16202e 0%, #0d1117 55%); }
#MainMenu, footer { visibility: hidden; }
.hero { background: linear-gradient(135deg,#2874f0,#1a4fa0); border:1px solid #1a4fa0;
  border-left:5px solid #ffe11b; border-radius:14px; padding:20px 26px; margin-bottom:8px; }
.hero h1 { margin:0; font-size:28px; font-weight:800; letter-spacing:-.5px; color:#ffffff; }
.hero p { margin:6px 0 0; color:#dce8ff; font-size:14px; }
.pill { display:inline-block; background:#ffffff; color:#2874f0; font-size:11px;
  padding:3px 10px; border-radius:999px; margin-right:6px; border:1px solid #ffe11b; font-weight:700; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:#161b22; border:1px solid #2a3441;
  border-radius:10px 10px 0 0; padding:10px 18px; font-weight:600; }
.stTabs [aria-selected="true"] { background:#1a3a6b; color:#ffffff;
  border-bottom:3px solid #ffe11b; }
[data-testid="stMetric"] { background:#161b22; border:1px solid #2a3441;
  border-radius:12px; padding:16px 18px; }
[data-testid="stMetricValue"] { font-size:28px; font-weight:800; }
[data-testid="stMetricLabel"] { color:#8b98a9; }
.stButton>button { background:linear-gradient(135deg,#ff8c42,#ff6a3d); color:#1a1208;
  border:none; border-radius:10px; font-weight:700; padding:10px 20px; }
.stButton>button:hover { filter:brightness(1.08); }
.stSelectbox label,.stSlider label,.stNumberInput label,.stCheckbox label {
  color:#aeb9c7 !important; font-weight:600; }
.sev { border-radius:12px; padding:14px 18px; font-weight:700; font-size:18px;
  margin:6px 0; border:1px solid; }
.sev.hi{background:#2a1414;color:#ff8585;border-color:#5b2020;}
.sev.med{background:#2a2110;color:#ffce6b;border-color:#5b4520;}
.sev.lo{background:#11261a;color:#6fe39a;border-color:#1f5237;}
.section { color:#9fb0c3; font-weight:700; font-size:13px; text-transform:uppercase;
  letter-spacing:1px; margin:10px 0 2px; }

/* Hero entrance + life */
@keyframes riseIn { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:translateY(0); } }
@keyframes glowPulse {
  0%,100% { box-shadow: 0 0 0 1px #1a4fa0, 0 0 22px -8px rgba(40,116,240,.5); }
  50%     { box-shadow: 0 0 0 1px #2a5fc0, 0 0 40px -6px rgba(40,116,240,.85); }
}
@keyframes sweep { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@keyframes blink { 0%,45% { opacity:1; } 50%,95% { opacity:.25; } 100% { opacity:1; } }

.hero { animation: riseIn .7s ease-out, glowPulse 4.5s ease-in-out infinite; }
.hero h1 {
  background: linear-gradient(90deg,#ffffff 35%,#ffe11b 50%,#ffffff 65%);
  background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; animation: sweep 6s linear infinite;
}
.hero p   { animation: riseIn .9s ease-out .15s both; }
.hero .pill { animation: riseIn .9s ease-out both; }
.hero .pill:nth-child(1){animation-delay:.25s;} .hero .pill:nth-child(2){animation-delay:.35s;}
.hero .pill:nth-child(3){animation-delay:.45s;} .hero .pill:nth-child(4){animation-delay:.55s;}
.live-dot { display:inline-block; width:9px; height:9px; border-radius:50%;
  background:#22c55e; margin-right:7px; animation: blink 1.6s infinite;
  box-shadow:0 0 8px #22c55e; }
.live-tag { color:#22c55e; font-size:12px; font-weight:700; letter-spacing:1px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Animated traffic-flow background — light streaks like vehicles on a highway */
.traffic-bg { position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none; }
.block-container { position: relative; z-index: 1; }   /* keep content above the animation */
.streak {
  position: absolute; height: 2px; border-radius: 3px; opacity: 0;
  filter: blur(1px); animation: drive linear infinite;
}
@keyframes drive {
  0%   { transform: translateX(-12vw); opacity: 0; }
  12%  { opacity: 0.55; }
  88%  { opacity: 0.55; }
  100% { transform: translateX(112vw); opacity: 0; }
}
</style>
<div class="traffic-bg">
  <div class="streak" style="top:9%;  width:70px;  background:linear-gradient(90deg,transparent,#ff8c42); animation-duration:9s;  animation-delay:0s;"></div>
  <div class="streak" style="top:16%; width:48px;  background:linear-gradient(90deg,transparent,#7fd1ff); animation-duration:13s; animation-delay:3s;"></div>
  <div class="streak" style="top:23%; width:95px;  background:linear-gradient(90deg,transparent,#ff8c42); animation-duration:7s;  animation-delay:1.5s;"></div>
  <div class="streak" style="top:31%; width:55px;  background:linear-gradient(90deg,transparent,#e6edf3); animation-duration:15s; animation-delay:6s;"></div>
  <div class="streak" style="top:39%; width:80px;  background:linear-gradient(90deg,transparent,#ef4444); animation-duration:10s; animation-delay:2s;"></div>
  <div class="streak" style="top:47%; width:40px;  background:linear-gradient(90deg,transparent,#7fd1ff); animation-duration:12s; animation-delay:8s;"></div>
  <div class="streak" style="top:55%; width:110px; background:linear-gradient(90deg,transparent,#ff8c42); animation-duration:8s;  animation-delay:4s;"></div>
  <div class="streak" style="top:63%; width:60px;  background:linear-gradient(90deg,transparent,#e6edf3); animation-duration:14s; animation-delay:1s;"></div>
  <div class="streak" style="top:71%; width:75px;  background:linear-gradient(90deg,transparent,#ef4444); animation-duration:11s; animation-delay:5.5s;"></div>
  <div class="streak" style="top:79%; width:50px;  background:linear-gradient(90deg,transparent,#7fd1ff); animation-duration:16s; animation-delay:2.5s;"></div>
  <div class="streak" style="top:87%; width:90px;  background:linear-gradient(90deg,transparent,#ff8c42); animation-duration:9.5s; animation-delay:7s;"></div>
  <div class="streak" style="top:93%; width:45px;  background:linear-gradient(90deg,transparent,#e6edf3); animation-duration:13s; animation-delay:3.5s;"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div style="margin-bottom:8px;"><span class="live-dot"></span><span class="live-tag">LIVE · BENGALURU TRAFFIC OPS</span></div>
  <h1>🚦 Bengaluru Event-Driven Congestion — Command Center</h1>
  <p>Live city risk · single-event forecasting · optimal officer deployment</p>
  <div style="margin-top:12px;">
    <span class="pill">Forecast</span><span class="pill">Optimize</span>
    <span class="pill">Learn</span><span class="pill">Dataset-only</span>
  </div>
</div>
""", unsafe_allow_html=True)

def officers_needed(severity, requires_closure, on_corridor):
    band = "High" if severity >= 0.66 else "Medium" if severity >= 0.33 else "Low"
    n = {"Low": 1, "Medium": 2, "High": 4}[band]
    if requires_closure: n += 2
    if on_corridor:      n += 1
    return n

def plan_for(severity, requires_closure, on_corridor):
    band = "High" if severity >= 0.66 else "Medium" if severity >= 0.33 else "Low"
    n = officers_needed(severity, requires_closure, on_corridor)
    barricade = bool(requires_closure or band == "High")
    return {"band": band, "officers": n,
            "barricade_units": (2 if band == "High" else 1) if barricade else 0,
            "response": {"Low": "Routine", "Medium": "Prompt", "High": "Immediate"}[band],
            "diversion": bool(requires_closure or band == "High")}

tab0, tab1, tab2, tab3 = st.tabs(["🌆 Live City Risk", "🎯 Single Event Planner & XAI",
                            "🗺️ City Command", "🧪 Urban Lab"])

# ---------------- TAB 0: Live City Risk (KPI strip + 3D time-scrubber + Plotly) -------
with tab0:
    st.markdown('<div class="section">Scrub through the day — watch where risk concentrates</div>',
                unsafe_allow_html=True)
    hr = st.slider("Hour of day (IST)", 0, 23, 18)
    sub = ALL[ALL["hour_ist"] == hr].copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active events", len(sub))
    k2.metric("Mean severity", f"{sub['severity'].mean():.2f}" if len(sub) else "—")
    k3.metric("High-severity events", int((sub["severity"] >= 0.66).sum()))
    real_zones = sub[sub["zone"] != "Unknown"]["zone"]
    busiest = real_zones.value_counts().idxmax() if len(real_zones) else "Sparse data"
    k4.metric("Busiest zone", busiest)
    k4.caption("Zone recorded for ~42% of events; busiest shown from available data.")
    
    
    

    col_map, col_chart = st.columns([2, 1])
    
    with col_map:
        st.markdown('<div class="section">3D Risk Map (Height & Color = Severity)</div>', unsafe_allow_html=True)
        if len(sub):
            try:
                import pydeck as pdk
                layer = pdk.Layer(
                    "HexagonLayer", data=sub,
                    get_position=["longitude", "latitude"],
                    radius=350, elevation_scale=60, extruded=True, pickable=True,
                    auto_highlight=True, coverage=0.9,
                    get_elevation_weight="severity", elevation_aggregation="SUM",
                    get_color_weight="severity", color_aggregation="MEAN",
                    color_range=[[34,197,94],[163,206,60],[245,200,66],
                                 [245,158,11],[239,120,68],[239,68,68]],
                )
                view = pdk.ViewState(latitude=float(sub["latitude"].mean()),
                                     longitude=float(sub["longitude"].mean()),
                                     zoom=10.3, pitch=50, bearing=10)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "Risk concentration here"}), height=400)
            except Exception as e:
                st.map(sub.rename(columns={"latitude": "lat", "longitude": "lon"}))
        else:
            st.info("No events recorded in this hour.")

    with col_chart:
        st.markdown('<div class="section">Severity by Event Cause (this hour)</div>', unsafe_allow_html=True)
        if len(sub):
            import plotly.express as px
            sev_by_cause = (sub.groupby("event_cause")["severity"]
                            .mean().sort_values(ascending=True).reset_index())
            sev_by_cause = sev_by_cause[sev_by_cause["event_cause"] != "Unknown"].tail(8)
            fig_bar = px.bar(sev_by_cause, x="severity", y="event_cause",
                             orientation="h", template="plotly_dark",
                             color="severity", color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                             range_color=[0, 1])
            fig_bar.update_layout(margin=dict(t=20, b=20, l=10, r=10),
                                  coloraxis_showscale=False,
                                  xaxis_title="Mean severity (0–1)", yaxis_title="",
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                  height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.write("Awaiting data...")

# ---------------- TAB 1: Single Event Planner & XAI --------------------------
with tab1:
    st.markdown('<div class="section">Scenario Generator</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        event_type = st.selectbox("Event type", ["unplanned", "planned"])
        event_cause = st.selectbox("Event cause", BUNDLE["event_causes"])
        corridor = st.selectbox("Corridor", BUNDLE["corridors"])
        priority = st.selectbox("Priority", ["High", "Low"])
        veh_type = st.selectbox("Vehicle type", BUNDLE["veh_types"])
    with c2:
        police_station = st.selectbox("Police station", BUNDLE["police_stations"])
        zone = st.selectbox("Zone", BUNDLE["zones"])
        requires_closure = st.checkbox("Requires road closure?", value=True)
        hour = st.slider("Hour of day (IST)", 0, 23, 18, key="t1hour")
        lat = st.number_input("Latitude", value=12.9716, format="%.4f")
        lon = st.number_input("Longitude", value=77.5946, format="%.4f")

    if st.button("⚡ Forecast Impact & Generate Dispatch Plan", type="primary"):
        on_corridor = int(corridor != "Non-corridor")
        is_rush = int(hour in [8, 9, 10, 18, 19, 20])
        
        row = {"event_type": event_type, "event_cause": event_cause, "corridor": corridor,
               "priority": priority, "veh_type": veh_type, "police_station": police_station,
               "zone": zone, "on_corridor": on_corridor,
               "is_planned": int(event_type == "planned"), "latitude": lat, "longitude": lon,
               "requires_road_closure": int(requires_closure), "hour_ist": hour,
               "day_of_week": 2, "is_weekend": 0, "month": 3,
               "sin_hour": np.sin(2 * np.pi * hour / 24), "cos_hour": np.cos(2 * np.pi * hour / 24),
               "is_rush_hour": is_rush}
        
        X = pd.DataFrame([row])[CAT_FEATURES + NUM_FEATURES]
        for c in CAT_FEATURES:
            X[c] = X[c].astype("category")
        sev = float(np.clip(model.predict(X)[0], 0, 1))
        p = plan_for(sev, requires_closure, on_corridor)
        cls = {"High": "hi", "Medium": "med", "Low": "lo"}[p["band"]]
        
        st.markdown(f'<div class="sev {cls}">{p["band"]} impact &nbsp;·&nbsp; '
                    f'Severity {sev:.2f} &nbsp;·&nbsp; {p["response"]} response required</div>',
                    unsafe_allow_html=True)
        
        # New Section: XAI and Prescriptive Engine
        xai_col, pre_col = st.columns([1, 1])
        
        with xai_col:
            st.markdown('<div class="section">🧠 Explainable AI — real model attribution (Tree-SHAP)</div>',
                        unsafe_allow_html=True)
            # REAL feature contributions from the model itself (LightGBM Tree-SHAP).
            # These sum exactly to the predicted value — not illustrative.
            contrib = model.predict(X, pred_contrib=True)[0]
            feat_names = CAT_FEATURES + NUM_FEATURES
            base_val = contrib[-1]
            pairs = sorted(zip(feat_names, contrib[:-1]), key=lambda t: -abs(t[1]))
            top = pairs[:4]
            other = sum(v for _, v in pairs[4:])

            labels = ["Base"] + [f.replace("_", " ") for f, _ in top] + ["Other", "Final"]
            measures = ["absolute"] + ["relative"] * (len(top) + 1) + ["total"]
            yvals = [base_val] + [v for _, v in top] + [other, sev]
            texts = [f"{base_val:.2f}"] + [f"{v:+.2f}" for _, v in top] + [f"{other:+.2f}", f"{sev:.2f}"]

            fig_wf = go.Figure(go.Waterfall(
                orientation="v", measure=measures, x=labels, y=yvals,
                text=texts, textposition="outside",
                connector={"line": {"color": "#4b5563"}},
                decreasing={"marker": {"color": "#22c55e"}},
                increasing={"marker": {"color": "#ef4444"}},
                totals={"marker": {"color": "#ff8c42"}}))
            fig_wf.update_layout(template="plotly_dark", height=280,
                                 margin=dict(t=20, b=20, l=10, r=10),
                                 plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_wf, use_container_width=True)
            st.caption("Each bar is the feature's actual SHAP contribution to this prediction.")

        with pre_col:
            st.markdown('<div class="section">🚨 Prescriptive Action Plan</div>', unsafe_allow_html=True)
            if requires_closure or p["band"] == "High":
                st.error(f"**CRITICAL ACTION:** {event_cause.replace('_', ' ').title()} actively blocking {corridor if on_corridor else 'local grid'}.")
                st.write(f"1. **Deploy {p['officers']} officers** from {police_station} immediately.")
                st.write(f"2. **Activate upstream diversion:** Place {p['barricade_units']} barricade units 200m ahead of coordinates.")
                st.write("3. **Digital Signage:** Update VMS boards to reroute traffic to nearest parallel arterial.")
            elif p["band"] == "Medium":
                st.warning(f"**WARNING:** Congestion building due to {event_cause.replace('_', ' ')}.")
                st.write(f"1. **Deploy {p['officers']} officers** to manage flow.")
                st.write("2. Monitor intersection spillback via nearest CCTV.")
            else:
                st.success("**NOMINAL IMPACT:** Event is within normal handling tolerances.")
                st.write(f"1. Dispatch {p['officers']} officer for standard clearance.")
                
        st.markdown('<div class="section">🗺️ Live Context: Active Events Heatmap Radius</div>', unsafe_allow_html=True)
        m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB dark_matter")
        folium.Marker([lat, lon], popup=f"Simulated: {event_cause}", icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
        if len(ACTIVE) > 0:
            heat_data = ACTIVE[['latitude', 'longitude']].dropna().values.tolist()
            HeatMap(heat_data, radius=15, blur=20, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
        st_folium(m, width=1200, height=350, returned_objects=[])

# ---------------- TAB 2: City Command ----------------------------------------
with tab2:
    st.markdown('<div class="section">City-wide optimal deployment engine</div>', unsafe_allow_html=True)
    st.caption("Advanced Logistics: Simulating real-world resource boundaries, jurisdictional constraints, and extreme weather events.")
    
    st.markdown("### 🌧️ Macro-Environmental Control Sandbox")
    monsoon_mode = st.toggle("🚨 Simulate Sudden Extreme Monsoon Downpour (Flash Flood Multiplier)", value=False)
    
    scen = ACTIVE.copy()
    if monsoon_mode:
        st.error("⛈️ Monsoon Emergency Protocol Active: Rain-sensitive incidents have had their risk profiles heavily amplified.")
        scen.loc[scen['event_cause'].isin(['water_logging', 'pot_holes']), 'severity'] *= 1.45
        scen.loc[scen['event_cause'] == 'accident', 'severity'] *= 1.25
        scen['severity'] = scen['severity'].clip(0, 1.0)
        scen.loc[scen['severity'] >= 0.66, 'officers_needed'] += 1

    colA, colB = st.columns(2)
    n_active = colA.slider("Events active right now", 4, len(scen), 12)
    scen = scen.head(n_active).reset_index(drop=True)
    
    st.markdown("### 🗂️ Jurisdictional Resource Allocation")
    
    c_pool1, c_pool2, c_pool3 = st.columns(3)
    pool_east = c_pool1.number_input("East/South Zones Pool", min_value=2, max_value=30, value=8)
    pool_west = c_pool2.number_input("West/North Zones Pool", min_value=2, max_value=30, value=8)
    pool_central = c_pool3.number_input("Central/Other Zones Pool", min_value=2, max_value=30, value=6)
    
    east_zones = ['J.P. Nagar', 'Madiwala', 'Hulimavu', 'Electronic City', 'Adugodi', 'HAL Old Airport']
    west_zones = ['Kamakshipalya', 'Kengeri', 'Yeshwanthpura', 'Byatarayanapura', 'Magadi Road', 'Kodigehalli', 'Hebbala', 'R.T. Nagar']
    scen['zone_pool_group'] = 'Central'
    scen.loc[scen['police_station'].isin(east_zones), 'zone_pool_group'] = 'East'
    scen.loc[scen['police_station'].isin(west_zones), 'zone_pool_group'] = 'West'
    
    need = scen["officers_needed"].values
    sev = scen["severity"].values
    groups = scen["zone_pool_group"].values
    total_officers_available = pool_east + pool_west + pool_central

    if st.button("🧠 Execute Zonal Optimization Solver", type="primary"):
        # Greedy
        order = np.argsort(-(sev / need))
        g_pick, g_used = [], {'East': 0, 'West': 0, 'Central': 0}
        limits = {'East': pool_east, 'West': pool_west, 'Central': pool_central}
        for i in order:
            g = groups[i]
            if g_used[g] + need[i] <= limits[g]:
                g_pick.append(i)
                g_used[g] += need[i]
        g_cov = sev[g_pick].sum() if g_pick else 0.0
        g_total_used = sum(g_used.values())

        # Optimizer
        prob = pulp.LpProblem("zonal_deploy", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(need))]
        prob += pulp.lpSum(sev[i] * x[i] for i in range(len(need)))
        prob += pulp.lpSum(need[i] * x[i] for i in range(len(need)) if groups[i] == 'East') <= pool_east
        prob += pulp.lpSum(need[i] * x[i] for i in range(len(need)) if groups[i] == 'West') <= pool_west
        prob += pulp.lpSum(need[i] * x[i] for i in range(len(need)) if groups[i] == 'Central') <= pool_central
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        o_pick = {i for i in range(len(need)) if x[i].value() == 1}
        o_cov = sev[list(o_pick)].sum() if o_pick else 0.0
        o_total_used = int(sum(need[i] for i in o_pick))
        gain = o_cov - g_cov
        
        col_metrics, col_gauge = st.columns([2, 1])
        
        with col_metrics:
            m1, m2, m3 = st.columns(3)
            m1.metric("Greedy (Risk Covered)", f"{g_cov:.2f}", f"{g_total_used}/{total_officers_available} units used")
            m2.metric("Optimal (Risk Covered)", f"{o_cov:.2f}", f"{o_total_used}/{total_officers_available} units used")
            m3.metric("Optimizer Edge", f"+{gain:.2f}", f"{(gain / g_cov * 100 if g_cov else 0):.1f}% better protection")
            
        with col_gauge:
            # NEW: Professional Plotly Gauge Chart for Resource Stress
            util_pct = (o_total_used / total_officers_available) * 100
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = util_pct, number={"suffix": "%"},
                title = {'text': "Fleet Utilization Stress", 'font': {'size': 14, 'color': '#8b98a9'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#ff8c42"},
                    'steps': [{'range': [0, 60], 'color': "#161b22"}, {'range': [60, 85], 'color': "#2a2110"}, {'range': [85, 100], 'color': "#2a1414"}],
                    'threshold': {'line': {'color': "#ef4444", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }))
            fig_gauge.update_layout(height=180, margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark",
                                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)

        disp = scen.copy()
        disp["staffed"] = ["Staffed" if i in o_pick else "Unstaffed" for i in range(len(scen))]
        disp["color_r"] = disp["staffed"].map({"Staffed": 34, "Unstaffed": 239})
        disp["color_g"] = disp["staffed"].map({"Staffed": 197, "Unstaffed": 68})
        disp["color_b"] = disp["staffed"].map({"Staffed": 94, "Unstaffed": 68})
        
        st.markdown('<div class="section">🟢 Green = Deployed Units | 🔴 Red = Unstaffed Risks</div>', unsafe_allow_html=True)
        try:
            import pydeck as pdk
            layer = pdk.Layer("ScatterplotLayer", data=disp, get_position=["longitude", "latitude"],
                              get_fill_color=["color_r", "color_g", "color_b", 200], get_radius="officers_needed * 140", pickable=True,
                              stroked=True, get_line_color=[255, 255, 255, 60])
            view = pdk.ViewState(latitude=float(disp["latitude"].mean()), longitude=float(disp["longitude"].mean()), zoom=10.6)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{event_cause} ({zone_pool_group} Pool)\nStation: {police_station}\nStatus: {staffed}"}))
        except Exception:
            st.map(disp.rename(columns={"latitude": "lat", "longitude": "lon"}))
            
        out = scen[["event_cause", "police_station", "zone_pool_group", "severity", "officers_needed"]].copy()
        out["decision"] = ["✅ Dispatch Unit" if i in o_pick else "⚠️ Resource Starved" for i in range(len(scen))]
        st.dataframe(out, use_container_width=True, hide_index=True)
        
        # Operational workflow download
        st.markdown("---")
        csv_export = out.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Official Dispatch Manifest (CSV)", data=csv_export,
                           file_name="bengaluru_dispatch_manifest.csv", mime="text/csv", type="secondary")
    else:
        st.map(scen.rename(columns={"latitude": "lat", "longitude": "lon"}))

        # ---------------- TAB 3: Urban Lab (What-if explorer, simplified) ------------
with tab3:
    st.markdown('<div class="section">Urban Lab — simplified what-if explorer</div>',
                unsafe_allow_html=True)
    st.info("A first-order what-if tool: it scales current event severity by the "
            "capacity gains you set. It is an illustrative planning aid, not a "
            "calibrated traffic-flow simulation.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Policy inputs")
        transit_boost = st.slider("Public transit capacity boost (%)", 0, 50, 0)
        lane_capacity = st.slider("Road lane efficiency gain (%)", 0, 30, 0)

    reduction_factor = 1 - ((transit_boost + lane_capacity) / 100)

    with col2:
        st.markdown("### Projected first-order impact")
        st.metric("Modelled congestion reduction", f"{((1 - reduction_factor) * 100):.1f}%")
        by_cause = ACTIVE.groupby("event_cause")["severity"].mean().sort_values()
        fig_impact = go.Figure()
        fig_impact.add_trace(go.Bar(x=by_cause.index, y=by_cause.values, name="Current",
                                    marker_color="#ef4444"))
        fig_impact.add_trace(go.Bar(x=by_cause.index, y=by_cause.values * reduction_factor,
                                    name="After policy", marker_color="#22c55e"))
        fig_impact.update_layout(template="plotly_dark", height=300, barmode="group",
                                 title="Mean severity per category — before vs after",
                                 plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_impact, use_container_width=True)

    if ((1 - reduction_factor) * 100) > 20:
        st.success(f"High-impact policy: this simplified model projects a "
                   f"{((1 - reduction_factor) * 100):.1f}% reduction in current event severity.")
    else:
        st.info("Adjust the sliders to explore how capacity investments scale down event severity.")
