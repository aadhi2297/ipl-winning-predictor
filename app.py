import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import requests
from collections import defaultdict

# --- App Config ---
st.set_page_config(page_title="IPL Win Predictor (Live)", page_icon="🏏", layout="wide")

# --- Branding ---
st.sidebar.image("ipl.png", use_column_width=True)

st.sidebar.markdown("### Developed by: Aadhi")
st.sidebar.markdown("Predict live IPL match win probabilities using ML & live cricket data.")


# --- Team Data ---
teams = {
    'Sunrisers Hyderabad': {'color': '#FF822A', 'logo': 'https://upload.wikimedia.org/wikipedia/en/8/81/Sunrisers_Hyderabad.png'},
    'Mumbai Indians': {'color': '#045093', 'logo': 'https://upload.wikimedia.org/wikipedia/en/c/cd/Mumbai_Indians_Logo.png'},
    'Royal Challengers Bangalore': {'color': '#DA1818', 'logo': 'https://upload.wikimedia.org/wikipedia/en/2/2f/Royal_Challengers_Bangalore_Logo.png'},
    'Kolkata Knight Riders': {'color': '#3B215D', 'logo': 'https://upload.wikimedia.org/wikipedia/en/4/4a/Kolkata_Knight_Riders_Logo.png'},
    'Kings XI Punjab': {'color': '#C8102E', 'logo': 'https://upload.wikimedia.org/wikipedia/en/d/d4/Punjab_Kings_Logo.png'},
    'Chennai Super Kings': {'color': '#F8CD05', 'logo': 'https://upload.wikimedia.org/wikipedia/en/2/2f/Chennai_Super_Kings_Logo.png'},
    'Rajasthan Royals': {'color': '#254AA5', 'logo': 'https://upload.wikimedia.org/wikipedia/en/6/60/Rajasthan_Royals_Logo.png'},
    'Delhi Capitals': {'color': '#17499D', 'logo': 'https://upload.wikimedia.org/wikipedia/en/3/3f/Delhi_Capitals_Logo.png'}
}

# Mapping for live feed team names
team_name_map = {team.lower(): team for team in teams.keys()}


cities = sorted([
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
])

# --- Load Model ---
pipe = pickle.load(open("pipe.pkl", "rb"))

# --- API Config ---
API_KEY = "9f5166a5-49c1-46cb-abfd-94b542e52215"
BASE_URL = "https://api.cricapi.com/v1"

# --- State ---
if "timeline" not in st.session_state:
    st.session_state.timeline = defaultdict(list)

# --- API Functions ---
def fetch_live_matches():
    try:
        url = f"{BASE_URL}/currentMatches?apikey={API_KEY}&offset=0"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        matches = []
        for m in r.json().get("data", []):
            if m.get("status") and "live" in m.get("status").lower():
                matches.append((m.get("id"), m.get("name", "Unknown")))
        return matches
    except:
        return []

def fetch_score(match_id: str):
    out = {"runs": None, "wkts": None, "overs": None, "target": None, "team": None}
    try:
        url = f"{BASE_URL}/match_score?apikey={API_KEY}&id={match_id}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", {})
        scores = data.get("score", [])
        if scores:
            latest = scores[-1]
            out["runs"] = latest.get("r")
            out["wkts"] = latest.get("w")
            out["overs"] = latest.get("o")
            out["team"] = latest.get("inning")
        if data.get("target"):
            try:
                out["target"] = int(data["target"])
            except:
                pass
        return out
    except:
        return out

# --- Main UI ---
st.markdown("<h1 style='text-align:center;color:#ff4b4b;'>🏏 IPL Win Predictor — Live Match Analysis</h1>", unsafe_allow_html=True)

# Sidebar Live Mode
live_mode = st.sidebar.toggle("Live Mode", value=False)
poll_secs = st.sidebar.slider("Auto-refresh every (sec)", 10, 120, 30, step=5)

# Get inputs
if live_mode:
    matches = fetch_live_matches()
    if matches:
        live_idx = st.selectbox("Select Live Match", range(len(matches)), format_func=lambda i: matches[i][1])
        match_id, match_title = matches[live_idx]
        sc = fetch_score(match_id)

        # Auto-map teams
        batting_team = team_name_map.get((sc.get("team") or "").lower(), list(teams.keys())[0])
        bowling_team = [t for t in teams.keys() if t != batting_team][0]

        selected_city = st.selectbox("Host City", cities)
        target = sc.get("target") or st.number_input("Target Score", min_value=1, step=1, value=150)
        score = sc.get("runs") or 0
        overs = sc.get("overs") or 0.0
        wickets = sc.get("wkts") or 0

        st_autorefresh = st.experimental_rerun if poll_secs else None
    else:
        st.warning("No live matches right now.")
        live_mode = False

if not live_mode:
    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox("Batting Team", sorted(teams.keys()))
    with col2:
        bowling_team = st.selectbox("Bowling Team", sorted(teams.keys()))
    selected_city = st.selectbox("Host City", cities)
    target = st.number_input("Target Score", min_value=1, step=1)
    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input("Score", min_value=0, step=1)
    with col4:
        overs = st.number_input("Overs", min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
    with col5:
        wickets = st.number_input("Wickets", min_value=0, max_value=10, step=1)

# Prediction
if overs == 0:
    st.warning("Enter overs > 0 to predict.")
elif score > target:
    st.error("Score > Target not possible.")
elif wickets >= 10:
    st.error("All out.")
else:
    runs_left = target - score
    balls_left = int(120 - (overs * 6))
    remaining_wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "city": [selected_city],
        "runs_left": [runs_left],
        "balls_left": [balls_left],
        "wickets": [remaining_wickets],
        "total_runs_x": [target],
        "crr": [crr],
        "rrr": [rrr]
    })

    result = pipe.predict_proba(input_df)
    win_prob = float(result[0][1])
    loss_prob = float(result[0][0])

    st.session_state.timeline["overs"].append(round(overs, 1))
    st.session_state.timeline["win_prob"].append(round(win_prob * 100, 1))

    # Match Situation Card
    st.markdown(f"""
    <div style="background:#f9f9f9;padding:10px;border-radius:10px;">
    <b>Batting:</b> {batting_team} | <b>Bowling:</b> {bowling_team} | <b>City:</b> {selected_city}<br>
    <b>Target:</b> {target} | <b>Score:</b> {score}/{wickets} in {overs} overs<br>
    <b>Runs Left:</b> {runs_left} | <b>Balls Left:</b> {balls_left}<br>
    <b>CRR:</b> {crr:.2f} | <b>RRR:</b> {rrr:.2f}
    </div>
    """, unsafe_allow_html=True)

    # Commentary
    commentary = []
    commentary.append("✅ RRR under control." if rrr <= crr else "⚠️ RRR above CRR.")
    commentary.append("💪 Wickets in hand." if remaining_wickets > 3 else "🛑 Low wickets left.")
    if runs_left <= 12 and balls_left <= 12:
        commentary.append("🔥 Endgame: every ball counts.")
    st.info(" ".join(commentary))

    # Probability Display
    colA, colB = st.columns(2)
    with colA:
        st.image(teams[batting_team]['logo'], width=100)
        st.markdown(f"<h3 style='color:{teams[batting_team]['color']};'>{batting_team}</h3>", unsafe_allow_html=True)
        st.progress(win_prob)
        st.success(f"{round(win_prob * 100, 1)}%")
    with colB:
        st.image(teams[bowling_team]['logo'], width=100)
        st.markdown(f"<h3 style='color:{teams[bowling_team]['color']};'>{bowling_team}</h3>", unsafe_allow_html=True)
        st.progress(loss_prob)
        st.error(f"{round(loss_prob * 100, 1)}%")

    # Timeline Chart
    st.subheader("📈 Win Probability Timeline")
    fig, ax = plt.subplots()
    ax.plot(st.session_state.timeline["overs"], st.session_state.timeline["win_prob"],
            marker="o", color=teams[batting_team]['color'])
    ax.set_xlabel("Overs")
    ax.set_ylabel("Win Probability (%)")
    ax.set_ylim(0, 100)
    ax.grid(True)
    st.pyplot(fig)

if st.button("🔄 Reset Timeline"):
    st.session_state.timeline.clear()
