import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load trained model and encoder
model = joblib.load("t20_model.joblib")
le_team = joblib.load("team_encoder.joblib")

# Page settings
st.set_page_config(page_title="T20 Final Score Predictor", page_icon="🏏")

st.title("🏏 T20 Final Score Predictor")
st.write("Enter the current match details below to predict the final team score:")

# Get team list from encoder
teams = list(le_team.classes_)

# Input fields
batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
current_over = st.number_input("Current Over (0 - 19)", min_value=0.0, max_value=19.5, step=0.1)
current_score = st.number_input("Current Score", min_value=0, max_value=300)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)
runs_last_5 = st.number_input("Runs Scored in Last 5 Overs", min_value=0, max_value=100)

# Predict button
if st.button("Predict Final Score"):
    try:
        input_df = pd.DataFrame({
            'BattingTeam': [le_team.transform([batting_team])[0]],
            'BowlingTeam': [le_team.transform([bowling_team])[0]],
            'CurrentOver': [current_over],
            'CurrentScore': [current_score],
            'Wickets': [wickets],
            'RunsLast5': [runs_last_5],
        })

        prediction = model.predict(input_df)[0]
        st.success(f"🏆 Predicted Final Score: {prediction:.0f} runs")
    except Exception as e:
        st.error(f"Prediction failed ❌: {e}")
