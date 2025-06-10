import streamlit as st
import pandas as pd
import pickle

# Load the model
with open("ipl_winner_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle['model']
label_encoders = bundle['label_encoders']

# Set page config
st.set_page_config(page_title="IPL Winner Predictor", page_icon="🏏", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #3366cc;'>🏆 IPL Match Winner Predictor</h1>
    <p style='text-align: center; color: #444;'>Predict the outcome of your favorite IPL matches using historical data and machine learning!</p>
    <hr style="margin-top: 10px; margin-bottom: 30px;">
""", unsafe_allow_html=True)

# UI layout
col1, col2 = st.columns(2)

teams = label_encoders['team1'].classes_
venues = label_encoders['venue'].classes_
decisions = label_encoders['toss_decision'].classes_

with col1:
    team1 = st.selectbox("🅰️ Select Team 1", teams)
with col2:
    team2 = st.selectbox("🅱️ Select Team 2", [team for team in teams if team != team1])

venue = st.selectbox("📍 Select Venue", venues)
toss_winner = st.selectbox("🪙 Toss Winner", [team1, team2])
toss_decision = st.radio("🎯 Toss Decision", decisions, horizontal=True)

# Predict Button
if st.button("Predict Winner 🧠"):
    input_features = []
    for feature in ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']:
        encoder = label_encoders[feature]
        input_features.append(encoder.transform([eval(feature)])[0])

    prediction = model.predict([input_features])[0]
    predicted_team = label_encoders['winner'].inverse_transform([prediction])[0]

    st.success(f"🎉 Predicted Winner: **{predicted_team}**")
    st.balloons()
