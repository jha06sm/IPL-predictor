import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
with open("ipl_winner_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle['model']
label_encoders = bundle['label_encoders']

st.title("🏏 IPL Winning Team Predictor")

teams = label_encoders['team1'].classes_
venues = label_encoders['venue'].classes_
decisions = label_encoders['toss_decision'].classes_

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])
venue = st.selectbox("Select Venue", venues)
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", decisions)

if st.button("Predict Winner"):
    features = []
    for col in ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']:
        features.append(label_encoders[col].transform([locals()[col]])[0])
    
    pred = model.predict([features])[0]
    winner = label_encoders['winner'].inverse_transform([pred])[0]
    st.success(f"🏆 Predicted Winner: **{winner}**")
