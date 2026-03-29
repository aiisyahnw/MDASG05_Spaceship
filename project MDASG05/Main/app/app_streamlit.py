import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
pipeline = joblib.load(BASE_DIR.parent / "artifacts/model.pkl")
def main():
    st.set_page_config(page_title="ASG 04 MD - Aisyah NW - Spaceship Titanic Model Deployment")
    
    st.title("ASG 04 MD - Aisyah NW - Spaceship Titanic Model Deployment")
    st.write("enter data")

    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            home_planet = st.selectbox("Home Planet", ["Earth", "Europa", "Mars"])
            cryo_sleep = st.selectbox("CryoSleep", [True, False])
            destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
            age = st.slider("Age", 0, 100, 25)
            vip = st.selectbox("VIP Status", [False, True])
            
        with col2:
            room_service = st.number_input("Room Service Bill", 0.0, 10000.0, 0.0)
            food_court = st.number_input("Food Court Bill", 0.0, 10000.0, 0.0)
            shopping_mall = st.number_input("Shopping Mall Bill", 0.0, 10000.0, 0.0)
            spa = st.number_input("Spa Bill", 0.0, 10000.0, 0.0)
            vr_deck = st.number_input("VR Deck Bill", 0.0, 10000.0, 0.0)

    if st.button("Predict Result"):
        input_data = pd.DataFrame([{
            'HomePlanet': home_planet,
            'CryoSleep': cryo_sleep,
            'Destination': destination,
            'Age': age,
            'VIP': vip,
            'RoomService': room_service,
            'FoodCourt': food_court,
            'ShoppingMall': shopping_mall,
            'Spa': spa,
            'VRDeck': vr_deck
        }])

        prediction = pipeline.predict(input_data)
        probability = pipeline.predict_proba(input_data)[0][1]
        st.divider()
        if prediction[0] == 1:
            st.success(f"RESULT: TRANSPORTED!** (Probability: {probability:.2%})")
            st.balloons()
        else:
            st.warning(f"RESULT: NOT TRANSPORTED** (Probability: {probability:.2%})")

if __name__ == "__main__":
    main()