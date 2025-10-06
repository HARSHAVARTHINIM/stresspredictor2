# app.py - Student Mental Health Predictor (Streamlit-native UI)

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Student Mental Health Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

feature_names = [
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying",
]
class_mapping = {0: "Low", 1: "Moderate", 2: "High"}

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("balanced_stress_model.pkl")
        return model
    except FileNotFoundError as e:
        st.error(f"❌ Error loading model file: {e}")
        st.error("Please place 'balanced_stress_model.pkl' next to this script.")
        st.stop()

def stress_chip(label: str) -> str:
    emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(label, "⚪")
    return f"{emoji} **{label} Stress**"

def create_stress_gauge(prediction_value: int, prediction_label: str, confidence: float):
    # Map confidence and class to gauge needle position
    percent = float(confidence)
    if prediction_label == "Low":
        # Needle moves from 0 to 33 based on confidence
        needle = percent * 33.0
        gauge_title = f"Low Stress Confidence: {percent*100:.1f}%"
    elif prediction_label == "Moderate":
        # Needle moves from 33 to 66 based on confidence
        needle = 33.0 + percent * 33.0
        gauge_title = f"Moderate Stress Confidence: {percent*100:.1f}%"
    else:  # High
        # Needle moves from 66 to 100 based on confidence
        needle = 66.0 + percent * 34.0
        gauge_title = f"High Stress Confidence: {percent*100:.1f}%"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=needle,
            title={"text": gauge_title},
            number={"font": {"color": "#111827"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["Left", "25%", "Center", "75%", "Right"],
                },
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 33], "color": "#d1fae5"},   # green
                    {"range": [33, 66], "color": "#fef3c7"},  # yellow
                    {"range": [66, 100], "color": "#fee2e2"}, # red
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 4},
                    "thickness": 0.75,
                    "value": needle,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(height=300, margin=dict(l=16, r=16, t=40, b=16))
    return fig

def create_risk_radar(user_inputs: dict):
    anxiety = user_inputs.get("anxiety_level", 5) / 10
    depression = user_inputs.get("depression", 5) / 10
    self_esteem_inv = (15 - user_inputs.get("self_esteem", 8)) / 15
    sleep_inv = (6 - user_inputs.get("sleep_quality", 3)) / 5
    social_inv = (6 - user_inputs.get("social_support", 3)) / 5
    activities_inv = (6 - user_inputs.get("extracurricular_activities", 3)) / 5
    acad_inv = (6 - user_inputs.get("academic_performance", 3)) / 5
    study = user_inputs.get("study_load", 3) / 5
    future = user_inputs.get("future_career_concerns", 3) / 5
    mental = np.clip(np.mean([anxiety, depression, self_esteem_inv]), 0, 1)
    lifestyle = np.clip(np.mean([sleep_inv, social_inv, activities_inv]), 0, 1)
    academic = np.clip(np.mean([acad_inv, study, future]), 0, 1)
    categories = ["Mental", "Lifestyle", "Academic"]
    values = [mental * 10, lifestyle * 10, academic * 10]
    values.append(values[0])
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(37,99,235,0.13)",  # valid rgba string
            line=dict(color="#2563eb", width=2),
            marker=dict(color="#2563eb"),
            name="Risk (higher = more risk)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10], color="#6b7280")),
        showlegend=False,
        height=320,
        margin=dict(l=16, r=16, t=16, b=16),
    )
    return fig

def generate_personalized_advice(prediction_level, user_inputs):
    advice = ""
    if prediction_level == 0:
        advice = "🌟 **Excellent! You're managing well!**\n\n"
        advice += "- Continue your current routines that are working\n"
        advice += "- Consider mentoring or helping peers who might be struggling\n"
        advice += "- Use your positive energy to pursue new goals or projects\n\n"
        advice += "**Maintenance tips:**\n"
        advice += "- Practice gratitude journaling\n"
        advice += "- Stay connected with your support network"
    elif prediction_level == 1:
        advice = "💡 **Time for some adjustments**\n\n"
        advice += "**Priority recommendations based on your inputs:**\n"
        if user_inputs.get("sleep_quality", 3) <= 2:
            advice += "- 🛌 Sleep: Aim for 7–8 hours. Establish a wind-down routine without screens.\n"
            if user_inputs.get("self_esteem", 8) <= 5:
                advice += "- 💪 Self-Esteem: Practice positive self-talk. Note three wins daily.\n"
        if user_inputs.get("social_support", 3) <= 2:
            advice += "- 👥 Social Support: Reach out to a friend. Join a club or activity.\n"
        if user_inputs.get("study_load", 3) >= 4:
            advice += "- 📚 Study Load: Break tasks into chunks. Try Pomodoro (25m work, 5m break).\n"
        advice += "\n**Quick stress relievers:**\n"
        advice += "- 5 deep breaths when overwhelmed\n"
        advice += "- 10-minute walk outside\n"
        advice += "- Calming music for 15 minutes"
    else:
        advice = "🤗 **Prioritize your well-being**\n\n"
        advice += "**Immediate actions to consider:**\n"
        advice += "- 🏥 Contact counseling services — asking for help is strength\n"
        advice += "- 📞 Talk to someone you trust — you’re not alone\n"
        advice += "- 🚨 If in crisis, call emergency services or a crisis hotline\n\n"
        advice += "**Focus areas based on your inputs:**\n"
        if user_inputs.get("anxiety_level", 5) >= 7:
            advice += "- 😰 Anxiety: Try box breathing (4s in, 4s hold, 4s out)\n"
        if user_inputs.get("depression", 10) >= 15:
            advice += "- 😔 Mood: Small daily wins matter. Celebrate them.\n"
        if user_inputs.get("sleep_quality", 3) <= 2:
            advice += "- 🛌 Sleep: Even modest improvements can help significantly\n"
        advice += "\n**Remember:** This is temporary, and help is available."
    return advice

def main():
    model = load_assets()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    st.title("🎓 Student Mental Health Predictor")
    st.markdown("Get instant insight into stress levels with personalized, actionable guidance.")
    tab_assess, tab_risk, tab_history, tab_resources = st.tabs(["Assessment", "Risk Drivers", "History", "Resources"])
    with tab_assess:
        st.header("Your Information")
        with st.form("assessment_form"):
            st.subheader("Mental Well-being")
            anxiety = st.slider("Anxiety Level", 0, 10, 5, help="0 = Low, 10 = High — How anxious do you feel most days?")
            self_esteem = st.slider("Self-Esteem", 0, 15, 8, help="0 = Low, 15 = High — How confident and positive do you feel?")
            depression = st.slider("Depression Score", 0, 10, 5, help="0 = Low, 10 = High — Frequency of feeling down or hopeless.")
            st.subheader("Lifestyle Factors")
            sleep = st.slider("Sleep Quality", 1, 5, 3, help="1 = Poor, 5 = Excellent — Rate your overall sleep quality.")
            social = st.slider("Social Support", 1, 5, 3, help="1 = Low, 5 = High — How supported do you feel?")
            activities = st.slider("Extracurricular Activities", 1, 5, 3, help="1 = None, 5 = Very Active — Non-academic activities.")
            st.subheader("Academic Factors")
            academics = st.slider("Academic Performance", 1, 5, 3, help="1 = Low, 5 = High — Your current performance.")
            study = st.slider("Study Load", 1, 5, 3, help="1 = Light, 5 = Heavy — Your current workload.")
            future = st.slider("Future Career Concerns", 1, 5, 3, help="1 = Low, 5 = High — Concerns regarding future career.")
            submitted = st.form_submit_button("🔍 Analyze My Stress Level")

        if submitted:
            input_data = {feature: 3 for feature in feature_names}
            input_data.update({
                "anxiety_level": anxiety,
                "self_esteem": self_esteem,
                "depression": depression,
                "sleep_quality": sleep,
                "social_support": social,
                "extracurricular_activities": activities,
                "academic_performance": academics,
                "study_load": study,
                "future_career_concerns": future,
            })
            input_df = pd.DataFrame([input_data], columns=feature_names)
            pred = int(model.predict(input_df)[0])
            label = class_mapping[pred]
            confidence = 0.0
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                confidence = float(proba[pred])
            st.session_state.last_result = {
                "timestamp": datetime.now(),
                "prediction": pred,
                "label": label,
                "confidence": confidence,
                "inputs": input_data,
            }
            st.session_state.history.append(st.session_state.last_result)
        if st.session_state.last_result:
            lr = st.session_state.last_result
            st.subheader(f"Assessment Summary: {stress_chip(lr['label'])}")
            fig = create_stress_gauge(lr["prediction"], lr["label"], lr["confidence"])
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Prediction", lr["label"])
            st.metric("Confidence", f"{lr['confidence']:.1%}")
            st.progress(min(1.0, max(0.0, lr["confidence"])))
            st.markdown("---")
            st.subheader("Personalized Advice")
            st.markdown(generate_personalized_advice(lr["prediction"], lr["inputs"]))
            result_df = pd.DataFrame(
                [
                    {
                        "timestamp": lr["timestamp"],
                        "prediction": lr["prediction"],
                        "label": lr["label"],
                        "confidence": lr["confidence"],
                    }
                ]
            )
            st.download_button(
                "⬇️ Download this result (CSV)",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="stress_assessment_result.csv",
                mime="text/csv",
            )
        else:
            st.info("Complete the form above to generate your personalized assessment.")

    with tab_risk:
        st.header("Risk Drivers Visualization")
        if st.session_state.last_result:
            lr = st.session_state.last_result
            st.plotly_chart(create_risk_radar(lr["inputs"]), use_container_width=True)
        else:
            st.info("Run an assessment to see your risk drivers visualization.")
    with tab_history:
        st.header("Your History")
        if st.session_state.history:
            hist_df = pd.DataFrame(
                [
                    {
                        "timestamp": h["timestamp"],
                        "prediction": h["prediction"],
                        "label": h["label"],
                        "confidence": h["confidence"],
                    }
                    for h in st.session_state.history
                ]
            ).sort_values("timestamp")
            col_a, col_b = st.columns([1.4, 1], gap="large")
            with col_a:
                fig_timeline = go.Figure()
                fig_timeline.add_trace(
                    go.Scatter(
                        x=hist_df["timestamp"],
                        y=hist_df["prediction"],
                        mode="lines+markers",
                        line=dict(color="#2563eb", width=2),
                        marker=dict(size=6),
                        name="Stress class (0=Low, 1=Moderate, 2=High)",
                    )
                )
                fig_timeline.update_yaxes(tickvals=[0, 1, 2], ticktext=["Low", "Moderate", "High"])
                fig_timeline.update_layout(
                    height=320, margin=dict(l=16, r=16, t=16, b=16), showlegend=False
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            with col_b:
                st.metric("Total Assessments", len(hist_df))
                if len(hist_df) >= 2:
                    delta = hist_df["prediction"].iloc[-1] - hist_df["prediction"].iloc[-2]
                    st.metric("Latest Change", f"{delta:+d} class")
                avg_conf = hist_df["confidence"].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            st.download_button(
                "⬇️ Download full history (CSV)",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="stress_assessment_history.csv",
                mime="text/csv",
            )
            if st.button("🧹 Reset history"):
                st.session_state.history = []
                st.session_state.last_result = None
                st.success("History cleared.")
        else:
            st.info("No history yet. Run at least one assessment to see your timeline.")
    with tab_resources:
        st.header("Helpful Resources")
        st.markdown(
            "- University Counseling Services: [Your University Link](https://example.com)\n"
            "- Crisis Text Line: Text HOME to 741741\n"
            "- National Suicide & Crisis Lifeline (US): 988 or 1-800-273-8255\n"
        )
        st.subheader("Disclaimer")
        st.markdown(
            "This tool is for educational and self-awareness purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. "
            "If you're experiencing a mental health emergency, contact local emergency services or a crisis hotline."
        )
    # Sidebar removed as requested

if __name__ == "__main__":
    main()
