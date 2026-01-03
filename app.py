import streamlit as st
import joblib
import numpy as np
from datetime import datetime, timedelta
import math

# Page config
st.set_page_config(
    page_title="Hospital Intelligence System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("hospital_intelligence.joblib")
    scaler = joblib.load("hospital_scaler.joblib")
    return model, scaler

model, scaler = load_models()

def calculate_features_from_date(date_obj, scenario='normal'):
    """Calculate all 35 features based on target date"""
    
    # Time features
    day_of_week = date_obj.weekday()
    month = date_obj.month
    day_of_year = date_obj.timetuple().tm_yday
    is_weekend = 1 if day_of_week >= 5 else 0
    is_monday = 1 if day_of_week == 0 else 0
    quarter = (month - 1) // 3 + 1
    
    # Cyclical encoding
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    day_sin = math.sin(2 * math.pi * day_of_year / 365)
    day_cos = math.cos(2 * math.pi * day_of_year / 365)
    
    # Seasonal factors
    flu_season = 1 if month in [11, 12, 1, 2, 3] else 0
    summer_season = 1 if month in [6, 7, 8] else 0
    
    # Base patterns by scenario
    if scenario == 'High Load (Surge)':
        lag1_adm, lag1_emg, lag1_icu = 8, 4, 3
        rolling7_adm, rolling7_emg = 7.5, 3.5
        trend_adm, trend_emg = 1.5, 1.0
        avg_age, avg_stay = 55, 7
        surgery, general, bed_util = 2, 3, 0.9
    elif scenario == 'Weekend Pattern':
        lag1_adm, lag1_emg, lag1_icu = 5, 2, 2
        rolling7_adm, rolling7_emg = 5.0, 2.0
        trend_adm, trend_emg = 0.0, 0.0
        avg_age, avg_stay = 40, 4
        surgery, general, bed_util = 0, 2, 0.5
    else:  # Auto-Detect
        lag1_adm, lag1_emg, lag1_icu = 3, 1, 1
        rolling7_adm, rolling7_emg = 3.5, 1.2
        trend_adm, trend_emg = -0.5, -0.2
        avg_age, avg_stay = 45, 5
        surgery, general, bed_util = 1, 1, 0.6
    
    # Adjust for day patterns
    if is_weekend:
        lag1_adm *= 1.2
        lag1_emg *= 1.3
    if is_monday:
        lag1_adm *= 1.4
        lag1_emg *= 1.5
    
    # Adjust for season
    if flu_season:
        lag1_emg *= 1.5
        lag1_icu *= 1.4
        rolling7_emg *= 1.3
    
    # Generate lags
    lag2_adm = lag1_adm * 0.95
    lag3_adm = lag1_adm * 0.98
    lag7_adm = lag1_adm * 1.05
    lag2_emg = lag1_emg * 0.9
    lag3_emg = lag1_emg * 1.1
    lag7_emg = lag1_emg * 0.95
    lag2_icu = lag1_icu * 0.85
    lag3_icu = lag1_icu * 1.0
    lag7_icu = lag1_icu * 1.1
    
    # Build feature array
    features = [
        lag1_adm, lag2_adm, lag3_adm, lag7_adm,
        lag1_emg, lag2_emg, lag3_emg, lag7_emg,
        lag1_icu, lag2_icu, lag3_icu, lag7_icu,
        rolling7_adm, rolling7_emg, rolling7_adm * 1.05,
        trend_adm, trend_emg,
        day_of_week, month, is_weekend, is_monday, quarter,
        month_sin, month_cos, day_sin, day_cos,
        flu_season, summer_season,
        avg_age, 20,
        avg_stay, avg_stay + 5,
        surgery, general, bed_util
    ]
    
    return features

def predict_hospital_load(date_obj, scenario):
    """Main prediction function"""
    
    try:
        # Calculate features
        features = calculate_features_from_date(date_obj, scenario)
        
        # Make prediction
        X = np.array([features])
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)[0]
        
        emergency, icu, total, nurse_hours, doctor_hours, shortage_risk = predictions
        
        # Calculate staff
        nurses_needed = int(nurse_hours / 8) + 1
        doctors_needed = int(doctor_hours / 8) + 1
        
        # Calculate load
        capacity = 50
        load_percentage = (total / capacity * 100)
        
        if load_percentage > 80:
            load_status = '🔴 Critical - Near Capacity'
        elif load_percentage > 60:
            load_status = '🟡 High - Busy'
        elif load_percentage > 40:
            load_status = '🔵 Moderate'
        else:
            load_status = '🟢 Low - Comfortable'
        
        # Generate recommendations
        day_name = date_obj.strftime('%A')
        month = date_obj.month
        
        recs = []
        recs.append(f"📅 Forecast for: {date_obj.strftime('%B %d, %Y')} ({day_name})")
        
        if emergency > 2:
            recs.append(f"🚨 HIGH ALERT: {emergency:.0f} emergency cases expected")
            recs.append("→ Pre-position trauma team and ensure supplies ready")
        else:
            recs.append(f"✅ Normal emergency load: {emergency:.0f} cases expected")
        
        if icu > 2:
            recs.append(f"🏥 ICU ALERT: Prepare {int(icu*1.5)} ICU beds")
        else:
            recs.append(f"✅ Normal ICU load: {icu:.0f} cases")
        
        recs.append(f"👥 Staff Allocation: Schedule {nurses_needed} nurses and {doctors_needed} doctors")
        
        if date_obj.weekday() >= 5:
            recs.append("🏖️ Weekend: Ensure backup staff available")
        elif day_name == 'Monday':
            recs.append("⚡ Monday Surge: Expect higher volume")
        
        if month in [11, 12, 1, 2, 3]:
            recs.append("❄️ Flu Season: Stock respiratory supplies")
        
        if shortage_risk > 0.5:
            recs.append("⚠️ STAFF SHORTAGE RISK - Contact backup staff")
        
        return {
            'total': total,
            'load_percentage': load_percentage,
            'load_status': load_status,
            'emergency': emergency,
            'icu': icu,
            'nurses_needed': nurses_needed,
            'doctors_needed': doctors_needed,
            'recommendations': recs
        }
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Streamlit UI
st.title("🏥 Predictive Hospital Resource & Emergency Load Intelligence System")
st.markdown("### AI-Powered 24-Hour Emergency & Resource Forecasting")
st.markdown("This system predicts hospital load, emergency admissions, ICU demand, and staff requirements for any selected date.")

st.divider()

# Sidebar inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    date_input = st.date_input(
        "📅 Select Prediction Date",
        value=datetime.now(),
        min_value=datetime.now() - timedelta(days=30),
        max_value=datetime.now() + timedelta(days=365)
    )
    
    scenario_input = st.selectbox(
        "Scenario Type",
        ["Auto-Detect Load Pattern", "High Load (Surge)", "Weekend Pattern"],
        index=0
    )
    
    predict_btn = st.button("🔮 Predict Hospital Load", type="primary", use_container_width=True)
    
    st.divider()
    st.markdown("""
    ### 🎯 How It Works
    The AI model analyzes:
    - Historical admission patterns
    - Day of week effects
    - Seasonal factors
    - Cyclical time patterns
    
    And predicts:
    - Patient load & capacity
    - Emergency & ICU demand
    - Staff requirements
    - Actionable recommendations
    """)

# Convert date to datetime
date_obj = datetime.combine(date_input, datetime.min.time())

# Make prediction on load or button click
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    results = predict_hospital_load(date_obj, scenario_input)
else:
    if predict_btn:
        results = predict_hospital_load(date_obj, scenario_input)
    else:
        results = predict_hospital_load(date_obj, scenario_input)

if results:
    # Display main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🏥 Total Admissions",
            value=f"{results['total']:.1f} patients"
        )
    
    with col2:
        st.metric(
            label="🚨 Emergency Cases",
            value=f"{results['emergency']:.0f}"
        )
    
    with col3:
        st.metric(
            label="🏥 ICU Cases",
            value=f"{results['icu']:.0f}"
        )
    
    # Hospital load status
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Hospital Capacity Status")
        st.progress(results['load_percentage'] / 100)
        st.markdown(f"### {results['load_percentage']:.1f}% Capacity - {results['load_status']}")
    
    with col2:
        st.subheader("👥 Staff Requirements")
        st.markdown(f"**Nurses:** {results['nurses_needed']}")
        st.markdown(f"**Doctors:** {results['doctors_needed']}")
    
    # Recommendations
    st.divider()
    st.subheader("⚡ AI-Generated Recommendations")
    for rec in results['recommendations']:
        st.markdown(f"- {rec}")

# Footer
st.divider()
st.markdown("**Built for 24hr Hackathon - January 2026** | Powered by AI")

