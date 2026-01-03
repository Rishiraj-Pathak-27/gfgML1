# 🏥 Hospital AI - Date-Based API
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from datetime import datetime
import math

app = Flask(__name__)
CORS(app)

# Load models
model = joblib.load("hospital_intelligence.joblib")
scaler = joblib.load("hospital_scaler.joblib")

@app.route('/api/predict', methods=['POST', 'GET'])
def predict():
    """Date-based predictions"""
    
    if request.method == 'POST':
        data = request.get_json()
        target_date = data.get('date')
        scenario = data.get('scenario', 'normal')
    else:
        target_date = request.args.get('date')
        scenario = request.args.get('scenario', 'normal')
    
    # Calculate features from date
    if target_date:
        try:
            date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            features = calculate_features_from_date(date_obj, scenario)
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    else:
        # Use predefined scenarios
        scenarios = {
            'normal': [3, 4, 3, 5, 1, 1, 2, 1, 1, 0, 1, 2, 3.5, 1.2, 3.8, -0.5, -0.2, 2, 4, 0, 0, 2, 0.0, 1.0, 0.5, 0.87, 0, 0, 45, 20, 5, 10, 1, 1, 0.6],
            'surge': [8, 7, 9, 6, 4, 3, 5, 3, 3, 2, 3, 2, 7.5, 3.5, 7.0, 1.5, 1.0, 0, 1, 0, 1, 1, 0.5, 0.87, -0.5, 0.87, 1, 0, 55, 25, 7, 14, 2, 3, 0.9],
            'weekend': [5, 6, 4, 8, 2, 3, 2, 2, 2, 1, 2, 1, 5.0, 2.0, 5.5, 0.0, 0.0, 6, 7, 1, 0, 3, -0.5, 0.87, 0.0, 1.0, 0, 1, 40, 18, 4, 8, 0, 2, 0.5]
        }
        features = scenarios.get(scenario, scenarios['normal'])
        date_obj = datetime.now()
    
    # Make prediction
    X = np.array([features])
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)[0]
    
    emergency, icu, total, nurse_hours, doctor_hours, shortage_risk = predictions
    
    # Calculate staff
    nurses_needed = int(nurse_hours / 8) + 1
    doctors_needed = int(doctor_hours / 8) + 1
    
    # Alert level
    alert_level = "normal"
    if emergency > 2 or icu > 2:
        alert_level = "high"
    elif emergency > 1.5 or icu > 1.5:
        alert_level = "medium"
    
    return jsonify({
        'success': True,
        'date': date_obj.strftime('%Y-%m-%d'),
        'day_name': date_obj.strftime('%A'),
        'predictions': {
            'emergency_admissions': round(emergency, 1),
            'icu_cases': round(icu, 1),
            'total_admissions': round(total, 1),
            'nurse_hours': round(nurse_hours, 0),
            'doctor_hours': round(doctor_hours, 0),
            'nurses_needed': nurses_needed,
            'doctors_needed': doctors_needed
        },
        'alert': {
            'level': alert_level,
            'shortage_risk': bool(shortage_risk > 0.5)
        },
        'recommendations': generate_recommendations(emergency, icu, total, nurses_needed, doctors_needed, shortage_risk, date_obj)
    })

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'online',
        'message': 'Hospital AI API - Date-Based Predictions',
        'version': '2.0'
    })

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
    if scenario == 'surge':
        lag1_adm, lag1_emg, lag1_icu = 8, 4, 3
        rolling7_adm, rolling7_emg = 7.5, 3.5
        trend_adm, trend_emg = 1.5, 1.0
        avg_age, avg_stay = 55, 7
        surgery, general, bed_util = 2, 3, 0.9
    elif scenario == 'weekend':
        lag1_adm, lag1_emg, lag1_icu = 5, 2, 2
        rolling7_adm, rolling7_emg = 5.0, 2.0
        trend_adm, trend_emg = 0.0, 0.0
        avg_age, avg_stay = 40, 4
        surgery, general, bed_util = 0, 2, 0.5
    else:  # normal
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
    
    # Build feature array (35 features)
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

def generate_recommendations(emergency, icu, total, nurses, doctors, shortage_risk, date_obj):
    """Generate date-aware recommendations"""
    recs = []
    
    day_name = date_obj.strftime('%A')
    is_weekend = date_obj.weekday() >= 5
    month = date_obj.month
    
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
    
    recs.append(f"👥 Staff Allocation: Schedule {nurses} nurses and {doctors} doctors")
    
    if is_weekend:
        recs.append("🏖️ Weekend: Ensure backup staff available")
    elif day_name == 'Monday':
        recs.append("⚡ Monday Surge: Expect higher volume")
    
    if month in [11, 12, 1, 2, 3]:
        recs.append("❄️ Flu Season: Stock respiratory supplies")
    
    if shortage_risk > 0.5:
        recs.append("⚠️ STAFF SHORTAGE RISK - Contact backup staff")
    
    if total > 40:
        recs.append("🛏️ High bed utilization - Plan discharges")
    
    return recs

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 HOSPITAL AI API v2.0 - DATE-BASED PREDICTIONS")
    print("="*60)
    print("\n📡 Endpoints:")
    print("   POST http://localhost:5000/api/predict")
    print("   GET  http://localhost:5000/api/status")
    print("\n📝 Example:")
    print('   {"date": "2026-01-15", "scenario": "normal"}')
    print("\n✨ Select any date and get predictions!")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
