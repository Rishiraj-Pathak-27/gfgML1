# --- Enhanced Hospital Intelligence System for Hackathon ---
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# --- Data Loading & Feature Engineering ---
def load_enhanced_data():
    """Load and prepare ALL datasets with comprehensive features"""
    
    # 1. Load patients data
    patients = pd.read_csv('patients.csv')
    patients['date'] = pd.to_datetime(patients['arrival_date'])
    patients['departure_date'] = pd.to_datetime(patients['departure_date'])
    patients['stay_duration'] = (patients['departure_date'] - patients['date']).dt.days
    
    # 2. Load staff data to calculate staff requirements
    staff = pd.read_csv('staff.csv')
    staff_by_service = staff.groupby(['service', 'role']).size().reset_index(name='count')
    
    # Calculate staff ratios per service
    emergency_nurses = staff[(staff['service'] == 'emergency') & (staff['role'] == 'nurse')].shape[0]
    icu_nurses = staff[(staff['service'] == 'ICU') & (staff['role'] == 'nurse')].shape[0]
    emergency_doctors = staff[(staff['service'] == 'emergency') & (staff['role'] == 'doctor')].shape[0]
    
    print(f"Staff available: Emergency nurses={emergency_nurses}, ICU nurses={icu_nurses}, Emergency doctors={emergency_doctors}")
    
    # 3. Create daily aggregated features
    patients['day_of_week'] = patients['date'].dt.dayofweek
    patients['month'] = patients['date'].dt.month
    patients['day_of_year'] = patients['date'].dt.dayofyear
    patients['is_weekend'] = patients['day_of_week'].isin([5, 6]).astype(int)
    patients['is_monday'] = (patients['day_of_week'] == 0).astype(int)  # Monday surge
    patients['quarter'] = patients['date'].dt.quarter  # Seasonal patterns
    
    # Group by date
    daily_data = patients.groupby('date').agg({
        'patient_id': 'count',
        'age': ['mean', 'std'],
        'stay_duration': ['mean', 'max'],
        'day_of_week': 'first',
        'month': 'first',
        'day_of_year': 'first',
        'is_weekend': 'first',
        'is_monday': 'first',
        'quarter': 'first'
    }).reset_index()
    
    daily_data.columns = ['date', 'total_admissions', 'avg_age', 'age_std', 
                          'avg_stay', 'max_stay', 'day_of_week', 'month', 
                          'day_of_year', 'is_weekend', 'is_monday', 'quarter']
    
    # Count by service type
    emergency_count = patients[patients['service'] == 'emergency'].groupby('date').size().reset_index(name='emergency_count')
    icu_count = patients[patients['service'] == 'ICU'].groupby('date').size().reset_index(name='icu_count')
    surgery_count = patients[patients['service'] == 'surgery'].groupby('date').size().reset_index(name='surgery_count')
    general_count = patients[patients['service'] == 'general_medicine'].groupby('date').size().reset_index(name='general_count')
    
    # Merge all
    daily_data = pd.merge(daily_data, emergency_count, on='date', how='left')
    daily_data = pd.merge(daily_data, icu_count, on='date', how='left')
    daily_data = pd.merge(daily_data, surgery_count, on='date', how='left')
    daily_data = pd.merge(daily_data, general_count, on='date', how='left')
    daily_data = daily_data.fillna(0)
    
    # Sort by date
    daily_data = daily_data.sort_values('date').reset_index(drop=True)
    
    # 4. Create advanced lagged features (time series patterns)
    for lag in [1, 2, 3, 7]:  # Yesterday, 2 days ago, 3 days ago, last week
        daily_data[f'lag{lag}_admissions'] = daily_data['total_admissions'].shift(lag)
        daily_data[f'lag{lag}_emergency'] = daily_data['emergency_count'].shift(lag)
        daily_data[f'lag{lag}_icu'] = daily_data['icu_count'].shift(lag)
    
    # Rolling averages (trend detection)
    daily_data['rolling7_admissions'] = daily_data['total_admissions'].rolling(7, min_periods=1).mean()
    daily_data['rolling7_emergency'] = daily_data['emergency_count'].rolling(7, min_periods=1).mean()
    daily_data['rolling14_admissions'] = daily_data['total_admissions'].rolling(14, min_periods=1).mean()
    
    # Trend indicators (increasing or decreasing load)
    daily_data['trend_admissions'] = daily_data['total_admissions'] - daily_data['rolling7_admissions']
    daily_data['trend_emergency'] = daily_data['emergency_count'] - daily_data['rolling7_emergency']
    
    # 5. Environmental & Seasonal Features
    # Simulate seasonal outbreak risk (higher in winter months and flu season)
    daily_data['flu_season'] = daily_data['month'].isin([11, 12, 1, 2, 3]).astype(int)
    daily_data['summer_season'] = daily_data['month'].isin([6, 7, 8]).astype(int)
    
    # Cyclical encoding for month and day (better than linear)
    daily_data['month_sin'] = np.sin(2 * np.pi * daily_data['month'] / 12)
    daily_data['month_cos'] = np.cos(2 * np.pi * daily_data['month'] / 12)
    daily_data['day_sin'] = np.sin(2 * np.pi * daily_data['day_of_year'] / 365)
    daily_data['day_cos'] = np.cos(2 * np.pi * daily_data['day_of_year'] / 365)
    
    # 6. Calculate STAFF WORKLOAD (key requirement!)
    # Based on patient volume and service type
    daily_data['nurse_hours_needed'] = (
        daily_data['emergency_count'] * 4 +  # 4 hours per emergency patient
        daily_data['icu_count'] * 8 +         # 8 hours per ICU patient (intensive)
        daily_data['surgery_count'] * 6 +     # 6 hours per surgery
        daily_data['general_count'] * 2       # 2 hours per general patient
    )
    
    daily_data['doctor_hours_needed'] = (
        daily_data['emergency_count'] * 2 +   # 2 hours per emergency
        daily_data['icu_count'] * 3 +         # 3 hours per ICU
        daily_data['surgery_count'] * 4 +     # 4 hours per surgery
        daily_data['general_count'] * 1       # 1 hour per general
    )
    
    # Staff shortage risk (workload exceeds capacity)
    daily_data['nurse_shortage_risk'] = (daily_data['nurse_hours_needed'] > (emergency_nurses * 8)).astype(int)
    daily_data['bed_utilization'] = daily_data['total_admissions'] / 50  # Assume 50 bed capacity
    
    # Drop rows with NaN from lagging
    daily_data = daily_data.dropna()
    
    # Feature selection
    feature_cols = [
        # Lagged features
        'lag1_admissions', 'lag2_admissions', 'lag3_admissions', 'lag7_admissions',
        'lag1_emergency', 'lag2_emergency', 'lag3_emergency', 'lag7_emergency',
        'lag1_icu', 'lag2_icu', 'lag3_icu', 'lag7_icu',
        # Rolling trends
        'rolling7_admissions', 'rolling7_emergency', 'rolling14_admissions',
        'trend_admissions', 'trend_emergency',
        # Time features
        'day_of_week', 'month', 'is_weekend', 'is_monday', 'quarter',
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        # Seasonal/environmental
        'flu_season', 'summer_season',
        # Patient characteristics
        'avg_age', 'age_std', 'avg_stay', 'max_stay',
        # Current load
        'surgery_count', 'general_count', 'bed_utilization'
    ]
    
    X = daily_data[feature_cols].values
    
    # Target variables - COMPLETE SET for hackathon
    target_cols = [
        'emergency_count',      # Emergency admissions prediction
        'icu_count',           # ICU demand prediction
        'total_admissions',    # Overall load prediction
        'nurse_hours_needed',  # STAFF WORKLOAD prediction
        'doctor_hours_needed', # STAFF WORKLOAD prediction
        'nurse_shortage_risk'  # Alert system
    ]
    
    y = daily_data[target_cols].values
    
    return X, y, feature_cols, target_cols

def train_enhanced_model(X, y):
    """Train comprehensive multi-output model"""
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model with more trees for accuracy
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    )
    
    model.fit(X_scaled, y)
    
    # Save both model and scaler
    joblib.dump(model, "hospital_intelligence.joblib")
    joblib.dump(scaler, "hospital_scaler.joblib")
    
    return model, scaler

def generate_recommendations(predictions):
    """Generate actionable recommendations (optimization component)"""
    emergency, icu, total, nurse_hours, doctor_hours, shortage_risk = predictions
    
    recommendations = []
    
    # Emergency preparedness
    if emergency > 2:
        recommendations.append(f"⚠️ HIGH ALERT: {emergency:.0f} emergency admissions expected - Pre-position trauma team")
    
    # ICU management
    if icu > 2:
        recommendations.append(f"🏥 ICU ALERT: {icu:.0f} ICU cases expected - Ensure {int(icu*1.2)} beds available")
    
    # Staff allocation
    nurses_needed = int(nurse_hours / 8) + 1
    doctors_needed = int(doctor_hours / 8) + 1
    
    recommendations.append(f"👥 STAFFING: Schedule {nurses_needed} nurses and {doctors_needed} doctors")
    
    if shortage_risk > 0.5:
        recommendations.append("🚨 STAFF SHORTAGE RISK - Consider calling backup staff or limiting electives")
    
    # Bed management
    if total > 40:
        recommendations.append(f"🛏️ BED MANAGEMENT: {total:.0f} admissions - Prepare discharge plans for stable patients")
    
    return recommendations

# --- Main Execution ---
if __name__ == "__main__":
    print("="*60)
    print("🏥 PREDICTIVE HOSPITAL RESOURCE & EMERGENCY INTELLIGENCE SYSTEM")
    print("="*60)
    
    print("\n📊 Loading comprehensive hospital data...")
    X, y, feature_names, target_names = load_enhanced_data()
    
    print(f"✓ Data loaded: {X.shape[0]} days of history")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Predictions: {len(target_names)}")
    
    print("\n🤖 Training AI model...")
    model, scaler = train_enhanced_model(X, y)
    
    print("\n✅ MODEL TRAINED SUCCESSFULLY!")
    print("\n📋 Prediction Capabilities:")
    for i, target in enumerate(target_names):
        print(f"   {i+1}. {target}")
    
    print("\n💾 Models saved:")
    print("   - hospital_intelligence.joblib (main model)")
    print("   - hospital_scaler.joblib (feature scaler)")
    
    print("\n" + "="*60)
    print("✨ Ready for hackathon presentation!")
    print("="*60)
