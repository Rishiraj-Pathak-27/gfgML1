---
title: Hospital AI Forecasting
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# 🏥 Predictive Hospital Resource & Emergency Load Intelligence System

## Overview
AI-powered system that predicts hospital load, emergency admissions, ICU demand, and staff requirements 24 hours in advance.

## Features
- 📅 Date-based predictions for any future date
- 🚨 Emergency admission forecasting
- 🏥 ICU demand prediction
- 👥 Staff requirement calculation (nurses & doctors)
- 📊 Hospital capacity utilization
- ⚡ AI-generated actionable recommendations

## How It Works
The model analyzes:
- Historical admission patterns
- Day of week effects (Monday surge, weekend patterns)
- Seasonal factors (flu season, summer)
- Cyclical time patterns

And predicts:
- Patient load and capacity %
- Emergency and ICU cases
- Exact staffing needs
- Proactive operational recommendations

## Technology
- **Model**: Random Forest Multi-Output Regressor
- **Features**: 35 temporal and seasonal features
- **Training**: 337 days of hospital data
- **Predictions**: 6 simultaneous outputs

## Usage

### Running with Streamlit
```bash
streamlit run app.py
```

### Running the Flask API
```bash
python api.py
```

### Using the Application
1. Select a date (today or future)
2. Choose scenario type (auto-detect recommended)
3. View predictions and recommendations
4. Plan resources accordingly

## Installation
```bash
pip install -r requirements.txt
```

## Impact
- 24-hour advance warning for surges
- Prevents staff burnout through optimal scheduling
- Reduces emergency overtime costs
- Improves patient outcomes through preparedness

Built for 24hr Hackathon - January 2026
