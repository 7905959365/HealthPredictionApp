# -*- coding: utf-8 -*-

# Core Libraries for Streamlit and Machine Learning
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import time

# New Imports for Neural Network
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --- Database Setup ---
import sqlite3

DB_NAME = 'health_predictions.db'

def init_db():
    """Sets up the database and creates the table for diabetes predictions if it doesn't exist."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS diabetes_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, -- Added user_id column
                pregnancies REAL,
                glucose REAL,
                blood_pressure REAL,
                skin_thickness REAL,
                insulin REAL,
                bmi REAL,
                diabetes_pedigree_function REAL,
                age REAL,
                prediction_result TEXT,
                prediction_probability REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS heart_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, -- Added user_id column
                age REAL,
                sex REAL,
                cp REAL,
                trestbps REAL,
                chol REAL,
                fbs REAL,
                restecg REAL,
                thalach REAL,
                exang REAL,
                oldpeak REAL,
                slope REAL,
                ca REAL,
                thal REAL,
                prediction_result TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS eco_health_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, -- Added user_id column
                city TEXT,
                aqi REAL,
                green_space_percent REAL,
                avg_age REAL,
                avg_bmi REAL,
                risk_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Something went wrong while setting up the database: {e}")
    finally:
        if conn:
            conn.close()

def save_diabetes_prediction(user_id, pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, diabetes_pedigree_function, age,
                             prediction_result, prediction_probability):
    """Saves the entered patient data and the prediction into the database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''
            INSERT INTO diabetes_predictions (
                user_id, pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree_function, age,
                prediction_result, prediction_probability
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, pregnancies, glucose, blood_pressure, skin_thickness,
              insulin, bmi, diabetes_pedigree_function, age,
              prediction_result, prediction_probability))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Oops! Couldn't save the diabetes prediction: {e}")
        return False
    finally:
        if conn:
            conn.close()

def fetch_diabetes_predictions_by_user(user_id):
    """Gets all the saved diabetes predictions from the database for a specific user_id."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('SELECT * FROM diabetes_predictions WHERE user_id = ? ORDER BY timestamp DESC', (user_id,))
        data = c.fetchall()
        return data
    except sqlite3.Error as e:
        st.error(f"Couldn't get saved diabetes predictions for user {user_id}: {e}")
        return []
    finally:
        if conn:
            conn.close()

def save_heart_prediction(user_id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                          oldpeak, slope, ca, thal, prediction_result):
    """Saves the entered patient data and the heart disease prediction into the database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''
            INSERT INTO heart_predictions (
                user_id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                oldpeak, slope, ca, thal, prediction_result
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
              oldpeak, slope, ca, thal, prediction_result))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Oops! Couldn't save the heart disease prediction: {e}")
        return False
    finally:
        if conn:
            conn.close()

def fetch_heart_predictions_by_user(user_id):
    """Gets all the saved heart disease predictions from the database for a specific user_id."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('SELECT * FROM heart_predictions WHERE user_id = ? ORDER BY timestamp DESC', (user_id,))
        data = c.fetchall()
        return data
    except sqlite3.Error as e:
        st.error(f"Couldn't get saved heart disease predictions for user {user_id}: {e}")
        return []
    finally:
        if conn:
            conn.close()

def save_eco_health_prediction(user_id, city, aqi, green_space_percent, avg_age, avg_bmi, risk_score):
    """Saves eco-health prediction data into the database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''
            INSERT INTO eco_health_predictions (
                user_id, city, aqi, green_space_percent, avg_age, avg_bmi, risk_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, city, aqi, green_space_percent, avg_age, avg_bmi, risk_score))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Oops! Couldn't save the eco-health prediction: {e}")
        return False
    finally:
        if conn:
            conn.close()

def fetch_eco_health_predictions_by_user(user_id):
    """Gets all the saved eco-health predictions from the database for a specific user_id."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('SELECT * FROM eco_health_predictions WHERE user_id = ? ORDER BY timestamp DESC', (user_id,))
        data = c.fetchall()
        return data
    except sqlite3.Error as e:
        st.error(f"Couldn't get saved eco-health predictions for user {user_id}: {e}")
        return []
    finally:
        if conn:
            conn.close()


# --- IMPORTANT: Call this function once when your app starts ---
init_db()

# --- Page Configuration ---
st.set_page_config(page_title="Eco-Health & Urban Planning AI",
                    layout="wide",
                    page_icon="🧑‍⚕️")

# --- 1. Paths to your saved models ---
diabetes_nn_model_path = 'diabetes_neural_network_model.h5'
diabetes_scaler_path = 'diabetes_scaler.pkl'
heartdisease_model_path = '1trained_model.sav'

# --- 2. Loading the saved models ---
diabetes_model = None
diabetes_scaler = None
heartdisease_model = None

# Load Neural Network for Diabetes
try:
    diabetes_model = load_model(diabetes_nn_model_path)
    with open(diabetes_scaler_path, 'rb') as f:
        diabetes_scaler = pickle.load(f)
    st.sidebar.success("Diabetes Neural Network model loaded successfully!")
except FileNotFoundError:
    st.sidebar.error(f"Error: Diabetes Neural Network model or scaler file not found. Ensure '{diabetes_nn_model_path}' and '{diabetes_scaler_path}' are in the correct directory.")
except Exception as e:
    st.sidebar.error(f"Error loading Diabetes Neural Network model: {e}")

# Load Heart Disease Model
try:
    heartdisease_model = pickle.load(open(heartdisease_model_path, 'rb'))
    st.sidebar.success("Heart Disease prediction model loaded successfully!")
except FileNotFoundError:
    st.sidebar.error(f"Error: Heart Disease model file not found at '{heartdisease_model_path}'. Please ensure the file is in the correct directory.")
except Exception as e:
    st.sidebar.error(f"Error loading Heart Disease model: {e}")


# --- Simulated AI Model for Eco-Health Risk Prediction (Enhanced Logic) ---
def predict_eco_health_risk(aqi, green_space_percent, avg_age, avg_bmi):
    """
    Calculates an eco-health risk score (0.0 to 1.0) based on input factors.
    This version uses a weighted sum, making it more dynamic than simple if/else chains.
    """
    normalized_aqi_risk = aqi / 500.0
    normalized_green_space_risk = 1.0 - (green_space_percent / 100.0)
    normalized_age_risk = max(0.0, (avg_age - 20) / 60.0)
    normalized_bmi_risk = max(0.0, (avg_bmi - 18) / 17.0)

    weight_aqi = 0.35
    weight_green_space = 0.30
    weight_age = 0.20
    weight_bmi = 0.15

    combined_risk = (normalized_aqi_risk * weight_aqi) + \
                    (normalized_green_space_risk * weight_green_space) + \
                    (normalized_age_risk * weight_age) + \
                    (normalized_bmi_risk * weight_bmi)

    eco_health_risk = min(1.0, max(0.0, combined_risk))
    return eco_health_risk


# --- 3. Sidebar for navigation and User ID input ---
with st.sidebar:
    st.title("User Identification")
    user_id = st.text_input("Enter Your Unique ID", key="user_id_input", help="This ID will be used to save and retrieve your predictions.")
    if not user_id:
        st.warning("Please enter a User ID to proceed with predictions and view your history.")

    st.markdown("---") # Separator

    selected = option_menu(
        'Multiple Disease & Eco-Health Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Eco-Health Risk Predictor'],
        icons=['activity', 'heart', 'globe'],
        default_index=0
    )

# --- 4. Main content area for each prediction page ---

# --- Diabetes Prediction Page ---
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using an Advanced Neural Network')
    st.markdown("---")
    st.write("Enter the following details to predict diabetes risk:")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, help="Number of times pregnant")
    with col2:
        glucose = st.number_input('Glucose Level (mg/dL)', min_value=0.0, value=100.0, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    with col3:
        blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=0.0, value=70.0, help="Diastolic blood pressure (mm Hg)")

    with col1:
        skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0.0, value=20.0, help="Triceps skin fold thickness (mm)")
    with col2:
        insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0.0, value=100.0, help="2-Hour serum insulin (mu U/ml)")
    with col3:
        bmi = st.number_input('BMI (kg/m²)', min_value=0.0, value=25.0, help="Body Mass Index (weight in kg / (height in m)^2)")

    with col1:
        diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.5, format="%.3f", help="A function that scores likelihood of diabetes based on family history")
    with col2:
        age = st.number_input('Age of the Person', min_value=0, max_value=120, value=30, help="Age in years")

    if st.button('Get Diabetes Test Result'):
        if not user_id:
            st.error("Please enter your Unique ID in the sidebar before making a prediction.")
        else:
            with st.spinner('Predicting diabetes risk using Neural Network...'):
                time.sleep(1)
                try:
                    input_data = [
                        pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, diabetes_pedigree_function, age
                    ]

                    input_data_as_numpy_array = np.asarray(input_data)
                    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                    if diabetes_model and diabetes_scaler:
                        input_data_scaled = diabetes_scaler.transform(input_data_reshaped)
                        diab_prediction_probability = diabetes_model.predict(input_data_scaled)[0][0]

                        if diab_prediction_probability >= 0.5:
                            diab_diagnosis_text = 'Diabetic'
                            display_message = 'The person is **diabetic**.'
                        else:
                            diab_diagnosis_text = 'Not Diabetic'
                            display_message = 'The person is **NOT diabetic**.'

                        st.success(display_message)
                        st.info(f"Confidence (Probability of being Diabetic): {diab_prediction_probability:.2f}")

                        if save_diabetes_prediction(user_id, pregnancies, glucose, blood_pressure, skin_thickness,
                                                     insulin, bmi, diabetes_pedigree_function, age,
                                                     diab_diagnosis_text, diab_prediction_probability):
                            st.success("🎉 Diabetes prediction saved successfully to your project's database!")
                        else:
                            st.error("❗ Failed to save diabetes prediction. Please check for errors.")
                    else:
                        st.warning("Diabetes prediction model or scaler not loaded. Cannot make prediction. Please check the sidebar for error messages and ensure 'diabetes_neural_network_model.h5' and 'diabetes_scaler.pkl' are in the same directory.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}. Please ensure all inputs are valid numbers.")

    st.subheader(f"📊 Your Diabetes Predictions (User ID: {user_id})")
    if user_id:
        saved_diabetes_data = fetch_diabetes_predictions_by_user(user_id)

        if saved_diabetes_data:
            df_saved_diabetes = pd.DataFrame(saved_diabetes_data,
                                             columns=['ID', 'User ID', 'Pregnancies', 'Glucose', 'Blood Pressure',
                                                      'Skin Thickness', 'Insulin', 'BMI',
                                                      'Diabetes Pedigree Function', 'Age',
                                                      'Prediction Result', 'Prediction Probability', 'Timestamp'])
            st.dataframe(df_saved_diabetes)

            # --- Download Button for Diabetes Predictions ---
            csv_diab = df_saved_diabetes.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Your Diabetes Predictions as CSV",
                data=csv_diab,
                file_name=f'{user_id}_diabetes_predictions.csv',
                mime='text/csv',
                key='download_diabetes_csv'
            )
        else:
            st.info(f"No diabetes predictions saved yet for User ID: {user_id}. Make a prediction to see it appear here!")
    else:
        st.info("Enter your User ID in the sidebar to view your past diabetes predictions.")


# --- Heart Disease Prediction Page ---
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using Machine Learning')
    st.markdown("---")
    st.write("Enter the following details to predict heart disease risk:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=50, help="Age in years")
    with col2:
        sex = st.number_input('Sex (0 = female, 1 = male)', min_value=0, max_value=1, value=0, help="0 for female, 1 for male")
        if sex not in [0, 1]:
            st.warning("Please enter 0 for female or 1 for male for Sex.")
    with col3:
        cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3, value=1, help="Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)")
        if cp not in [0, 1, 2, 3]:
            st.warning("Please enter a value between 0 and 3 for Chest Pain Type.")
    with col1:
        trestbps = st.number_input('Resting Blood Pressure (mmHg)', min_value=0.0, value=120.0, help="Resting blood pressure (in mm Hg on admission to the hospital)")

    with col2:
        chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=0.0, value=200.0, help="Serum cholestoral in mg/dl")
    with col3:
        fbs = st.number_input('Fasting Blood Sugar (> 120 mg/dl) (0=No, 1=Yes)', min_value=0, max_value=1, value=0, help="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    with col1:
        restecg = st.number_input('Resting Electrocardiographic Results (0,1,2)', min_value=0, max_value=2, value=1, help="Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: definite left ventricular hypertrophy)")
    with col2:
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0.0, value=150.0, help="Maximum heart rate achieved")
    with col3:
        exang = st.number_input('Exercise Induced Angina (0 = no, 1 = yes)', min_value=0, max_value=1, value=0, help="Exercise induced angina (1 = yes; 0 = no)")

    with col1:
        oldpeak = st.number_input('Oldpeak', min_value=0.0, value=1.0, format="%.2f", help="ST depression induced by exercise relative to rest")
    with col2:
        slope = st.number_input('Slope of the Peak Exercise ST Segment (0,1,2)', min_value=0, max_value=2, value=1, help="The slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)")
    with col3:
        ca = st.number_input('Number of Major Vessels (0-3)', min_value=0, max_value=3, value=0, help="Number of major vessels (0-3) colored by flourosopy")
    with col1:
        thal = st.number_input('Thal (0 = normal; 1 = fixed defect; 2 = reversable defect)', min_value=0, max_value=2, value=2, help="Thal (0: normal; 1: fixed defect; 2: reversable defect)")

    if st.button('Get Heart Disease Test Result'):
        if not user_id:
            st.error("Please enter your Unique ID in the sidebar before making a prediction.")
        else:
            with st.spinner('Predicting heart disease risk...'):
                time.sleep(1)
                try:
                    heart_input_data = [
                        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                        oldpeak, slope, ca, thal
                    ]

                    heart_input_data_as_numpy_array = np.asarray(heart_input_data)
                    heart_input_data_reshaped = heart_input_data_as_numpy_array.reshape(1, -1)

                    if heartdisease_model:
                        heart_prediction = heartdisease_model.predict(heart_input_data_reshaped)

                        if heart_prediction[0] == 1:
                            heart_diagnosis_text = 'Has Heart Disease'
                            display_message = 'The person has **heart disease**.'
                        else:
                            heart_diagnosis_text = 'Does NOT have Heart Disease'
                            display_message = 'The person does **NOT have heart disease**.'

                        st.success(display_message)

                        if save_heart_prediction(user_id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                                                 oldpeak, slope, ca, thal, heart_diagnosis_text):
                            st.success("🎉 Heart disease prediction saved successfully to your project's database!")
                        else:
                            st.error("❗ Failed to save heart disease prediction. Please check for errors.")

                    else:
                        st.warning("Heart Disease model not loaded. Cannot make prediction. Please check the sidebar for error messages.")

                except Exception as e:
                    st.error(f"An error occurred during heart disease prediction: {e}. Please ensure all inputs are valid numbers.")

    st.subheader(f"📊 Your Heart Disease Predictions (User ID: {user_id})")
    if user_id:
        saved_heart_data = fetch_heart_predictions_by_user(user_id)

        if saved_heart_data:
            df_saved_heart = pd.DataFrame(saved_heart_data,
                                          columns=['ID', 'User ID', 'Age', 'Sex', 'Chest Pain Type', 'Resting BP',
                                                   'Cholesterol', 'Fasting BS', 'Resting ECG', 'Max HR',
                                                   'Exercise Angina', 'Oldpeak', 'Slope', 'CA', 'Thal',
                                                   'Prediction Result', 'Timestamp'])
            st.dataframe(df_saved_heart)

            # --- Download Button for Heart Disease Predictions ---
            csv_heart = df_saved_heart.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Your Heart Disease Predictions as CSV",
                data=csv_heart,
                file_name=f'{user_id}_heart_disease_predictions.csv',
                mime='text/csv',
                key='download_heart_csv'
            )
        else:
            st.info(f"No heart disease predictions saved yet for User ID: {user_id}. Make a prediction to see it appear here!")
    else:
        st.info("Enter your User ID in the sidebar to view your past heart disease predictions.")


# --- Eco-Health Risk Predictor Page ---
if selected == 'Eco-Health Risk Predictor':
    st.title('Eco-Health Risk Predictor & Sustainable Urban Planning Advisor')
    st.markdown("---")
    st.write("This tool assesses community health risk based on environmental factors and provides urban planning recommendations, aligning with our vision for 'Building Healthy Cities'.")

    st.subheader("1. Enter Community/Environmental Data")

    cities = {
        "Delhi": {"lat": 28.7041, "lon": 77.1025},
        "Mumbai": {"lat": 19.0760, "lon": 72.8777},
        "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
        "Chennai": {"lat": 13.0827, "lon": 80.2707},
        "Kolkata": {"lat": 22.5726, "lon": 88.3639}
    }
    selected_city_name = st.selectbox("Select a City/Region", list(cities.keys()), help="Choose a city to simulate environmental conditions.")
    selected_city_coords = cities[selected_city_name]

    st.info("*(In a full implementation, environmental data would be integrated from real-time IoT sensors and external APIs for selected regions.)*")

    col_env1, col_env2 = st.columns(2)
    with col_env1:
        aqi = st.slider('Average Air Quality Index (AQI) in the area', min_value=0, max_value=500, value=150, help="Higher AQI means worse air quality (e.g., 0-50 Good, 101-150 Unhealthy for Sensitive Groups, 301-500 Hazardous).")
    with col_env2:
        green_space_percent = st.slider('Percentage of Green Space in the area (%)', min_value=0, max_value=100, value=20, help="Higher percentage means more green spaces (e.g., parks, trees, gardens).")

    st.subheader("2. Enter General Health Indicators (for contextual risk assessment)")
    col_health_eco1, col_health_eco2 = st.columns(2)
    with col_health_eco1:
        avg_age = st.slider('Average Age of Community Members', min_value=18, max_value=90, value=40, help="Average age helps in assessing general population vulnerability.")
    with col_health_eco2:
        avg_bmi = st.slider('Average BMI of Community Members', min_value=15.0, max_value=40.0, value=25.0, format="%.1f", help="Average BMI gives an indication of obesity rates in the community.")

    if st.button('Assess Eco-Health Risk & Get Recommendations'):
        if not user_id:
            st.error("Please enter your Unique ID in the sidebar before making an assessment.")
        else:
            with st.spinner('Assessing eco-health risk and generating recommendations...'):
                time.sleep(2)

                try:
                    eco_health_risk_score = predict_eco_health_risk(aqi, green_space_percent, avg_age, avg_bmi)

                    st.subheader(f"Eco-Health Risk Score for {selected_city_name}: **{eco_health_risk_score:.2f}**")

                    st.subheader("Sustainable Urban Planning Recommendations:")
                    st.markdown("Based on the input data, here are tailored recommendations for your selected area:")

                    if aqi > 200:
                        st.markdown("- **Urgent Air Quality Intervention:** Implement strict emission standards for industries and vehicles. Promote electric vehicles and expand public transport networks. Consider deploying large-scale air purification systems in highly affected zones.")
                    elif aqi > 100:
                        st.markdown("- **Enhanced Pollution Monitoring & Awareness:** Strengthen real-time air quality monitoring and launch public awareness campaigns on pollution effects. Encourage carpooling, cycling, and walking.")
                    else:
                        st.markdown("- **Maintain & Protect Air Quality:** Continue promoting clean energy sources, sustainable transportation, and green infrastructure to preserve good air quality standards.")

                    if green_space_percent < 10:
                        st.markdown("- **Aggressive Green Space Development:** Prioritize creating new parks, urban forests, and green corridors. Implement mandatory green rooftops and vertical gardens for new constructions and existing buildings.")
                    elif green_space_percent < 25:
                        st.markdown("- **Expand Green Infrastructure:** Invest in community gardens, street tree planting initiatives, and converting unused urban spaces into accessible green areas. Promote urban farming.")
                    else:
                        st.markdown("- **Green Space Preservation & Enhancement:** Protect existing natural areas within the city and explore opportunities for further expansion, biodiversity enhancement, and ecological restoration.")

                    if avg_age > 50 or avg_bmi > 28:
                        st.markdown("- **Targeted Community Health Programs:** Implement widespread health screenings, promote healthy lifestyle workshops (diet, exercise), and improve access to preventive care, especially focusing on chronic diseases like diabetes and heart disease in vulnerable populations.")
                    else:
                        st.markdown("- **General Wellness & Preventive Care:** Encourage active lifestyles, balanced diets, and regular health check-ups across all age groups to maintain overall community well-being.")

                    if eco_health_risk_score >= 0.7:
                        st.error("🚨 **Overall HIGH ECO-HEALTH RISK DETECTED! Urgent and comprehensive action is critical across all sectors.**")
                        eco_health_status_text = "High Risk"
                    elif eco_health_risk_score >= 0.4:
                        st.warning("⚠️ **Overall MODERATE ECO-HEALTH RISK. Proactive and integrated measures are highly recommended to mitigate future health challenges.**")
                        eco_health_status_text = "Moderate Risk"
                    else:
                        st.success("✅ **Overall LOW ECO-HEALTH RISK. Continue and enhance sustainable practices to maintain and improve community well-being.**")
                        eco_health_status_text = "Low Risk"

                    if save_eco_health_prediction(user_id, selected_city_name, aqi, green_space_percent, avg_age, avg_bmi, eco_health_risk_score):
                        st.success("🎉 Eco-Health risk assessment saved successfully to your project's database!")
                    else:
                        st.error("❗ Failed to save eco-health prediction. Please check for errors.")

                    st.subheader("Geospatial Context:")
                    st.info("*(Future Enhancement: This map will integrate real-time sensor data and AI-driven insights as dynamic digital overlays, providing a comprehensive, live view of eco-health factors across urban areas.)*")

                    m = folium.Map(location=[selected_city_coords["lat"], selected_city_coords["lon"]], zoom_start=11)

                    risk_color = "red" if eco_health_risk_score >= 0.7 else ("orange" if eco_health_risk_score >= 0.4 else "green")
                    risk_radius = 5000 + (eco_health_risk_score * 10000)

                    folium.Circle(
                        location=[selected_city_coords["lat"], selected_city_coords["lon"]],
                        radius=risk_radius,
                        color=risk_color,
                        fill=True,
                        fill_color=risk_color,
                        fill_opacity=0.4,
                        popup=f"Eco-Health Risk: {eco_health_risk_score:.2f} (Higher is worse)"
                    ).add_to(m)

                    folium.Marker(
                        location=[selected_city_coords["lat"], selected_city_coords["lon"]],
                        popup=f"{selected_city_name} - Center",
                        icon=folium.Icon(color="blue", icon="info-sign")
                    ).add_to(m)

                    st_folium(m, width=700, height=500)

                    st.markdown("*(Note: This map is a conceptual representation. For real urban planning, you'd integrate actual geospatial data and more complex risk models.)*")

                except Exception as e:
                    st.error(f"An error occurred during eco-health assessment: {e}. Please ensure all inputs are valid.")

    st.subheader(f"📊 Your Eco-Health Assessments (User ID: {user_id})")
    if user_id:
        saved_eco_health_data = fetch_eco_health_predictions_by_user(user_id)

        if saved_eco_health_data:
            df_saved_eco_health = pd.DataFrame(saved_eco_health_data,
                                               columns=['ID', 'User ID', 'City', 'AQI', 'Green Space (%)',
                                                        'Avg Age', 'Avg BMI', 'Risk Score', 'Timestamp'])
            st.dataframe(df_saved_eco_health)

            # --- Download Button for Eco-Health Predictions ---
            csv_eco_health = df_saved_eco_health.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Your Eco-Health Assessments as CSV",
                data=csv_eco_health,
                file_name=f'{user_id}_eco_health_assessments.csv',
                mime='text/csv',
                key='download_eco_health_csv'
            )
        else:
            st.info(f"No eco-health assessments saved yet for User ID: {user_id}. Make an assessment to see it appear here!")
    else:
        st.info("Enter your User ID in the sidebar to view your past eco-health assessments.")