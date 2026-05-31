import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from match_predictor import MatchPredictor
from xg_calculator import xGCalculator

# Page configuration
st.set_page_config(
    page_title="Football Analytics Colombia",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚽ Football Analytics Colombia & Champions AI")
st.markdown("""
### Predicciones diarias para fútbol colombiano y Champions League
Plataforma de analítica deportiva con Machine Learning y Big Data
""")

# Sidebar
st.sidebar.header("⚙️ Configuración")
show_raw_data = st.sidebar.checkbox("Mostrar datos crudos", value=False)
refresh_data = st.sidebar.button("Actualizar predicciones")

# Initialize predictors
@st.cache_resource
def load_predictors():
    match_predictor = MatchPredictor()
    xg_calculator = xGCalculator()
    
    # Try to load pre-trained models
    model_dir = "./data/models"
    try:
        match_predictor.load_model(model_dir)
        xg_calculator.load_model(model_dir)
        st.sidebar.success("✓ Modelos cargados")
    except:
        st.sidebar.warning("⚠ Entrenando modelos básicos...")
        # In production, you would train with recent data
        # For demo, we'll use default initialization
    
    return match_predictor, xg_calculator

match_predictor, xg_calculator = load_predictors()

# Function to get today's fixtures (sample data)
def get_today_fixtures():
    """Get sample fixtures for today's Colombian matches."""
    today = datetime.now().strftime("%Y-%m-%d")
    
    fixtures = [
        {
            'match_id': f'COL_{today.replace("-", "")}_01',
            'date': today,
            'home_team': 'Atlético Nacional',
            'away_team': 'Millonarios FC',
            'venue': 'Estadio Atanasio Girardot, Medellín',
            'competition': 'Liga BetPlay Dimayor',
            'stage': 'Fecha 18',
            'team_form_home': 0.78,
            'team_form_away': 0.72,
            'xG_home': 1.65,
            'xG_away': 1.52,
            'xGA_home': 0.88,
            'xGA_away': 0.95,
            'possession_home': 52,
            'possession_away': 48,
            'pressing_home': 0.65,
            'pressing_away': 0.60,
            'injuries_home': 1,
            'injuries_away': 2,
            'shots_home': 14,
            'shots_away': 13,
            'shots_on_target_home': 6,
            'shots_on_target_away': 5
        },
        {
            'match_id': f'COL_{today.replace("-", "")}_02',
            'date': today,
            'home_team': 'América de Cali',
            'away_team': 'Deportivo Cali',
            'venue': 'Estadio Pascual Guerrero, Cali',
            'competition': 'Liga BetPlay Dimayor',
            'stage': 'Fecha 18',
            'team_form_home': 0.65,
            'team_form_away': 0.58,
            'xG_home': 1.40,
            'xG_away': 1.30,
            'xGA_home': 1.05,
            'xGA_away': 0.98,
            'possession_home': 50,
            'possession_away': 50,
            'pressing_home': 0.55,
            'pressing_away': 0.50,
            'injuries_home': 2,
            'injuries_away': 1,
            'shots_home': 12,
            'shots_away': 14,
            'shots_on_target_home': 4,
            'shots_on_target_away': 6
        }
    ]
    
    return fixtures

# Function to make prediction for a fixture
def predict_fixture(fixture):
    """Generate prediction for a single fixture."""
    match_features = {
        'team_form_home': fixture['team_form_home'],
        'team_form_away': fixture['team_form_away'],
        'xG_home': fixture['xG_home'],
        'xG_away': fixture['xG_away'],
        'xGA_home': fixture['xGA_home'],
        'xGA_away': fixture['xGA_away'],
        'possession_home': fixture['possession_home'],
        'possession_away': fixture['possession_away'],
        'pressing_home': fixture['pressing_home'],
        'pressing_away': fixture['pressing_away'],
        'injuries_home': fixture['injuries_home'],
        'injuries_away': fixture['injuries_away'],
        'shots_home': fixture['shots_home'],
        'shots_away': fixture['shots_away'],
        'shots_on_target_home': fixture['shots_on_target_home'],
        'shots_on_target_away': fixture['shots_on_target_away']
    }
    
    try:
        probabilities = match_predictor.predict(match_features)
        predicted_outcome = match_predictor.predict_outcome(match_features)
        
        # Calculate expected score
        expected_home_goals = round(fixture['xG_home'], 1)
        expected_away_goals = round(fixture['xG_away'], 1)
        
        # Format readable outcome
        if predicted_outcome == 'win':
            readable_prediction = f"{fixture['home_team']} gana"
        elif predicted_outcome == 'draw':
            readable_prediction = "Empate"
        else:
            readable_prediction = f"{fixture['away_team']} gana"
        
        return {
            'match_id': fixture['match_id'],
            'date': fixture['date'],
            'home_team': fixture['home_team'],
            'away_team': fixture['away_team'],
            'venue': fixture['venue'],
            'competition': fixture['competition'],
            'stage': fixture['stage'],
            'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'probabilities': probabilities,
            'predicted_outcome': predicted_outcome,
            'readable_prediction': readable_prediction,
            'expected_score': f"{expected_home_goals} - {expected_away_goals}",
            'confidence': max(probabilities['prob_win'], probabilities['prob_draw'], probabilities['prob_loss']),
            'raw_fixture': fixture if show_raw_data else None
        }
    except Exception as e:
        st.error(f"Error al predecir {fixture['home_team']} vs {fixture['away_team']}: {e}")
        return None

# Main content
tab1, tab2 = st.tabs(["📅 Predicciones de Hoy", "ℹ️ Información"])

with tab1:
    st.header("Predicciones para los partidos de hoy")
    
    # Get fixtures and make predictions
    fixtures = get_today_fixtures()
    
    if refresh_data or 'predictions' not in st.session_state:
        with st.spinner("Calculando predicciones..."):
            predictions = []
            for fixture in fixtures:
                pred = predict_fixture(fixture)
                if pred:
                    predictions.append(pred)
            st.session_state.predictions = predictions
    
    predictions = st.session_state.get('predictions', [])
    
    if predictions:
        # Display predictions in cards
        for pred in predictions:
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    st.subheader(f"{pred['home_team']}")
                    st.caption(f"Local • {pred['venue']}")
                
                with col2:
                    st.markdown("<div style='text-align: center; padding: 1rem;'>VS</div>", unsafe_allow_html=True)
                
                with col3:
                    st.subheader(f"{pred['away_team']}")
                    st.caption(f"Visitante • {pred['date']}")
                
                # Prediction details
                st.info(f"**Predicción:** {pred['readable_prediction']} ({pred['confidence']:.1%} confianza)")
                
                # Probabilities
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                with prob_col1:
                    st.metric(
                        label=f"{pred['home_team']} gana",
                        value=f"{pred['probabilities']['prob_win']:.1%}"
                    )
                with prob_col2:
                    st.metric(
                        label="Empate",
                        value=f"{pred['probabilities']['prob_draw']:.1%}"
                    )
                with prob_col3:
                    st.metric(
                        label=f"{pred['away_team']} gana",
                        value=f"{pred['probabilities']['prob_loss']:.1%}"
                    )
                
                # Expected score
                st.success(f"**Marcador esperado:** {pred['expected_score']}")
                
                # Show raw data if requested
                if show_raw_data and pred['raw_fixture']:
                    with st.expander("Ver datos crudos"):
                        st.json(pred['raw_fixture'])
                
                st.divider()
    else:
        st.warning("No se pudieron generar predicciones. Verifique los logs de error.")

with tab2:
    st.header("Acerca de la plataforma")
    st.markdown("""
    ### Football Analytics Colombia & Champions AI
    
    Esta plataforma utiliza:
    - **Machine Learning**: Modelos XGBoost para predicción de resultados
    - **Expected Goals (xG)**: Calculadora avanzada de oportunidades de gol
    - **Análisis de talento**: Sistema de scouting para jugadores colombianos
    - **Automatización**: Bot de predicciones diarias
    
    #### Características principales:
    - Predicción de resultados de partidos (win/draw/loss)
    - Cálculo de Expected Goals basado en ubicación y contexto
    - Identificación de patrones tácticos y estilos de juego
    - Detección de jugadores con potencial europeo
    - Evaluación de riesgo de lesión
    
    #### Tecnologías utilizadas:
    - Python, XGBoost, Scikit-learn
    - Streamlit para la interfaz web
    - Pandas y NumPy para procesamiento de datos
    - Modelos guardados con Joblib
    
    #### Próximos pasos:
    - Integración con datos en tiempo real de APIs deportivas
    - Análisis de video con computer vision
    - Sistema de alertas para lesiones y cambios tácticos
    - Aplicación móvil para scouts y entrenadores
    """)
    
    st.info("💡 **Tip**: Use el sidebar para actualizar las predicciones o ver datos crudos.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Football Analytics Colombia & Champions AI © 2026</p>
        <p>Desarrollado con ❤️ para el análisis deportivo avanzado</p>
    </div>
    """,
    unsafe_allow_html=True
)