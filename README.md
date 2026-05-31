# Football Analytics Colombia & Champions AI

Plataforma avanzada de analítica deportiva con Big Data, Machine Learning e Inteligencia Artificial enfocada en la final de la UEFA Champions League y el fútbol colombiano.

## 📋 Descripción

Este proyecto implementa una plataforma completa de analítica deportiva que responde preguntas clave como:
- ¿Qué equipo tiene mayor probabilidad de ganar?
- ¿Qué jugadores generan más peligro?
- ¿Qué patrones tácticos utiliza cada equipo?
- ¿Qué cambios producen más impacto?
- ¿Qué jugadores del fútbol colombiano tienen potencial para ligas europeas?
- ¿Qué futbolistas están infravalorados en la liga local?
- ¿Cuál es el riesgo de lesión de cada jugador?

## 🏗️ Arquitectura

La plataforma sigue una arquitectura profesional de datos:

```
Fuentes de Datos
        │
        ▼
    Data Lake
        │
        ▼
  ETL y Procesamiento
        │
        ▼
  Feature Engineering
        │
        ▼
    Machine Learning
        │
        ▼
   Dashboard + API
        │
        ▼
Predicciones y Alertas
```

## 🚀 Características

### 1. Predictor de Partidos
- Modelo XGBoost para predecir resultados (victoria/empate/derrota)
- Características: forma del equipo, xG/xGA, posesión, presión, lesiones, tiros
- Entrenado con datos sintéticos que reflejan patrones reales de élite

### 2. Calculadora xG Avanzada
- Calcula Expected Goals basado en ubicación del tiro, tipo de juego y contexto
- Características: coordenadas (x,y), distancia/ángulo al arco, parte del cuerpo, tipo de asistencia, situación de juego, etc.

### 3. Sistema de Scouting Colombiano
- Clustering de jugadores por roles (8 posiciones identificadas)
- Búsqueda de jugadores similares mediante similitud coseno
- Predicción de potencial para ligas europeas
- Detección de jugadores subvalorados

### 4. Bot de Predicciones Automáticas
- Predicciones diarias automatizadas para Champions League y fútbol colombiano
- Entrenamiento automático de modelos cuando es necesario
- Reportes en formato JSON y texto

## 📁 Estructura del Proyecto

```
football_analytics/
├── index.html                  # Landing page estática
├── streamlit_app.py            # Aplicación web interactiva (Streamlit)
├── today_colombia_prediction.py # Script para predicciones diarias de Colombia
├── predict_champions_final.py  # Predicción específica para final de Champions
├── requirements.txt            # Dependencias de Python
└── models/
    ├── match_predictor.py      # Predictor de resultados (XGBoost)
    ├── xg_calculator.py       # Calculadora de Expected Goals
    ├── colombian_scouting.py  # Sistema de scouting para talento colombiano
    └── saved_model/           # Modelos entrenados (generados durante ejecución)
```

## 🛠️ Instalación y Uso

### Requisitos
- Python 3.8+
- pip

### Instalación de dependencias
```bash
pip install -r requirements.txt
```

### Ejecutar la aplicación Streamlit
```bash
streamlit run streamlit_app.py
```

### Ejecutar predicciones para hoy (Colombia)
```bash
python today_colombia_prediction.py
```

### Ejecutar predicción para final de Champions League
```bash
python predict_champions_final.py
```

### Ejecutar el bot de predicciones automáticas
```bash
python auto_prediction_bot.py
```

## 📊 Modelos Utilizados

- **XGBoost**: Para predicción de resultados de partidos y cálculo de xG
- **K-Means**: Para clustering de jugadores por estilo de juego
- **Random Forest**: Para predicción de potencial europeo
- **Cosine Similarity**: Para búsqueda de jugadores similares

## 📈 Métricas Avanzadas Implementadas

- Expected Goals (xG)
- Expected Assists (xA)
- Progressive Passes
- PPDA (Presses Per Defensive Action)
- Various team and player performance metrics

## 🌐 Despliegue

La aplicación Streamlit se puede desplegar fácilmente en:
- [Streamlit Community Cloud](https://streamlit.io/cloud)
- Heroku
- AWS Elastic Beanstalk
- Docker

Ejemplo de archivo `Dockerfile` para despliegue:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:
1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Haz commit de tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- Inspirado en el trabajo de empresas como StatsBomb, Opta Sports y Wyscout
- Basado en metodologías de análisis de departamentos de élite de clubes de fútbol
- Utiliza técnicas estándar de la industria de analítica deportiva

---

**Football Analytics Colombia & Champions AI** - Transformando datos en decisiones inteligentes para el fútbol.