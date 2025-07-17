
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import joblib

st.set_page_config(page_title="Prometheus – Fuel Monitor", layout="centered")

st.title("🚍 Prometheus – Fuel Monitor")
st.subheader("Monitoreo inteligente de consumo, eco-conducción y predicción")

st.markdown("""
Esta plataforma permite monitorear y analizar el rendimiento de combustible de una flota de transporte:

- 📈 Carga y análisis de datos reales
- 🧠 Detección de anomalías por ML y Z-score
- 🏁 Ranking de eco-conducción por unidad
- 🔮 Predicción de consumo esperado (L/100km)
- ♻️ Estimación de emisiones CO₂e acumuladas
""")

# --- Carga de archivo ---
archivo = st.file_uploader("Carga un archivo CSV con los datos de consumo", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.success("Archivo cargado exitosamente.")
    st.dataframe(df.head())

    if 'Distancia_km' in df.columns and 'Combustible_L' in df.columns:
        df['Consumo_L100km'] = (df['Combustible_L'] / df['Distancia_km']) * 100
        df['CO2_eq_kg'] = df['Combustible_L'] * 2.68  # Ejemplo: 2.68 kg CO2 por litro

        st.metric("Consumo promedio (L/100km)", round(df['Consumo_L100km'].mean(), 2))
        st.metric("Emisiones CO₂e acumuladas (kg)", round(df['CO2_eq_kg'].sum(), 1))

        # Anomalías por Z-score
        df['z_score'] = (df['Consumo_L100km'] - df['Consumo_L100km'].mean()) / df['Consumo_L100km'].std()
        df['anomalía'] = df['z_score'].abs() > 2

        st.metric("Anomalías detectadas", df['anomalía'].sum())

        # Visualización
        st.subheader("Distribución de consumo")
        fig, ax = plt.subplots()
        sns.histplot(df['Consumo_L100km'], kde=True, ax=ax)
        st.pyplot(fig)

    else:
        st.warning("El archivo debe contener las columnas: 'Distancia_km' y 'Combustible_L'")
else:
    st.info("Esperando carga de archivo CSV para comenzar el análisis.")
