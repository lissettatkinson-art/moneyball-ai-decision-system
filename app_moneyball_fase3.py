# app_moneyball_fase3_formateado.py
# Moneyball AI Decision System - Fase 3
# Componentes 1, 2 y 3: arquitectura, prototipo funcional y demo interactiva

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Moneyball AI Decision System",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f4f7fb;
}

[data-testid="stSidebar"] {
    background-color: #0f172a;
}

[data-testid="stSidebar"] label {
    color: white;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: white;
}

.hero-box {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero-box h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero-box p {
    font-size: 18px;
    color: #dbeafe;
}

.custom-card {
    background-color: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.10);
    margin-bottom: 18px;
}

.custom-metric {
    background-color: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.10);
    text-align: center;
}

.custom-metric h4 {
    color: #64748b;
    margin-bottom: 5px;
}

.custom-metric h2 {
    color: #0f172a;
    font-size: 34px;
}

.success-box {
    background-color: #dcfce7;
    border-left: 7px solid #16a34a;
    padding: 18px;
    border-radius: 14px;
    color: #14532d;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero-box">
    <h1>⚾ Moneyball AI Decision System</h1>
    <p>Plataforma inteligente para evaluar jugadores, predecir rendimiento ofensivo y generar recomendaciones con IA.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# ESTILOS VISUALES
# =========================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }

        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
            padding: 34px 38px;
            border-radius: 24px;
            color: white;
            margin-bottom: 26px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.22);
        }

        .hero h1 {
            font-size: 42px;
            margin-bottom: 8px;
            font-weight: 800;
        }

        .hero p {
            font-size: 18px;
            color: #cbd5e1;
            max-width: 940px;
            line-height: 1.55;
        }

        .section-title {
            font-size: 25px;
            font-weight: 800;
            color: #0f172a;
            margin-top: 18px;
            margin-bottom: 14px;
        }

        .subtitle {
            color: #475569;
            font-size: 15px;
            margin-bottom: 15px;
        }

        .metric-card {
            background: #ffffff;
            padding: 22px;
            border-radius: 18px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
            border: 1px solid #e2e8f0;
            min-height: 138px;
        }

        .metric-card h4 {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-card h2 {
            color: #0f172a;
            font-size: 30px;
            margin: 0px;
            font-weight: 800;
        }

        .metric-card p {
            color: #64748b;
            font-size: 13px;
            margin-top: 8px;
        }

        .info-card {
            background: #ffffff;
            padding: 24px;
            border-radius: 18px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.07);
            border: 1px solid #e2e8f0;
            margin-bottom: 18px;
        }

        .success-box {
            background-color: #dcfce7;
            border-left: 7px solid #16a34a;
            padding: 18px 20px;
            border-radius: 14px;
            font-weight: 700;
            color: #14532d;
            margin-top: 16px;
            margin-bottom: 18px;
        }

        .warning-box {
            background-color: #fef9c3;
            border-left: 7px solid #ca8a04;
            padding: 18px 20px;
            border-radius: 14px;
            font-weight: 700;
            color: #713f12;
            margin-top: 16px;
            margin-bottom: 18px;
        }

        .danger-box {
            background-color: #fee2e2;
            border-left: 7px solid #dc2626;
            padding: 18px 20px;
            border-radius: 14px;
            font-weight: 700;
            color: #7f1d1d;
            margin-top: 16px;
            margin-bottom: 18px;
        }

        .llm-box {
            background: linear-gradient(135deg, #eff6ff, #f8fafc);
            border: 1px solid #bfdbfe;
            border-left: 7px solid #2563eb;
            padding: 22px;
            border-radius: 16px;
            color: #0f172a;
            line-height: 1.6;
            margin-top: 12px;
        }

        .small-note {
            color: #64748b;
            font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FUNCIONES
# =========================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    """Carga el dataset desde la misma carpeta del app o desde Downloads como respaldo."""
    local_path = Path(__file__).parent / "moneyball_dataset_fase3.csv"
    downloads_path = Path.home() / "Downloads" / "moneyball_dataset_fase3.csv"

    if local_path.exists():
        return pd.read_csv(local_path)
    if downloads_path.exists():
        return pd.read_csv(downloads_path)

    st.error(
        "No se encontró el archivo moneyball_dataset_fase3.csv. "
        "Colócalo en la misma carpeta del archivo app o en Downloads."
    )
    st.stop()


@st.cache_resource
def train_model(data: pd.DataFrame):
    features = ["age", "experience_years", "OBP", "SLG", "AVG", "HR", "RBI", "BB", "salary_usd"]
    target = "projected_runs"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        max_depth=8
    )
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2 = r2_score(y_test, pred_test)

    return model, mae, rmse, r2, features


def get_groq_client():
    """Obtiene la API Key desde secrets o variable de entorno."""
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def generar_explicacion_ia(client, datos_jugador: dict, resultados: dict) -> str:
    prompt = f"""
    Eres un analista deportivo experto en Moneyball, analítica avanzada e inteligencia artificial aplicada al negocio deportivo.

    Analiza el siguiente jugador evaluado por el sistema:

    Edad: {datos_jugador['age']}
    Experiencia: {datos_jugador['experience_years']}
    OBP: {datos_jugador['OBP']}
    SLG: {datos_jugador['SLG']}
    AVG: {datos_jugador['AVG']}
    Home Runs: {datos_jugador['HR']}
    RBI: {datos_jugador['RBI']}
    BB: {datos_jugador['BB']}
    Salario: {datos_jugador['salary_usd']}

    Resultados del modelo:
    - Carreras proyectadas: {resultados['projected_runs_predicted']}
    - Victorias estimadas: {resultados['estimated_wins_added']}
    - Costo por carrera proyectada: {resultados['cost_per_projected_run']}
    - Índice de valor: {resultados['value_index']}
    - Recomendación del motor de decisión: {resultados['business_decision']}

    Genera un análisis ejecutivo en español, claro y profesional, con esta estructura:
    1. Lectura general del perfil del jugador.
    2. Fortalezas principales.
    3. Riesgos o alertas.
    4. Relación costo-beneficio.
    5. Recomendación final para la gerencia deportiva.

    No inventes datos adicionales. Basa tu análisis solo en los indicadores entregados.
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=900,
    )

    return chat_completion.choices[0].message.content


# =========================================================
# CARGA Y ENTRENAMIENTO
# =========================================================
df = load_data()
model, mae, rmse, r2, features = train_model(df)


# =========================================================
# ENCABEZADO PRINCIPAL
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>⚾ Moneyball AI Decision System</h1>
        <p>
        Plataforma inteligente para evaluar jugadores, predecir rendimiento ofensivo,
        identificar talento subvalorado y generar recomendaciones ejecutivas mediante IA generativa.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR - ENTRADAS DEL USUARIO
# =========================================================
st.sidebar.title("⚙️ Evaluación de jugador")
st.sidebar.caption("Ingrese las métricas ofensivas, económicas y de contexto del jugador.")

age = st.sidebar.slider("Edad", 20, 40, 28)
experience_years = st.sidebar.slider("Años de experiencia", 0, 18, 5)
OBP = st.sidebar.number_input("OBP - On Base Percentage", min_value=0.200, max_value=0.500, value=0.340, step=0.001, format="%.3f")
SLG = st.sidebar.number_input("SLG - Slugging", min_value=0.250, max_value=0.700, value=0.430, step=0.001, format="%.3f")
AVG = st.sidebar.number_input("AVG - Promedio de bateo", min_value=0.150, max_value=0.400, value=0.260, step=0.001, format="%.3f")
HR = st.sidebar.number_input("Home Runs", min_value=0, max_value=60, value=15)
RBI = st.sidebar.number_input("RBI", min_value=0, max_value=150, value=60)
BB = st.sidebar.number_input("Bases por bolas", min_value=0, max_value=120, value=45)
salary_usd = st.sidebar.number_input("Salario USD", min_value=300000, max_value=20000000, value=2500000, step=100000)

evaluar = st.sidebar.button("🚀 Evaluar jugador", use_container_width=True)


# =========================================================
# PESTAÑAS PRINCIPALES
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Arquitectura",
    "🤖 Prototipo IA",
    "📊 Dataset y variables",
    "🧠 Explicación generativa"
])


# =========================================================
# TAB 1 - ARQUITECTURA
# =========================================================
with tab1:
    st.markdown('<div class="section-title">Componente 1 — Arquitectura y diseño del sistema</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Pipeline completo de la solución desde la entrada de datos hasta la recomendación final.</p>', unsafe_allow_html=True)

    st.markdown("""
```mermaid
flowchart LR
    A[Entrada: datos del jugador] --> B[Procesamiento en Python/Pandas]
    B --> C[Limpieza y validación de variables]
    C --> D[Modelo ML: Random Forest Regressor]
    D --> E[Predicción de rendimiento ofensivo]
    E --> F[Motor de reglas de negocio]
    F --> G[Recomendación: Fichar / Monitorear / No fichar]
    E --> H[LLM: explicación generativa con Groq]
    F --> I[Log estructurado: DataFrame/JSON]
    G --> J[Interfaz Streamlit]
    H --> J
    I --> J
```
""")

    st.markdown('<div class="section-title">Stack tecnológico documentado</div>', unsafe_allow_html=True)

    stack = pd.DataFrame({
        "Componente": [
            "Carga y preparación de datos",
            "Modelo predictivo",
            "IA generativa",
            "Motor de decisión",
            "Log de resultados",
            "Interfaz demo",
            "Visualización ejecutiva"
        ],
        "Herramienta": [
            "Python + Pandas",
            "Scikit-learn / Random Forest Regressor",
            "Groq API / Llama 3.3 70B Versatile",
            "Reglas de negocio en Python",
            "DataFrame / CSV / JSON",
            "Streamlit",
            "Power BI"
        ],
        "Uso en el proyecto": [
            "Lectura, limpieza y transformación de métricas ofensivas y económicas.",
            "Predicción del rendimiento ofensivo esperado del jugador.",
            "Generación automática de explicación ejecutiva sobre la recomendación.",
            "Clasificación del jugador según valor, salario, riesgo y rendimiento esperado.",
            "Registro trazable de entradas, predicciones y recomendaciones.",
            "Demo interactiva para probar jugadores nuevos en tiempo real.",
            "Seguimiento de KPIs estratégicos y análisis costo-rendimiento."
        ]
    })

    st.dataframe(stack, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Justificación de decisiones de diseño</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
        El sistema se diseña como una solución híbrida que combina analítica predictiva, reglas de negocio e IA generativa.
        El modelo Random Forest Regressor se utiliza porque permite capturar relaciones no lineales entre variables ofensivas,
        económicas y de contexto. Esto es coherente con el enfoque Moneyball, donde el valor real del jugador no depende de
        una sola métrica, sino de una combinación de factores.
        <br><br>
        El motor de reglas traduce la predicción en una decisión de negocio comprensible: fichar, monitorear o descartar.
        La IA generativa se integra para explicar la recomendación en lenguaje ejecutivo, facilitando la adopción por parte
        de directivos y scouts que no necesariamente dominan la parte técnica.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# TAB 2 - PROTOTIPO IA
# =========================================================
with tab2:
    st.markdown('<div class="section-title">Componente 2 — Prototipo funcional</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Modelo predictivo entrenado con datos simulados de la industria deportiva.</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>MAE</h4>
            <h2>{round(mae, 2)}</h2>
            <p>Error absoluto medio del modelo.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>RMSE</h4>
            <h2>{round(rmse, 2)}</h2>
            <p>Error cuadrático medio.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>R²</h4>
            <h2>{round(r2, 2)}</h2>
            <p>Capacidad explicativa del modelo.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Evaluación interactiva</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
        Use el panel lateral para ingresar los datos de un nuevo jugador. Al presionar <b>Evaluar jugador</b>,
        el sistema ejecuta el pipeline completo: predicción, motor de reglas, cálculo de KPIs y generación del log.
        </div>
        """,
        unsafe_allow_html=True
    )

    if evaluar:
        nuevo_jugador = pd.DataFrame([{
            "age": age,
            "experience_years": experience_years,
            "OBP": OBP,
            "SLG": SLG,
            "AVG": AVG,
            "HR": HR,
            "RBI": RBI,
            "BB": BB,
            "salary_usd": salary_usd
        }])

        prediccion = model.predict(nuevo_jugador)[0]

        value_index = (prediccion / salary_usd) * 1_000_000
        estimated_wins_added = prediccion / 10
        cost_per_projected_run = salary_usd / prediccion

        # ==============================
        # Motor de decisión avanzado
        # ==============================

        score = 0
        criterios = []

        # 1. Rendimiento ofensivo proyectado
        if prediccion >= 115:
            score += 25
            criterios.append("Alto rendimiento ofensivo proyectado")
        elif prediccion >= 90:
            score += 15
            criterios.append("Rendimiento ofensivo aceptable")
        else:
            score -= 10
            criterios.append("Rendimiento ofensivo bajo")

        # 2. Capacidad para llegar a base
        if OBP >= 0.360:
            score += 20
            criterios.append("OBP superior al promedio, alineado al enfoque Moneyball")
        elif OBP >= 0.330:
            score += 10
            criterios.append("OBP competitivo")
        else:
            score -= 10
            criterios.append("OBP bajo para el perfil buscado")

        # 3. Potencia ofensiva
        if SLG >= 0.470:
            score += 15
            criterios.append("Buen nivel de slugging y producción ofensiva")
        elif SLG >= 0.400:
            score += 8
            criterios.append("Slugging aceptable")
        else:
            score -= 8
            criterios.append("Baja capacidad de generación de bases")

        # 4. Relación costo-rendimiento
        if value_index >= 30:
            score += 25
            criterios.append("Excelente relación costo-rendimiento")
        elif value_index >= 15:
            score += 15
            criterios.append("Buena relación costo-rendimiento")
        elif value_index >= 8:
            score += 5
            criterios.append("Relación costo-rendimiento moderada")
        else:
            score -= 15
            criterios.append("Baja eficiencia económica")

        # 5. Costo por carrera proyectada
        if cost_per_projected_run <= 25000:
            score += 15
            criterios.append("Costo por carrera altamente eficiente")
        elif cost_per_projected_run <= 60000:
            score += 8
            criterios.append("Costo por carrera razonable")
        else:
            score -= 10
            criterios.append("Costo por carrera elevado")

        # 6. Edad y etapa deportiva
        if 24 <= age <= 31:
            score += 10
            criterios.append("Edad dentro del rango óptimo de rendimiento")
        elif 32 <= age <= 35:
            score += 3
            criterios.append("Jugador con experiencia, pero posible etapa de declive")
        else:
            score -= 5
            criterios.append("Edad fuera del rango ideal")

        # 7. Experiencia
        if experience_years >= 5:
            score += 8
            criterios.append("Experiencia competitiva relevante")
        elif experience_years >= 2:
            score += 4
            criterios.append("Experiencia moderada")
        else:
            score -= 3
            criterios.append("Experiencia limitada")

        # 8. Producción ofensiva tradicional
        if HR >= 20 or RBI >= 75:
            score += 8
            criterios.append("Buena producción ofensiva tradicional")
        elif HR >= 10 or RBI >= 50:
            score += 4
            criterios.append("Producción ofensiva media")
        else:
            score -= 4
            criterios.append("Baja producción ofensiva tradicional")

        # ==============================
        # Decisión final
        # ==============================

        if score >= 70:
            decision = "Fichar"
            decision_class = "success-box"
            nivel_riesgo = "Bajo"
        elif score >= 45:
            decision = "Monitorear"
            decision_class = "warning-box"
            nivel_riesgo = "Medio"
        else:
            decision = "No fichar"
            decision_class = "danger-box"
            nivel_riesgo = "Alto"

       

        st.markdown("### Criterios evaluados por el motor de decisión")
        for criterio in criterios:
            st.write(f"- {criterio}")

        st.session_state["ultimo_resultado"] = {
            "age": age,
            "experience_years": experience_years,
            "OBP": OBP,
            "SLG": SLG,
            "AVG": AVG,
            "HR": HR,
            "RBI": RBI,
            "BB": BB,
            "salary_usd": salary_usd,
            "projected_runs_predicted": round(prediccion, 2),
            "estimated_wins_added": round(estimated_wins_added, 2),
            "cost_per_projected_run": round(cost_per_projected_run, 2),
            "value_index": round(value_index, 2),
            "strategic_score": score,
            "risk_level": nivel_riesgo,
            "decision_criteria": " | ".join(criterios),
            "business_decision": decision
        }

    if "ultimo_resultado" in st.session_state:
        resultado = st.session_state["ultimo_resultado"]

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Carreras proyectadas", resultado["projected_runs_predicted"])
        with r2:
            st.metric("Victorias estimadas", resultado["estimated_wins_added"])
        with r3:
            st.metric("Índice de valor", resultado["value_index"])
        with r4:
            st.metric("Costo por carrera", f"${resultado['cost_per_projected_run']:,.2f}")

        box_class = "success-box" if resultado["business_decision"] == "Fichar" else "warning-box" if resultado["business_decision"] == "Monitorear" else "danger-box"
        
        st.markdown(f"""
        <div class="{decision_class}">
            Recomendación del motor de decisión: {decision}<br>
            Puntaje estratégico: {score}/100<br>
            Nivel de riesgo: {nivel_riesgo}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Log estructurado del resultado</div>', unsafe_allow_html=True)
        log_resultado = pd.DataFrame([resultado])
        st.dataframe(log_resultado, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Descargar log en CSV",
            data=log_resultado.to_csv(index=False).encode("utf-8-sig"),
            file_name="log_resultado_moneyball.csv",
            mime="text/csv"
        )
    else:
        st.info("Ingrese los datos en el panel lateral y presione 'Evaluar jugador' para ejecutar el prototipo.")


# =========================================================
# TAB 3 - DATASET Y VARIABLES
# =========================================================
with tab3:
    st.markdown('<div class="section-title">Dataset base del prototipo</div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Jugadores simulados", len(df))
    d2.metric("Salario promedio", f"${df['salary_usd'].mean():,.0f}")
    d3.metric("Rendimiento promedio", f"{df['projected_runs'].mean():.1f}")
    d4.metric("Índice valor promedio", f"{df['value_index'].mean():.2f}")

    st.dataframe(df.head(30), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Variables utilizadas</div>', unsafe_allow_html=True)
    variables = pd.DataFrame({
        "Variable": [
            "OBP", "SLG", "AVG", "HR", "RBI", "BB", "age", "experience_years",
            "position", "injury_risk", "salary_usd", "projected_runs"
        ],
        "Tipo": [
            "Predictora", "Predictora", "Predictora", "Predictora", "Predictora", "Predictora",
            "Contexto", "Contexto", "Contexto", "Riesgo", "Económica", "Objetivo"
        ],
        "Justificación": [
            "Mide la capacidad del jugador para llegar a base; clave en Moneyball.",
            "Mide potencia ofensiva y capacidad de generar bases.",
            "Métrica tradicional usada como referencia comparativa.",
            "Indica poder ofensivo.",
            "Mide contribución a carreras impulsadas.",
            "Representa disciplina y capacidad para obtener bases por bolas.",
            "Ayuda a interpretar ciclo de vida deportivo.",
            "Captura madurez competitiva.",
            "Permite comparar rendimiento por rol.",
            "Controla exposición a pérdida de rendimiento.",
            "Permite evaluar eficiencia económica.",
            "Variable objetivo que estima aporte ofensivo."
        ]
    })
    st.dataframe(variables, use_container_width=True, hide_index=True)


# =========================================================
# TAB 4 - IA GENERATIVA
# =========================================================
with tab4:
    st.markdown('<div class="section-title">Componente generativo con LLM</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
        Esta sección integra IA generativa real mediante Groq. El LLM recibe los resultados del modelo predictivo
        y genera una explicación ejecutiva para apoyar la toma de decisiones de la gerencia deportiva.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "ultimo_resultado" not in st.session_state:
        st.warning("Primero evalúe un jugador en la pestaña 'Prototipo IA'.")
    else:
        client = get_groq_client()

        if client is None:
            st.error(
                "No se encontró la API Key de Groq. Configure GROQ_API_KEY en st.secrets o como variable de entorno."
            )
            st.code('GROQ_API_KEY = "tu_appi"', language="toml") 
        else:
            if st.button("🧠 Generar explicación ejecutiva con IA", use_container_width=True):
                with st.spinner("Generando análisis con IA generativa..."):
                    datos_jugador = {
                        "age": st.session_state["ultimo_resultado"]["age"],
                        "experience_years": st.session_state["ultimo_resultado"]["experience_years"],
                        "OBP": st.session_state["ultimo_resultado"]["OBP"],
                        "SLG": st.session_state["ultimo_resultado"]["SLG"],
                        "AVG": st.session_state["ultimo_resultado"]["AVG"],
                        "HR": st.session_state["ultimo_resultado"]["HR"],
                        "RBI": st.session_state["ultimo_resultado"]["RBI"],
                        "BB": st.session_state["ultimo_resultado"]["BB"],
                        "salary_usd": st.session_state["ultimo_resultado"]["salary_usd"]
                    }
                    respuesta_ia = generar_explicacion_ia(
                        client,
                        datos_jugador,
                        st.session_state["ultimo_resultado"]
                    )
                    st.session_state["respuesta_ia"] = respuesta_ia

            if "respuesta_ia" in st.session_state:
                st.markdown(
                    f"""
                    <div class="llm-box">
                    {st.session_state['respuesta_ia'].replace(chr(10), '<br>')}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================================================
# NOTA FINAL
# =========================================================
st.caption(
    "Proyecto académico basado en Moneyball: prototipo de IA híbrida con Machine Learning, reglas de negocio e IA generativa."
)
