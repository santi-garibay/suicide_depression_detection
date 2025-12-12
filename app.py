import streamlit as st
import joblib
import re
import spacy
import speech_recognition as sr # Nueva librería para audio
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. CONFIGURACIÓN Y CARGA ---
st.set_page_config(page_title="Detector de Riesgo", page_icon="🧠")

@st.cache_resource
def load_assets():
    # Asegúrate de que los nombres coincidan con tus archivos .pkl
    model = joblib.load('modelo_suicidio.pkl')
    vectorizer = joblib.load('processed_data/tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_assets()
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# --- 2. FUNCIONES ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def analyze_text(text_input):
    """Función auxiliar para reutilizar la lógica de predicción"""
    cleaned_text = preprocess_text(text_input)
    # IMPORTANTE: Usar .transform() y luego .toarray() para compatibilidad
    vectorized_text = vectorizer.transform([cleaned_text])
    
    try:
        # Logistic Regression / Naive Bayes
        probability = model.predict_proba(vectorized_text)[0][1]
    except:
        # Keras
        input_data = vectorized_text.toarray()
        pred_raw = model.predict(input_data)
        probability = pred_raw[0][0] 

    return probability

# Cargar VADER una sola vez
analyzer = SentimentIntensityAnalyzer()

def analyze_robust(text_input):
    # 1. PREPROCESAMIENTO Y MODELO BASE (Tu TF-IDF)
    cleaned_text = preprocess_text(text_input)
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Obtener probabilidad cruda del modelo de suicidio
    try:
        suicide_prob = model.predict_proba(vectorized_text)[0][1]
    except:
        input_data = vectorized_text.toarray()
        pred_raw = model.predict(input_data)
        suicide_prob = pred_raw[0][0] 

    # 2. ANÁLISIS DE SENTIMIENTO (El Juez)
    sentiment_scores = analyzer.polarity_scores(text_input)
    compound_score = sentiment_scores['compound'] 
    # compound va de -1 (Muy Negativo) a +1 (Muy Positivo)

    print(f"Probabilidad Modelo: {suicide_prob:.2f}")
    print(f"Score Sentimiento: {compound_score:.2f}")

    # 3. LÓGICA DE ENSAMBLE (La regla robusta)
    
    # CASO A: Falso Positivo por Felicidad Extrema
    # Si el texto es claramente positivo (compound > 0.5), 
    # es casi imposible que sea una nota suicida inminente.
    if compound_score > 0.5:
        final_risk = 0.0 # Forzamos riesgo bajo
        reason = "Sentiment Override (Positivo)"
        
    # CASO B: Falso Positivo por "No quiero morir" (Negación fuerte)
    # VADER detecta muy bien el "not want to die".
    elif "not" in text_input.lower() and compound_score > -0.2:
        final_risk = suicide_prob * 0.4 # Castigamos la probabilidad
        reason = "Negation Penalty"
        
    # CASO C: Riesgo Real (Texto negativo + Modelo alerta)
    else:
        final_risk = suicide_prob
        reason = "Model Consensus"

    return final_risk, reason

def transcribe_audio(audio_file):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
            # Usamos Google Speech Recognition (requiere internet, pero es gratis y bueno)
            text = r.recognize_google(audio_data, language="en-US")
            return text
    except sr.UnknownValueError:
        return "No se pudo entender el audio."
    except sr.RequestError:
        return "Error de conexión con el servicio de transcripción."
    except Exception as e:
        return f"Error procesando audio: {e}"

# --- 3. INTERFAZ GRÁFICA ---
st.title("🧠 Detección de Riesgo (Texto y Audio)")
st.markdown("Esta herramienta analiza patrones lingüísticos para detectar riesgo de suicidio.")

# Creamos pestañas para organizar la app
tab1, tab2 = st.tabs(["📝 Escribir Texto", "🎙️ Subir/Grabar Audio"])

# --- PESTAÑA 1: TEXTO ---
with tab1:
    user_input = st.text_area("Escribe aquí para analizar:", height=150)
    if st.button("Analizar Texto"):
        if user_input:
            # 1. DESEMPACAMOS LOS DOS VALORES AQUÍ
            # p recibe el número, 'reason' recibe la explicación
            p, reason = analyze_robust(user_input) 
            
            st.divider()
            
            # 2. Ahora 'p' ya es un número (float), así que el IF funciona
            if p > 0.85:
                st.error(f"🚨 **ALERTA DE RIESGO ALTO** (Probabilidad: {p:.1%})")
                st.write("El modelo detectó patrones muy fuertes de ideación suicida.")
                # Opcional: Mostrar por qué decidió esto
                st.caption(f"Motivo técnico: {reason}")
                
            elif p > 0.55:
                st.warning(f"⚠️ **Riesgo Moderado / Alerta de Estrés** (Probabilidad: {p:.1%})")
                st.write("Se detectan palabras negativas, pero podría ser estrés o ansiedad.")
                st.caption(f"Motivo técnico: {reason}")
                
            else:
                st.success(f"✅ **Bajo Riesgo** (Probabilidad: {p:.1%})")
                st.info("No se detectaron patrones de alerta inminente.")
                st.caption(f"Motivo técnico: {reason}")

# --- PESTAÑA 2: AUDIO ---
with tab2:
    st.header("🎙️ Cuéntame cómo te sientes")
    st.write("Presiona el micrófono para empezar a grabar.")
    
    # 1. EL NUEVO GRABADOR NATIVO
    # Esto crea un botón rojo de grabación
    audio_val = st.audio_input("Graba tu nota de voz aquí")

    if audio_val is not None:
        # Mostrar reproductor para que el usuario se escuche antes de enviar
        st.audio(audio_val)
        
        if st.button("Analizar Grabación"):
            with st.spinner('Procesando audio...'):
                
                # 2. Transcribir
                # La función transcribe_audio acepta el objeto de audio directo
                transcription = transcribe_audio(audio_val)
                
                if "Error" in transcription or "No se pudo" in transcription:
                    st.error(f"⚠️ {transcription}")
                    st.info("Intenta hablar un poco más fuerte o revisar tu micrófono.")
                else:
                    st.subheader("Lo que entendí:")
                    st.info(f'"{transcription}"')
                    
                    # 3. Analizar (Usando la función robusta que hicimos antes)
                    # Nota: Asegúrate de usar 'analyze_robust' si ya la implementaste
                    # Si no, usa 'analyze_text'
                    
                    # Desempacamos el resultado (Probabilidad y Razón)
                    # OJO: Cambia 'analyze_robust' por el nombre real de tu función final
                    p, reason = analyze_robust(transcription) 
                    
                    st.divider()
                    st.subheader("Resultado del Modelo:")
                    
                    # Lógica de visualización
                    if p > 0.85:
                        st.error(f"🚨 **RIESGO ALTO** ({p:.1%})")
                        st.caption(f"Motivo: {reason}")
                    elif p > 0.55:
                        st.warning(f"⚠️ **Riesgo Moderado** ({p:.1%})")
                        st.caption(f"Motivo: {reason}")
                    else:
                        st.success(f"✅ **Bajo Riesgo** ({p:.1%})")
                        st.caption(f"Motivo: {reason}")