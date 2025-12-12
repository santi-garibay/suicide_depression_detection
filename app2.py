import streamlit as st
import joblib
import re
import spacy
import speech_recognition as sr # Nueva librería para audio

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
    
    # Logistic Regression / Naive Bayes
    #prediction = model.predict(vectorized_text)[0]
    #probability = model.predict_proba(vectorized_text)[0][1] # Probabilidad de clase 1 (Suicidio)
    
    # Keras
    # 1. Convertir a denso (arreglo numpy)
    input_data = vectorized_text.toarray()

    # 2. Predecir (Keras devuelve probabilidad directamente)
    pred_raw = model.predict(input_data)

    # 3. Extraer el valor
    # Si tu última capa fue Dense(1, activation='sigmoid'), el resultado es [[0.85]]
    probability = pred_raw[0][0] 
    
    return probability

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
            p = analyze_text(user_input)
            
            st.divider()
            if p > 0.5: # Umbral
                st.error(f"⚠️ **Riesgo Detectado** (Probabilidad: {p:.1%})")
                st.warning("El texto contiene patrones asociados a ideación suicida.")
            else:
                st.success(f"✅ **Bajo Riesgo** (Probabilidad: {p:.1%})")
                st.info("No se detectaron patrones de alerta inminente.")

# --- PESTAÑA 2: AUDIO ---
with tab2:
    st.write("Sube un archivo de audio (.wav) o graba un mensaje.")
    
    # Nota: Streamlit tiene un grabador nativo 'st.audio_input' en versiones nuevas (1.39+)
    # Usaremos file_uploader para máxima compatibilidad por ahora
    audio_file = st.file_uploader("Sube tu audio (WAV)", type=["wav"])

    if audio_file is not None:
        st.audio(audio_file) # Reproductor para verificar
        
        if st.button("Transcribir y Analizar"):
            with st.spinner('Escuchando y procesando...'):
                # 1. Transcribir
                transcription = transcribe_audio(audio_file)
                
                if "Error" in transcription or "No se pudo" in transcription:
                    st.error(transcription)
                else:
                    st.subheader("Texto Transcrito:")
                    st.info(f'"{transcription}"')
                    
                    # 2. Analizar el texto transcrito
                    p = analyze_text(transcription)
                    
                    st.divider()
                    st.subheader("Resultado del Modelo:")
                    if p > 0.5:
                        st.error(f"⚠️ **Riesgo Detectado** (Probabilidad: {p:.1%})")
                    else:
                        st.success(f"✅ **Bajo Riesgo** (Probabilidad: {p:.1%})")