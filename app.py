import streamlit as st
import joblib
import re
import spacy
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. CONFIGURATION AND LOADING ---
st.set_page_config(page_title="Risk Detector", page_icon="🧠")

@st.cache_resource
def load_assets():
    # Make sure filenames match your .pkl files
    model = joblib.load('models/modelo_suicidio.pkl')
    # Update this path if necessary
    vectorizer = joblib.load('processed_data/tfidf_vectorizer.pkl') 
    return model, vectorizer

try:
    model, vectorizer = load_assets()
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# --- 2. FUNCTIONS ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Load VADER once
analyzer = SentimentIntensityAnalyzer()

def analyze_robust(text_input):
    # 1. PREPROCESSING AND BASE MODEL (Your TF-IDF)
    cleaned_text = preprocess_text(text_input)
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Get raw probability from suicide model
    try:
        suicide_prob = model.predict_proba(vectorized_text)[0][1]
    except:
        # Fallback for models without predict_proba
        input_data = vectorized_text.toarray()
        pred_raw = model.predict(input_data)
        suicide_prob = pred_raw[0][0] 

    # 2. SENTIMENT ANALYSIS (The Judge)
    sentiment_scores = analyzer.polarity_scores(text_input)
    compound_score = sentiment_scores['compound'] 
    # compound ranges from -1 (Very Negative) to +1 (Very Positive)

    # Debug prints (visible in terminal)
    print(f"Model Probability: {suicide_prob:.2f}")
    print(f"Sentiment Score: {compound_score:.2f}")

    # 3. ENSEMBLE LOGIC (The Robust Rule)
    
    # CASE A: False Positive due to Extreme Happiness
    if compound_score > 0.5:
        final_risk = 0.0 # Force low risk
        reason = "Sentiment Override (Positive)"
        
    # CASE B: False Positive due to "Not want to die" (Strong Negation)
    elif "not" in text_input.lower() and compound_score > -0.2:
        final_risk = suicide_prob * 0.4 # Penalize probability
        reason = "Negation Penalty"
        
    # CASE C: Real Risk (Negative Text + Model Alert)
    else:
        final_risk = suicide_prob
        reason = "Model Consensus"

    return final_risk, reason

def transcribe_audio(audio_file):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
            # using en-US since the app is now in English
            text = r.recognize_google(audio_data, language="en-US")
            return text
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Connection error with transcription service."
    except Exception as e:
        return f"Error processing audio: {e}"

# --- 3. USER INTERFACE ---
st.title("🧠 Risk Detection (Text & Audio)")
st.markdown("This tool analyzes linguistic patterns to detect potential suicide risk or depression.")

# Tabs for organizing the app
tab1, tab2 = st.tabs(["📝 Write Text", "🎙️ Record Audio"])

# --- TAB 1: TEXT ---
with tab1:
    user_input = st.text_area("Write here to analyze:", height=150)
    
    if st.button("Analyze Text"):
        if user_input:
            # 1. Unpack the two values
            p, reason = analyze_robust(user_input) 
            
            st.divider()
            
            # 2. Logic to display results
            if p > 0.85:
                st.error(f"🚨 **HIGH RISK ALERT** (Probability: {p:.1%})")
                st.write("The model detected strong patterns of suicidal ideation.")
                st.caption(f"Technical reason: {reason}")
                
            elif p > 0.55:
                st.warning(f"⚠️ **Moderate Risk / Stress Alert** (Probability: {p:.1%})")
                st.write("Negative words detected, but it could be severe stress or anxiety.")
                st.caption(f"Technical reason: {reason}")
                
            else:
                st.success(f"✅ **Low Risk** (Probability: {p:.1%})")
                st.info("No imminent alert patterns detected.")
                st.caption(f"Technical reason: {reason}")

# --- TAB 2: AUDIO ---
with tab2:
    st.header("🎙️ Tell me how you feel")
    st.write("Press the microphone to start recording.")
    
    # 1. NATIVE AUDIO RECORDER
    audio_val = st.audio_input("Record your voice note here")

    if audio_val is not None:
        # Audio player to verify recording
        st.audio(audio_val)
        
        if st.button("Analyze Recording"):
            with st.spinner('Processing audio...'):
                
                # 2. Transcribe
                transcription = transcribe_audio(audio_val)
                
                if "Error" in transcription or "Could not" in transcription:
                    st.error(f"⚠️ {transcription}")
                    st.info("Try speaking a bit louder or check your microphone.")
                else:
                    st.subheader("What I understood:")
                    st.info(f'"{transcription}"')
                    
                    # 3. Analyze
                    p, reason = analyze_robust(transcription) 
                    
                    st.divider()
                    st.subheader("Model Result:")
                    
                    # Display Logic
                    if p > 0.85:
                        st.error(f"🚨 **HIGH RISK ALERT** (Probability: {p:.1%})")
                        st.write("The model detected strong patterns of suicidal ideation.")
                        st.caption(f"Technical reason: {reason}")
                        
                    elif p > 0.55:
                        st.warning(f"⚠️ **Moderate Risk / Stress Alert** (Probability: {p:.1%})")
                        st.write("Negative words detected, but it could be severe stress or anxiety.")
                        st.caption(f"Technical reason: {reason}")
                        
                    else:
                        st.success(f"✅ **Low Risk** (Probability: {p:.1%})")
                        st.info("No imminent alert patterns detected.")
                        st.caption(f"Technical reason: {reason}")