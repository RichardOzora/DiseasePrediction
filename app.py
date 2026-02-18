import streamlit as st
import torch
import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# --- 1. App Configuration & Setup ---
st.set_page_config(
    page_title="BioBERT Disease Predictor",
    page_icon="🩺",
    layout="centered"
)

# Function to download NLTK data quietly if not present
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

setup_nltk()

# Initialize Preprocessing tools
eng_stopwords = stopwords.words("english")
wnl = WordNetLemmatizer()

# --- 2. Loading Artifacts (Cached for Performance) ---
@st.cache_resource
def load_model_artifacts():
    """
    Loads the model, tokenizer, label encoder, and mapping table.
    Expects files to be in the same directory or specific paths.
    """
    try:
        # Load Model and Tokenizer
        # NOTE: Ensure './saved_biobert_model' exists from Step 1
        model_path = "./saved_biobert_model" 
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Move model to CPU for inference (safer for basic web apps)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load Label Encoder
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)

        # Load Group-to-Disease Mapping
        mapping_df = pd.read_csv("group_disease_mapping.csv")
        
        return model, tokenizer, le, mapping_df, device
        
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.warning("Did you run the 'Step 1' code to save your model files first?")
        return None, None, None, None, None

model, tokenizer, le, mapping_df, device = load_model_artifacts()

# --- 3. Preprocessing Function ---
def preprocess_text(text):
    """
    Same preprocessing logic as used during training.
    """
    if not text:
        return ""
    words = word_tokenize(text.lower())
    # Keep only alphabetic words and remove stopwords
    words = [wnl.lemmatize(word) for word in words if word.isalpha() and word not in eng_stopwords]
    return " ".join(words)

# --- 4. Prediction Logic ---
def predict_disease(text, model, tokenizer, le, device):
    # 1. Preprocess
    processed_text = preprocess_text(text)
    
    if not processed_text:
        return None, None
        
    # 2. Tokenize
    inputs = tokenizer(
        processed_text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=256
    )
    
    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 3. Inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 4. Get Prediction
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence, predicted_class_idx = torch.max(probs, dim=1)
    
    predicted_label = le.inverse_transform([predicted_class_idx.item()])[0]
    
    return predicted_label, confidence.item()

# --- 5. User Interface ---
st.title("🩺 BioBERT Symptom Prediction ")
st.markdown("----")
st.markdown("Describe your symptoms as detailed as possible below, and the AI will analyze which disease group matches best.")

# Input Area
user_input = st.text_area("Enter Symptoms:", height=150, placeholder="e.g., I have a high fever, severe headache, and stiff neck...")

predict_btn = st.button("Analyze Symptoms", type="primary")

if predict_btn and user_input:
    if model is None:
        st.error("Model not loaded. Please ensure artifact files are present.")
    else:
        with st.spinner("Analyzing with BioBERT..."):
            predicted_group, confidence = predict_disease(user_input, model, tokenizer, le, device)
            
            if predicted_group:
                # Display Results
                st.success(f"Analysis Complete")
                
                # Main Prediction
                st.subheader(f"Predicted Category: {predicted_group}")
                st.progress(confidence)
                st.caption(f"Model Confidence: {confidence*100:.2f}%")
                
                # Show specific diseases in this group
                st.divider()
                st.markdown("### Possible conditions in this category:")
                
                # Filter diseases belonging to this group
                possible_diseases = mapping_df[mapping_df['GroupedDisease'] == predicted_group]['Disease'].unique()
                
                if len(possible_diseases) > 0:
                    for disease in possible_diseases:
                        st.markdown(f"- **{disease}**")
                else:
                    st.write("No specific disease mapping found for this group.")
                    
            else:
                st.warning("Could not extract meaningful symptoms from input. Please try describing more details.")

elif predict_btn and not user_input:
    st.warning("Please enter some symptoms first.")

# Sidebar info
with st.sidebar:
    st.image("digisease.png")
    st.info("This model uses BioBERT (dmis-lab/biobert-base-cased-v1.1) fine-tuned on medical symptom data.")
    st.markdown("---")
    st.write("⚠️ **Disclaimer:** This tool is for educational purposes only and does not constitute medical advice. Made by: Richard - 2602170045")