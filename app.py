import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

st.title("IMDb Sentiment Analyzer")

text = st.text_area("Enter your movie review:")

if st.button("Analyze"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    st.write("Positive 😊" if pred == 1 else "Negative 😡")
