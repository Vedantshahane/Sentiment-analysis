from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    return "Positive" if pred == 1 else "Negative"

review = input("Enter a movie review: ")
print("Sentiment:", predict_sentiment(review))
