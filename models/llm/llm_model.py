# load and train the large language model

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_llm_model(model_name='bert-base-uncased'):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def preprocess_data(tokenizer, texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

def train_llm_model(model, tokenizer, train_texts, train_labels):
    inputs = preprocess_data(tokenizer, train_texts)
    labels = torch.tensor(train_labels)

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    return model
