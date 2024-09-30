import torch
from transformers import BertTokenizer, BertForMultipleChoice
from torch.nn.functional import softmax
import numpy as np


def load_model():
    # Load pre-trained BERT tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMultipleChoice.from_pretrained(model_name)
    return tokenizer, model

def process_mcq(tokenizer, question, choices):
    """
    Prepare inputs for BERT model from a question and multiple choices.
    """
    inputs = tokenizer([f"{question} {choice}" for choice in choices],
                        padding=True, truncation=True, max_length=256, return_tensors="pt")
    input_ids = inputs["input_ids"].unsqueeze(0)  # Add batch dimension
    attention_mask = inputs["attention_mask"].unsqueeze(0)  # Add batch dimension
    return input_ids, attention_mask


def predict_mcq(model, tokenizer, question, choices):
    """
    Predict the most likely answer for the multiple-choice question.
    """
    model.eval()
    input_ids, attention_mask = process_mcq(tokenizer, question, choices)

    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = softmax(logits, dim=1).squeeze()

    # Get the predicted answer
    predicted_idx = torch.argmax(probs).item()
    confidence = probs[predicted_idx].item()
    return choices[predicted_idx], confidence



def mcq_solver(question, choices):
    """
    Main function to solve MCQ using BERT model.
    """
    tokenizer, model = load_model()  # Load the model and tokenizer
    answer, confidence = predict_mcq(model, tokenizer, question, choices)
    
    print(f"\nQuestion: {question}")
    print("Choices:")
    for idx, choice in enumerate(choices):
        print(f"  {idx + 1}. {choice}")
    
    print(f"\nPredicted Answer: {answer} (Confidence: {confidence:.2f})")



if __name__ == "__main__":
    # Sample MCQ
    question = "What is the capital of France?"
    choices = ["Berlin", "Paris", "London", "Madrid"]
    
    # Solve the MCQ using BERT
    mcq_solver(question, choices)
