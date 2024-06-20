import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Cache the model and tokenizer to avoid reloading them on every run
@st.cache_resource
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("Shokouhi/YTFineTuneBert")
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = get_model()

# User input for text analysis
user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

# Dictionary for interpreting the model's output
label_dict = {
    1: 'Toxic',
    0: 'Non Toxic'
}

# Perform analysis when the button is clicked and user_input is provided
if user_input and button:
    # Tokenize the input text
    test_sample = tokenizer(
        [user_input],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Get the model's output
    output = model(**test_sample)
    
    # Display the raw logits
    st.write("Logits: ", output.logits)
    
    # Determine the prediction (0 or 1)
    y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
    
    # Display the prediction
    st.write("Prediction: ", label_dict[y_pred[0]])

    # For better readability, add a message for prediction
    st.success(f"The text is classified as: {label_dict[y_pred[0]]}")
