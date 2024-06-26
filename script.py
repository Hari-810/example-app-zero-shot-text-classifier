import streamlit as st
from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

st.title("Zero-Shot Classification with Hugging Face")

# User input for the sequence to classify
sequence_to_classify = st.text_area("Enter the text you want to classify:", "The movie was fantastic with great performances and direction.")

# User input for the candidate labels
candidate_labels = st.text_input("Enter the candidate labels separated by commas:", "positive, negative, neutral")

# Convert the candidate labels to a list
candidate_labels = [label.strip() for label in candidate_labels.split(",")]

if st.button("Classify"):
    # Perform zero-shot classification
    result = classifier(sequence_to_classify, candidate_labels)
    
    # Find the label with the highest score
    highest_score_index = result["scores"].index(max(result["scores"]))
    highest_label = result["labels"][highest_score_index]
    highest_score = result["scores"][highest_score_index]
    
    # Display the results
    st.subheader("Results")
    for label, score in zip(result["labels"], result["scores"]):
        if label == highest_label:
            st.success(f"{label}: {score:.4f}")
        else:
            st.write(f"{label}: {score:.4f}")

# To run the app, save this script and use the following command in your terminal:
# streamlit run zero_shot_app.py
