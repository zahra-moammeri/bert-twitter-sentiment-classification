import numpy as np
import pandas as pd
import streamlit as st

from transformers import pipeline


st.title("Fine Tuning Bert for Twitter Multiclass Sentiment Classification")

classifier = pipeline("text-classification", "bert-base-uncased-sentiment-model")

text = st.text_area("Enter yout tweet here: ")

if st.button("predict"):
    result = classifier(text)
    st.write(f"I say it is '{result[0]['label']}'")
    score = result[0]['score'] * 100
    st.write(f"I am sure '%{score:.2f}'")