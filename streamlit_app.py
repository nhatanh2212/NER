import streamlit as st
from numpy import arange
import joblib
import datetime as dt
import pandas as pd
import numpy as np
from collections import Counter
import streamlit as st
# from sklearn_crfsuite import CRF, metrics
import re

LABEL_COLORS = {
    "PER": "blue",
    "LOC": "green",
    "DTM": "purple",
    "ORG": "red",
    "TITLE": "orange",
    "DOC": "teal",
    "O": "gray"
}
# Show title and description.
st.markdown(
    """
    <style>
    .title {
        color: #054279; 
        font-size: 2em; 
        text-align: left; 

        margin-bottom: 20px;
    }
    .section-title {
        color: #054279; 
        font-size: 1.5em; 
        margin-top: 30px; 
        margin-bottom: 10px;
    }
    .text {
        font-size: 1.1em; 
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


col1, col2 = st.columns([1,4], vertical_alignment="center",gap="large")
col1.image("logo.jpg", width=120)
col2.markdown(
            '<div class="title">ENHANCING NAMED ENTITY RECOGNITION FOR VIETNAMESE PROSE WRITTEN IN CHINESE USING FINETUNE CRF </div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Named Entity Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="text">Named Entity Recognition (NER) is an essential task in the field of Natural Language Processing (NLP). The goal of NER is to identify specific entities with meaningful names from the text, such as person names, location names, organization names, dates, times, currencies, etc. NER finds widespread applications in information extraction, question-answering systems, machine translation, text classification, and more. By recognizing named entities in the text, it helps in understanding the meaning of the text, extracting key information, and enabling more in-depth semantic analysis and reasoning.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Named Entity Recognition For Traditional Vietnamese Prose Written in Chinese</div>', unsafe_allow_html=True)
st.markdown('<div class="text">Named Entity Recognition (NER) for traditional Vietnamese prose written in Chinese involves automatically identifying and extracting specific meaningful entity names from the text. These entities can include names of historical figures, locations, literary works, and cultural terms within Vietnamese literature composed in the Chinese language. Implementing NER in this context supports automated processing, search capabilities, and analytical tasks, thereby enriching scholarly research, cultural preservation efforts, and linguistic studies related to Vietnamese literature in its historical Chinese form.</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Entity Tag</div>', unsafe_allow_html=True)

df = pd.DataFrame({
    "Entity": ["Person(PER)","Location (LOC)","Time (DTM)", "Organization(ORG)", "Title(TITLE)", "Document(DOC)", "Other(O)"],
    "Introduction": ["Person entities include historical figures, writers, politicians, and other individuals mentioned in documents, playing important roles in various events.", 
                     " Location entities refer to geographical locations, regions, and buildings that are crucial for setting the context in documents.",
                     "Time entities involve specific dates, eras, and times, helping to build the timeline and context of events in Vietnamese documents.",
                     "Organization entities cover entities such as government, social groups, influencing different aspects of society",
                     "Title entities include names of books, articles, and poems",
                     "Document entities refer to texts, records, and books that serve as media for knowledge and cultural heritage in Vietnamese documents.",
                     "Other text not in the label"
                     ]
})

st.table(df)

model = joblib.load('crf_model.joblib')

def extract_features(sequence):
    def word_shape(word):
        if word.isupper():
            return 'UPPER'
        elif word.islower():
            return 'LOWER'
        elif word.istitle():
            return 'TITLE'
        else:
            return 'MIXED'

    return [{'char': char,
             'is_upper': char.isupper(),
             'is_digit': char.isdigit(),
             'is_punctuation': char in '!?,.:;"\'()',
             'prev_char': '' if i == 0 else sequence[i-1],
             'next_char': '' if i == len(sequence)-1 else sequence[i+1],
             'char_position': i,
             'word_shape': word_shape(char)}
            for i, char in enumerate(sequence)]

def predict_labels(model, input_line):
    features = extract_features(list(input_line))
    pred_labels = model.predict_single(features)

    combined_results = []
    current_label = None
    current_word = ''

    for char, label in zip(input_line, pred_labels):
        if label == current_label:
            current_word += char
        else:
            if current_word:
                combined_results.append((current_word, current_label))
            current_word = char
            current_label = label

    if current_word:
        combined_results.append((current_word, current_label))

    return combined_results


def preprocess_data(text):

    text = text.replace(" ", "")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences


def format_output(sentence, predicted_labels):
    formatted_sentence = sentence
    for label_pair in predicted_labels:
        entity, label = label_pair
        # Get the color for the label
        color = LABEL_COLORS.get(label, "black")
        # Highlight the entity with the corresponding color
        formatted_sentence = formatted_sentence.replace(entity, f"<span style='color:{color}; font-weight:bold;'>{entity} ({label})</span>")
    return formatted_sentence

def main():
    raw_data = st.text_area("Enter Input Text here")
    if st.button("Show Entities"):
        if raw_data == '':
            st.warning("Sorry, Please input your data to access this functionality!!")
        else:
            preprocessed_data = preprocess_data(raw_data)
            for sentence in preprocessed_data:
                sentence = sentence.replace(".", "")
                predicted_labels = predict_labels(model, sentence)
                formatted_sentence = format_output(sentence, predicted_labels)
                st.markdown(formatted_sentence, unsafe_allow_html=True)
                
main()
