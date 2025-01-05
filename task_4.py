import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import streamlit as st
from transformers import pipeline

# Streamlit UI
st.title("Text Generation with LSTM and GPT-2")

# Text Upload Section
st.header("Upload Training Text for LSTM")
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    st.success("Text uploaded and preprocessed successfully!")

    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    # Create sequences for text generation
    input_sequences = []
    for line in text.split('.'):  # Splitting into lines
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # Split into inputs and labels
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    # Build the LSTM model
    model = Sequential()
    model.add(Embedding(total_words, 128, input_length=max_sequence_len - 1))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model (optional due to time constraints)
    if st.button("Train LSTM Model"):
        with st.spinner("Training in progress..."):
            model.fit(X, y, epochs=50, verbose=1)
        st.success("Model trained successfully!")

    # Text generation function
    def generate_text_lstm(seed_text, next_words, max_sequence_len, temperature=1.0):
        if not seed_text:  # Check for empty seed text
            return "Seed text cannot be empty."
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            predictions = model.predict(token_list, verbose=0)[0]

            # Apply temperature
            predictions = np.log(predictions + 1e-7) / temperature if temperature > 0 else predictions
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

            predicted_index = np.random.choice(len(predictions), p=predictions)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
            seed_text += " " + output_word
        return seed_text

    # Generate text using LSTM
    st.header("Generate Text with LSTM")
    seed_text = st.text_input("Enter a seed text:", "Once upon a time")
    num_words = st.slider("Number of words to generate:", 1, 50, 20)
    if st.button("Generate Text with LSTM"):
        generated_text = generate_text_lstm(seed_text, num_words, max_sequence_len)
        st.text_area("Generated Text:", generated_text, height=200)

# GPT-2 Text Generation Section
st.header("Generate Text with GPT-2")
text_prompt = st.text_input("Enter a prompt for GPT-2:", "OpenAI is transforming")
if st.button("Generate Text with GPT-2"):
    generator = pipeline(task='text-generation', model='gpt2')
    generated = generator(text_prompt, max_length=100, num_return_sequences=1)
    st.text_area("Generated Text with GPT-2:", generated[0]['generated_text'], height=200)
