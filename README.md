# TEXT-GENERATION-LSTM-GPT2_ASSIGNMENT_4


**COMPANY**: CODTECH IT Solutions

**NAME**: Pratik Mitkar

**INTERN ID**: CT08GJB

**DOMAIN**: Artificial Intelligence

**BATCH DURATION**: January 5th, 2025 to February 5th, 2025

**MENTOR NAME**: Neela Santhosh

## PROJECT DESCRIPTION

The **Text Generation with LSTM and GPT-2** project utilizes two deep learning models—LSTM (Long Short-Term Memory) and GPT-2 (Generative Pretrained Transformer 2)—to generate human-like text. The LSTM model is trained on user-uploaded text, while GPT-2 generates text based on a given prompt. The project provides a user-friendly web interface built with Streamlit, enabling real-time text generation using both models.

The core of the project involves two separate text generation techniques:
1. **LSTM Text Generation**: A custom-trained LSTM model generates text based on a seed or initial input from the user.
2. **GPT-2 Text Generation**: The GPT-2 model from Hugging Face generates text based on a user-specified prompt.

### Key Features:
1. **Text Upload for LSTM**: Users can upload a `.txt` file to train the LSTM model with their custom text.
2. **Pre-trained GPT-2 Model**: The GPT-2 model from Hugging Face is used for generating text based on user prompts.
3. **Streamlit Web Interface**: The project provides a clean and simple interface where users can interact with both models.
4. **Real-Time Text Generation**: Both models generate text in real-time and display the results instantly on the web interface.

### Development Process:
1. **Text Preprocessing**: Implemented functions to load, tokenize, and preprocess text data to train the LSTM model.
2. **LSTM Training**: Built and trained an LSTM model using TensorFlow to generate text based on the input seed.
3. **GPT-2 Integration**: Integrated Hugging Face's GPT-2 model to generate text based on a given prompt.
4. **Streamlit Web Interface**: Developed an intuitive interface to upload text files, input prompts, and display generated text in real-time.

### Skills Gained:
- **Deep Learning with LSTM**: Gained hands-on experience with LSTM models for text generation.
- **Using GPT-2 for Text Generation**: Integrated GPT-2, a pre-trained transformer model, for efficient text generation.
- **Python and Streamlit**: Improved skills in building interactive web applications with Streamlit and TensorFlow.
- **Natural Language Processing (NLP)**: Worked with tokenization, sequence generation, and text generation techniques.

### Future Enhancements:
- **User Controls for Text Generation**: Allow users to customize the generation process by controlling parameters like temperature and the number of words.
- **Additional Text Generation Models**: Support other pre-trained models like GPT-3 for enhanced text generation capabilities.
- **Fine-tuning LSTM**: Allow users to fine-tune the LSTM model with their own datasets for personalized text generation.

## Installation Instructions:

To run the Text Generation System, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PratikMitkar/Text-Generation-LSTM-GPT2.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd Text-Generation-LSTM-GPT2
   ```

3. **Install Required Packages**:
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Application**:
   Start the application with the following command:
   ```bash
   streamlit run task4_1.py
   ```

5. **Access the Application**:
   Open your web browser and go to `http://localhost:8501` to access the text generation interface.


## Acknowledgments

- Thanks to the CODTECH IT Solutions team for their support and guidance.
- Special thanks to Neela Santhosh for mentorship throughout the project.
